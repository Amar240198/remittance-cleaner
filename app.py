import io
import re
import time
from typing import Dict, Tuple, Optional

import pandas as pd
import streamlit as st

import pdfplumber
from PIL import Image
import pytesseract


# =========================
# SAFETY LIMITS
# =========================
MAX_PARSE_SECONDS = 15


# =========================
# UTILITIES
# =========================

def parse_money(x: str) -> Optional[float]:
    if not x:
        return None
    x = x.replace("£", "").replace("€", "").replace("$", "")
    x = x.replace(",", "").strip()
    try:
        return float(x)
    except:
        return None


def normalize_invoice(inv: str) -> str:
    inv = re.sub(r"[A-Z]", "", inv.upper())
    return inv.lstrip("0")


def find_invoices(text: str):
    raw = re.findall(r"\b(?:INV)?0*\d{5,12}\b", text.upper())
    return [normalize_invoice(r) for r in raw]


# =========================
# PDF EXTRACTION (ROBUST)
# =========================

def extract_pdf_text_safe(file_bytes: bytes) -> Tuple[str, float]:
    """
    1) Try pdfplumber
    2) If weak or fails → OCR fallback
    Returns text + confidence
    """
    start = time.time()

    # --- Attempt 1: pdfplumber ---
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts).strip()

        if len(text) > 200:
            return text, 0.9

    except Exception:
        pass  # silently fallback

    # --- Attempt 2: OCR fallback ---
    try:
        images = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                images.append(page.to_image(resolution=300).original)

        ocr_text = []
        for img in images:
            if time.time() - start > MAX_PARSE_SECONDS:
                raise TimeoutError
            ocr_text.append(
                pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            )

        text = "\n".join(ocr_text).strip()
        if text:
            return text, 0.6

    except Exception:
        pass

    return "", 0.0


# =========================
# EXCEL EXTRACTION
# =========================

def extract_excel_safe(upload) -> Tuple[pd.DataFrame, float]:
    try:
        df = pd.read_excel(upload)
        return df, 0.95
    except Exception:
        return pd.DataFrame(), 0.0


# =========================
# GENERIC UPLOAD HANDLER
# =========================

def extract_upload(upload) -> Tuple[str | pd.DataFrame, float, str]:
    if upload is None:
        return "", 0.0, "none"

    name = upload.name.lower()
    data = upload.read()

    if name.endswith(".xlsx"):
        df, conf = extract_excel_safe(io.BytesIO(data))
        return df, conf, "excel"

    if name.endswith(".pdf"):
        text, conf = extract_pdf_text_safe(data)
        return text, conf, "pdf"

    try:
        text = data.decode("utf-8", errors="ignore")
        return text, 0.8, "text"
    except:
        return "", 0.0, "unknown"


# =========================
# PARSERS
# =========================

def parse_statement(text: str) -> Dict[str, float]:
    result = {}
    for line in text.splitlines():
        invoices = find_invoices(line)
        amounts = re.findall(r"\d+\.\d{2}", line)
        if invoices and amounts:
            val = parse_money(amounts[-1])
            for inv in invoices:
                result[inv] = val
    return result


def parse_remittance(text: str) -> Dict[str, float]:
    result = {}
    for line in text.splitlines():
        invoices = find_invoices(line)
        amounts = re.findall(r"\d+\.\d{2}", line)
        if invoices and amounts:
            val = parse_money(amounts[-1])
            for inv in invoices:
                result[inv] = round(result.get(inv, 0) + val, 2)
    return result


def parse_excel(df: pd.DataFrame) -> Dict[str, float]:
    result = {}
    for _, row in df.iterrows():
        row_text = " ".join(map(str, row.values))
        invoices = find_invoices(row_text)
        amounts = [parse_money(x) for x in row.values if isinstance(x, (int, float, str))]
        amounts = [a for a in amounts if a is not None]
        if invoices and amounts:
            for inv in invoices:
                result[inv] = float(amounts[-1])
    return result


# =========================
# RECONCILIATION
# =========================

def reconcile(stmt_map, remit_map, stmt_conf, remit_conf):
    rows = []
    invoices = sorted(set(stmt_map) | set(remit_map))

    for inv in invoices:
        expected = stmt_map.get(inv)
        paid = remit_map.get(inv, 0.0)

        diff = None
        status = "MISSING FROM STATEMENT"

        if expected is not None:
            diff = round(paid - expected, 2)
            if paid == 0:
                status = "NOT PAID"
            elif abs(diff) < 0.01:
                status = "FULLY PAID"
            elif diff < 0:
                status = "UNDERPAID"
            else:
                status = "OVERPAID"

        confidence = round(min(stmt_conf, remit_conf), 2)

        rows.append({
            "Invoice_Number": inv,
            "Expected_Total": expected,
            "Paid_Total": paid,
            "Difference": diff,
            "Status": status,
            "Confidence": confidence
        })

    return pd.DataFrame(rows)


# =========================
# STREAMLIT UI
# =========================

st.set_page_config("Remittance Cleaner — MVP", layout="wide")
st.title("Remittance Cleaner — MVP")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text")
    st.markdown("**or**")
    remit_file = st.file_uploader("Upload remittance file")

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text")
    st.markdown("**or**")
    stmt_file = st.file_uploader("Upload statement file")

st.divider()

if st.button("Run Reconciliation"):
    try:
        stmt_content, stmt_conf, stmt_type = extract_upload(stmt_file)
        remit_content, remit_conf, remit_type = extract_upload(remit_file)

        if isinstance(stmt_content, pd.DataFrame):
            stmt_map = parse_excel(stmt_content)
        else:
            stmt_map = parse_statement(stmt_content or stmt_text)

        if isinstance(remit_content, pd.DataFrame):
            remit_map = parse_excel(remit_content)
        else:
            remit_map = parse_remittance(remit_content or remit_text)

        if not stmt_map or not remit_map:
            st.warning(
                "⚠️ We couldn’t reliably read one of the inputs.\n\n"
                "For best results, copy & paste the text directly."
            )
            st.stop()

        df = reconcile(stmt_map, remit_map, stmt_conf, remit_conf)
        st.success("Reconciliation complete")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "reconciliation_output.csv",
            "text/csv"
        )

    except TimeoutError:
        st.error(
            "⏱️ Processing timed out.\n\n"
            "Try smaller files or copy & paste the text instead."
        )

    except Exception:
        st.error(
            "This app encountered an error while parsing.\n\n"
            "Try copy/paste instead of upload."
        )
