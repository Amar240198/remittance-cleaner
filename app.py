import io
import re
import time
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import streamlit as st
import pdfplumber
from pypdf import PdfReader


# =========================
# CONFIG / SAFETY LIMITS
# =========================
MAX_PARSE_SECONDS = 15
MIN_TEXT_LEN_OK = 150


# =========================
# UTILITIES
# =========================
def parse_money(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).replace("£", "").replace("€", "").replace("$", "")
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def normalize_invoice(inv: str) -> str:
    inv = re.sub(r"[A-Z]", "", inv.upper())
    return inv.lstrip("0")


def find_invoices(text: str):
    raw = re.findall(r"\b(?:INV)?0*\d{5,12}\b", text.upper())
    return [normalize_invoice(r) for r in raw]


# =========================
# PDF EXTRACTION (CLOUD SAFE)
# =========================
def extract_pdf_text_safe(file_bytes: bytes) -> str:
    deadline = time.time() + MAX_PARSE_SECONDS

    # --- Attempt 1: pypdf ---
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            if time.time() > deadline:
                raise TimeoutError
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        if len(text) >= MIN_TEXT_LEN_OK:
            return text
    except Exception:
        pass

    # --- Attempt 2: pdfplumber ---
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                if time.time() > deadline:
                    raise TimeoutError
                parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        if len(text) >= MIN_TEXT_LEN_OK:
            return text
    except Exception:
        pass

    return ""


# =========================
# EXCEL EXTRACTION
# =========================
def extract_excel_safe(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")


# =========================
# GENERIC UPLOAD HANDLER
# =========================
def extract_upload(upload) -> Tuple[Union[str, pd.DataFrame], str]:
    if upload is None:
        return "", "none"

    name = upload.name.lower()
    data = upload.read()

    if name.endswith(".xlsx"):
        return extract_excel_safe(data), "excel"

    if name.endswith(".pdf"):
        return extract_pdf_text_safe(data), "pdf"

    try:
        return data.decode("utf-8", errors="ignore"), "text"
    except Exception:
        return "", "unknown"


# =========================
# PARSERS
# =========================
def parse_statement(text: str) -> Dict[str, float]:
    result = {}
    for line in text.splitlines():
        invoices = find_invoices(line)
        amounts = re.findall(r"\d+\.\d{2}", line)
        if invoices and amounts:
            expected = parse_money(amounts[-1])
            for inv in invoices:
                result[inv] = expected
    return result


def parse_remittance(text: str) -> Dict[str, float]:
    result = {}
    for line in text.splitlines():
        invoices = find_invoices(line)
        amounts = re.findall(r"\d+\.\d{2}", line)
        if invoices and amounts:
            paid = parse_money(amounts[-1])
            for inv in invoices:
                result[inv] = round(result.get(inv, 0.0) + (paid or 0.0), 2)
    return result


def parse_excel(df: pd.DataFrame) -> Dict[str, float]:
    result = {}
    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row.values)
        invoices = find_invoices(row_text)
        numbers = [parse_money(x) for x in row.values if parse_money(x) is not None]
        if invoices and numbers:
            for inv in invoices:
                result[inv] = numbers[-1]
    return result


# =========================
# RECONCILIATION
# =========================
def reconcile(stmt_map, remit_map):
    rows = []
    invoices = sorted(set(stmt_map) | set(remit_map))

    for inv in invoices:
        expected = stmt_map.get(inv)
        paid = remit_map.get(inv, 0.0)

        if expected is None:
            rows.append({
                "Invoice_Number": inv,
                "Expected_Total": None,
                "Paid_Total": paid,
                "Difference": None,
                "Status": "MISSING FROM STATEMENT",
            })
            continue

        diff = round(paid - expected, 2)

        if paid == 0:
            status = "NOT PAID"
        elif abs(diff) < 0.01:
            status = "FULLY PAID"
            diff = 0.0
        elif diff < 0:
            status = "UNDERPAID"
        else:
            status = "OVERPAID"

        rows.append({
            "Invoice_Number": inv,
            "Expected_Total": expected,
            "Paid_Total": paid,
            "Difference": diff,
            "Status": status,
        })

    return pd.DataFrame(rows)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")
st.title("Remittance Cleaner — MVP")
st.caption("Expected comes from Statement | Paid comes from Remittance | Difference = Paid − Expected")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=180)
    st.markdown("**or**")
    remit_file = st.file_uploader("Upload remittance file (PDF / XLSX / TXT)")

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=180)
    st.markdown("**or**")
    stmt_file = st.file_uploader("Upload statement file (PDF / XLSX / TXT)")

st.divider()

if st.button("Run Reconciliation", type="primary"):
    try:
        stmt_content, stmt_type = extract_upload(stmt_file)
        remit_content, remit_type = extract_upload(remit_file)

        if isinstance(stmt_content, pd.DataFrame):
            stmt_map = parse_excel(stmt_content)
        else:
            stmt_map = parse_statement((stmt_content or "") + "\n" + (stmt_text or ""))

        if isinstance(remit_content, pd.DataFrame):
            remit_map = parse_excel(remit_content)
        else:
            remit_map = parse_remittance((remit_content or "") + "\n" + (remit_text or ""))

        if not stmt_map or not remit_map:
            st.warning(
                "⚠️ We couldn’t reliably read one of the inputs.\n\n"
                "If the PDF is scanned/image-only, please copy & paste the text."
            )
            st.stop()

        df = reconcile(stmt_map, remit_map)

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
            "Try smaller files or copy/paste instead."
        )

    except Exception:
        st.error(
            "This app encountered an error while parsing.\n\n"
            "Try copy/paste instead of upload."
        )
