import io
import re
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract
from pdfminer.pdfparser import PDFSyntaxError

# =========================
# UTILITIES
# =========================

def parse_money(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.replace("£", "").replace("€", "").replace("$", "")
    text = text.replace(",", "").strip()
    try:
        return float(text)
    except:
        return None


def normalize_invoice(inv: str) -> str:
    inv = re.sub(r"[A-Z]", "", inv.upper())
    return inv.lstrip("0")


def find_invoices(text: str):
    raw = re.findall(r"\b(?:INV)?0*\d{5,12}\b", text.upper())
    return [normalize_invoice(r) for r in raw]


# =========================
# SAFE FILE EXTRACTION
# =========================

def extract_from_pdf(file_bytes: bytes) -> Tuple[str, float]:
    try:
        text = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        combined = "\n".join(text).strip()
        return combined, 0.85 if combined else 0.3
    except PDFSyntaxError:
        return "", 0.0
    except Exception:
        return "", 0.0


def extract_upload(upload):
    if upload is None:
        return "", 0.0, "empty"

    data = upload.read()
    name = upload.name.lower()

    if name.endswith(".pdf"):
        text, conf = extract_from_pdf(data)
        return text, conf, "pdf" if text else "pdf_failed"

    if name.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
        return text.strip(), 0.6 if text.strip() else 0.2, "image"

    if name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(data))
        return df, 0.95, "excel"

    try:
        return data.decode("utf-8", errors="ignore"), 0.9, "text"
    except:
        return "", 0.0, "empty"


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
                result[inv] = round(result.get(inv, 0) + paid, 2)
    return result


def parse_excel(df: pd.DataFrame) -> Dict[str, float]:
    mapping = {}
    for _, row in df.iterrows():
        for cell in row.astype(str):
            invs = find_invoices(cell)
            amts = re.findall(r"\d+\.\d{2}", cell)
            if invs and amts:
                val = parse_money(amts[-1])
                for inv in invs:
                    mapping[inv] = val
    return mapping


# =========================
# RECONCILIATION
# =========================

def reconcile(statement: Dict[str, float], remittance: Dict[str, float], confidence: float) -> pd.DataFrame:
    rows = []
    invoices = sorted(set(statement) | set(remittance))

    for inv in invoices:
        expected = statement.get(inv)
        paid = remittance.get(inv, 0.0)

        if expected is None:
            rows.append({
                "Invoice_Number": inv,
                "Expected_Total": None,
                "Paid_Total": paid,
                "Difference": None,
                "Status": "MISSING FROM STATEMENT",
                "Confidence": round(confidence, 2)
            })
            continue

        diff = round(paid - expected, 2)

        if paid == 0:
            status = "NOT PAID"
        elif abs(diff) < 0.01:
            status = "FULLY PAID"
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
            "Confidence": round(confidence, 2)
        })

    return pd.DataFrame(rows)


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")

st.title("Remittance Cleaner — MVP")
st.caption("No data stored • Statement = Expected • Remittance = Paid")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=200)
    st.markdown("**or**")
    remit_file = st.file_uploader("Upload remittance file", type=["pdf", "png", "jpg", "xlsx"])

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=200)
    st.markdown("**or**")
    stmt_file = st.file_uploader("Upload statement file", type=["pdf", "png", "jpg", "xlsx"])

st.divider()

if st.button("Run Reconciliation"):
    stmt_content, stmt_conf, stmt_type = extract_upload(stmt_file)
    remit_content, remit_conf, remit_type = extract_upload(remit_file)

    if stmt_type == "pdf_failed" or remit_type == "pdf_failed":
        st.warning(
            "⚠️ We couldn’t reliably read one of the uploaded files.\n\n"
            "Best results: **copy & paste text** or upload **Excel (.xlsx)**."
        )

    if isinstance(stmt_content, pd.DataFrame):
        stmt_map = parse_excel(stmt_content)
    else:
        stmt_map = parse_statement(stmt_content + "\n" + stmt_text)

    if isinstance(remit_content, pd.DataFrame):
        remit_map = parse_excel(remit_content)
    else:
        remit_map = parse_remittance(remit_content + "\n" + remit_text)

    if not stmt_map or not remit_map:
        st.error("No matching invoices found. Please copy & paste or upload Excel.")
        st.stop()

    confidence = min(stmt_conf or 0.8, remit_conf or 0.8)

    df = reconcile(stmt_map, remit_map, confidence)

    st.success("Reconciliation complete")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "reconciliation_output.csv",
        "text/csv"
    )
