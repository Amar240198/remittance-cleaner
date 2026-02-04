import re
import io
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract

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
# FILE EXTRACTION
# =========================

def extract_from_pdf(file_bytes: bytes) -> Tuple[str, float]:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)

    if text:
        return "\n".join(text), 0.75

    # OCR fallback
    images_text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=300).original
            images_text.append(
                pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            )

    if images_text:
        return "\n".join(images_text), 0.60

    return "", 0.0


def extract_from_image(file_bytes: bytes) -> Tuple[str, float]:
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6"), 0.60


def extract_from_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, float]:
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df, 0.95


def extract_upload(upload):
    if not upload:
        return None, 0.0, None

    data = upload.read()
    name = upload.name.lower()

    if name.endswith(".pdf"):
        text, conf = extract_from_pdf(data)
        return text, conf, "pdf"

    if name.endswith((".png", ".jpg", ".jpeg")):
        text, conf = extract_from_image(data)
        return text, conf, "image"

    if name.endswith(".xlsx"):
        df, conf = extract_from_excel(data)
        return df, conf, "excel"

    return "", 0.0, None


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
        row_text = " ".join(str(x) for x in row.values)
        invoices = find_invoices(row_text)
        amounts = re.findall(r"\d+\.\d{2}", row_text)
        if invoices and amounts:
            val = parse_money(amounts[-1])
            for inv in invoices:
                result[inv] = val
    return result


# =========================
# RECONCILIATION
# =========================

def reconcile(statement, remittance, confidence) -> pd.DataFrame:
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

st.set_page_config("Remittance Cleaner — MVP", layout="wide")

st.title("Remittance Cleaner — MVP")
st.caption("Statement = expected | Remittance = paid | Difference calculated")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=200)
    st.markdown("**— OR —**")
    remit_file = st.file_uploader("Upload remittance file (PDF / Excel)")

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=200)
    st.markdown("**— OR —**")
    stmt_file = st.file_uploader("Upload statement file (PDF / Excel)")

st.divider()

if st.button("Run Reconciliation"):
    stmt_conf = remit_conf = 0.0

    stmt_content, stmt_conf, stmt_type = extract_upload(stmt_file)
    remit_content, remit_conf, remit_type = extract_upload(remit_file)

    stmt_map = {}
    remit_map = {}

    if isinstance(stmt_content, pd.DataFrame):
        stmt_map = parse_excel(stmt_content)
    else:
        stmt_map = parse_statement((stmt_content or "") + "\n" + stmt_text)

    if isinstance(remit_content, pd.DataFrame):
        remit_map = parse_excel(remit_content)
    else:
        remit_map = parse_remittance((remit_content or "") + "\n" + remit_text)

    if not stmt_map or not remit_map:
        st.warning(
            "⚠️ We couldn't reliably read one of the uploaded files.\n\n"
            "For best results, **copy & paste the text** instead."
        )

    if not stmt_map or not remit_map:
        st.stop()

    confidence = min(stmt_conf or 0.9, remit_conf or 0.9)

    df = reconcile(stmt_map, remit_map, confidence)

    st.success("Reconciliation complete")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "reconciliation_output.csv",
        "text/csv"
    )
