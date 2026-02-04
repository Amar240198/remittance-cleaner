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
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Remittance Cleaner — Private Beta",
    layout="wide",
)

st.title("Remittance Cleaner — MVP (Private Beta)")
st.caption("Statement = Expected • Remittance = Paid • Difference calculated")

# =========================
# UTILITIES
# =========================

def parse_money(text: str) -> Optional[float]:
    if not text:
        return None
    text = (
        text.replace("£", "")
        .replace("€", "")
        .replace("$", "")
        .replace(",", "")
        .strip()
    )
    try:
        return float(text)
    except ValueError:
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

def extract_text_from_pdf(data: bytes) -> str:
    try:
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except PDFSyntaxError:
        return ""
    except Exception:
        return ""


def extract_text_from_image(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    except Exception:
        return ""


def extract_upload(upload) -> Tuple[str, float, bool]:
    """
    Returns (text, confidence, success)
    """
    if upload is None:
        return "", 0.0, False

    data = upload.read()
    name = upload.name.lower()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf(data)
        return text, 0.85 if text else 0.0, bool(text)

    if name.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(data)
        return text, 0.6 if text else 0.0, bool(text)

    if name.endswith(".xlsx"):
        try:
            df = pd.read_excel(io.BytesIO(data))
            return df.to_string(), 0.95, True
        except Exception:
            return "", 0.0, False

    try:
        text = data.decode("utf-8", errors="ignore")
        return text, 0.9, bool(text.strip())
    except Exception:
        return "", 0.0, False


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
                result[inv] = round(result.get(inv, 0.0) + paid, 2)
    return result


# =========================
# RECONCILIATION (NO REMAINING BALANCE)
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
                "Confidence": confidence,
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
            "Confidence": confidence,
        })

    return pd.DataFrame(rows)


# =========================
# UI
# =========================

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=180)
    st.markdown("**— OR —**")
    remit_file = st.file_uploader("Upload remittance file", type=["pdf", "png", "jpg", "xlsx"])
    st.caption("Files processed in memory only. Nothing is stored.")

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=180)
    st.markdown("**— OR —**")
    stmt_file = st.file_uploader("Upload statement file", type=["pdf", "png", "jpg", "xlsx"])
    st.caption("Files processed in memory only. Nothing is stored.")

st.divider()

# =========================
# RUN
# =========================

if st.button("Run Reconciliation"):
    stmt_file_text, stmt_conf, stmt_ok = extract_upload(stmt_file)
    remit_file_text, remit_conf, remit_ok = extract_upload(remit_file)

    if stmt_file and not stmt_ok:
        st.warning("⚠️ Could not reliably read the statement file. Try copy & paste.")

    if remit_file and not remit_ok:
        st.warning("⚠️ Could not reliably read the remittance file. Try copy & paste.")

    stmt_raw = stmt_file_text + "\n" + stmt_text
    remit_raw = remit_file_text + "\n" + remit_text

    if not stmt_raw.strip() or not remit_raw.strip():
        st.error("Statement and remittance are required.")
        st.stop()

    stmt_map = parse_statement(stmt_raw)
    remit_map = parse_remittance(remit_raw)

    if not stmt_map or not remit_map:
        st.error("No matching invoices found. Copy & paste works best.")
        st.stop()

    confidence = round(min(stmt_conf or 0.85, remit_conf or 0.85), 2)

    df = reconcile(stmt_map, remit_map, confidence)

    st.success("Reconciliation complete")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "reconciliation_output.csv",
        "text/csv",
    )
