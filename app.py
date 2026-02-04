import re
import io
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract


# =========================
# STREAMLIT SAFETY GUARD
# =========================

if "ready" not in st.session_state:
    st.session_state.ready = False


# =========================
# BASIC UTILITIES
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

def extract_text(upload) -> Tuple[str, bool]:
    """
    Returns (text, success)
    """
    if upload is None:
        return "", False

    try:
        data = upload.read()
        name = upload.name.lower()

        if name.endswith(".pdf"):
            text = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text), True

        if name.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(io.BytesIO(data))
            return pytesseract.image_to_string(img, config="--oem 3 --psm 6"), True

        if name.endswith(".txt"):
            return data.decode("utf-8", errors="ignore"), True

    except Exception:
        return "", False

    return "", False


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


# =========================
# RECONCILIATION
# =========================

def reconcile(statement: Dict[str, float], remittance: Dict[str, float]) -> pd.DataFrame:
    rows = []
    invoices = sorted(set(statement.keys()) | set(remittance.keys()))

    for inv in invoices:
        expected = statement.get(inv)
        paid = remittance.get(inv, 0.0)

        if expected is None:
            rows.append({
                "Invoice_Number": inv,
                "Expected_Total": None,
                "Paid_Total": paid,
                "Difference": None,
                "Status": "MISSING FROM STATEMENT"
            })
            continue

        difference = round(paid - expected, 2)

        if paid == 0:
            status = "NOT PAID"
        elif abs(difference) < 0.01:
            status = "FULLY PAID"
            difference = 0.0
        elif difference < 0:
            status = "UNDERPAID"
        else:
            status = "OVERPAID"

        rows.append({
            "Invoice_Number": inv,
            "Expected_Total": expected,
            "Paid_Total": paid,
            "Difference": difference,
            "Status": status
        })

    return pd.DataFrame(rows)


# =========================
# UI
# =========================

st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")
st.title("Remittance Cleaner — MVP")
st.caption("Statement = Expected | Remittance = Paid | Difference calculated")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Copy & paste remittance text", height=200)
    st.markdown("**or**")
    remit_file = st.file_uploader("Upload remittance file")

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Copy & paste statement text", height=200)
    st.markdown("**or**")
    stmt_file = st.file_uploader("Upload statement file")

st.divider()

if st.button("Run Reconciliation"):
    st.session_state.ready = True


# =========================
# PROCESS (SAFE)
# =========================

if st.session_state.ready:

    stmt_file_text, stmt_ok = extract_text(stmt_file)
    remit_file_text, remit_ok = extract_text(remit_file)

    stmt_raw = (stmt_file_text if stmt_ok else "") + "\n" + stmt_text
    remit_raw = (remit_file_text if remit_ok else "") + "\n" + remit_text

    stmt_map = parse_statement(stmt_raw)
    remit_map = parse_remittance(remit_raw)

    if not stmt_map or not remit_map:
        st.warning(
            "⚠️ We couldn’t reliably read one of the inputs.\n\n"
            "For best results, **copy & paste the text directly**."
        )
        st.stop()

    df = reconcile(stmt_map, remit_map)

    if df.empty:
        st.warning("No matching invoices found between statement and remittance.")
    else:
        st.success("Reconciliation complete")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "reconciliation_output.csv",
            "text/csv"
        )
