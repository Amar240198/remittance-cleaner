import re
import io
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract


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


def extract_text(upload) -> str:
    if upload is None:
        return ""

    data = upload.read()
    name = upload.name.lower()

    if name.endswith(".pdf"):
        text = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    if name.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(img, config="--oem 3 --psm 6")

    return data.decode("utf-8", errors="ignore")


def normalize_invoice(inv: str) -> str:
    """
    Strip prefixes and leading zeros.
    INV0531398 -> 531398
    """
    inv = re.sub(r"[A-Z]", "", inv.upper())
    inv = inv.lstrip("0")
    return inv


def find_invoices(text: str):
    raw = re.findall(r"\b(?:INV)?0*\d{5,12}\b", text.upper())
    return [normalize_invoice(r) for r in raw]


# =========================
# PARSERS
# =========================

def parse_statement(text: str) -> Dict[str, float]:
    """
    Invoice -> Expected Total (from statement ONLY)
    """
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
    """
    Invoice -> Paid Total (aggregated)
    """
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
# RECONCILIATION (CORRECT ORDER)
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
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")

st.title("Remittance Cleaner — MVP")
st.caption("Expected = Statement | Paid = Remittance | Difference = Paid − Expected")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=200)
    remit_file = st.file_uploader("Upload remittance file")

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=200)
    stmt_file = st.file_uploader("Upload statement file")

st.divider()

if st.button("Run Reconciliation"):
    stmt_raw = extract_text(stmt_file) + "\n" + stmt_text
    remit_raw = extract_text(remit_file) + "\n" + remit_text

    if not stmt_raw.strip() or not remit_raw.strip():
        st.error("You must provide BOTH statement and remittance text.")
        st.stop()

    stmt_map = parse_statement(stmt_raw)
    remit_map = parse_remittance(remit_raw)

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

