import re
import io
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Remittance Cleaner — Private Beta",
    layout="wide",
)

# =========================
# BETA BANNER
# =========================
st.markdown(
    """
    <div style="
        background-color:#f5c542;
        padding:12px;
        border-radius:6px;
        text-align:center;
        font-weight:700;
        color:#000;
        margin-bottom:16px;
    ">
        🚧 PRIVATE BETA — Parsing may fail on some files. Copy & paste is most reliable.
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Remittance Cleaner — MVP")
st.caption("Statement = Expected • Remittance = Paid • Difference calculated automatically")

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
# FILE EXTRACTION (SAFE)
# =========================

def extract_upload(upload) -> Tuple[str, float]:
    """
    Returns (text, confidence score)
    """
    if upload is None:
        return "", 0.0

    data = upload.read()
    name = upload.name.lower()

    try:
        if name.endswith(".pdf"):
            pages = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
            return "\n".join(pages), 0.9

        if name.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(io.BytesIO(data))
            text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            return text, 0.6

        if name.endswith(".txt"):
            return data.decode("utf-8", errors="ignore"), 1.0

    except Exception:
        return "", 0.0

    return "", 0.0


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
# SAMPLE DATA DOWNLOAD
# =========================

sample_data = """Invoice,Expected_Total
INV10001,120.00
INV10002,75.50
INV10003,210.00
"""

st.download_button(
    "⬇️ Download sample statement file",
    sample_data,
    "sample_statement.txt",
    help="Use this sample to test the app without real data",
)

# =========================
# UI INPUTS
# =========================

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=180)
    st.markdown("**— OR —**")
    remit_file = st.file_uploader("Upload remittance file")

    st.caption("🔒 Files are processed in memory only. Nothing is stored.")

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=180)
    st.markdown("**— OR —**")
    stmt_file = st.file_uploader("Upload statement file")

    st.caption("🔒 Files are processed in memory only. Nothing is stored.")

st.divider()

# =========================
# RUN
# =========================

if st.button("Run Reconciliation"):
    stmt_file_text, stmt_conf = extract_upload(stmt_file)
    remit_file_text, remit_conf = extract_upload(remit_file)

    stmt_raw = (stmt_file_text or "") + "\n" + (stmt_text or "")
    remit_raw = (remit_file_text or "") + "\n" + (remit_text or "")

    if not stmt_raw.strip() or not remit_raw.strip():
        st.warning(
            "⚠️ We couldn’t reliably read one of the inputs.\n\n"
            "For best results, **copy & paste the text** directly."
        )
        st.stop()

    stmt_map = parse_statement(stmt_raw)
    remit_map = parse_remittance(remit_raw)

    if not stmt_map or not remit_map:
        st.warning(
            "⚠️ Parsing failed.\n\n"
            "Try copy-pasting the text instead of uploading."
        )
        st.stop()

    confidence = round(min(stmt_conf or 0.9, remit_conf or 0.9), 2)

    df = reconcile(stmt_map, remit_map, confidence)

    st.success("Reconciliation complete")

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Confidence": st.column_config.NumberColumn(
                "Confidence",
                help="Estimated reliability of extracted data (1.0 = highest)",
                format="%.2f",
            )
        },
    )

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "reconciliation_output.csv",
        "text/csv",
    )

# =========================
# FOOTER
# =========================
st.markdown(
    "<hr style='margin-top:40px'>"
    "<p style='text-align:center;color:gray;font-size:0.9em'>"
    "© Private Beta — Not for production use"
    "</p>",
    unsafe_allow_html=True,
)
