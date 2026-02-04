import re
import io
import os
import time
import signal
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract


# =========================
# SAFETY / LIMITS
# =========================
MAX_UPLOAD_MB = 15
MAX_PDF_PAGES = 25
MAX_TEXT_CHARS = 1_500_000
MAX_LINES = 50_000
MAX_EXCEL_ROWS = 50_000
MAX_EXCEL_COLS = 50

EXTRACT_TIMEOUT_S = 10
PARSE_TIMEOUT_S = 10


# =========================
# TIMEOUT UTIL
# =========================
class TimeoutError(Exception):
    pass


@contextmanager
def hard_timeout(seconds: int, label: str):
    if os.name != "posix" or seconds <= 0:
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"Timeout during {label}")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# =========================
# BASIC UTILITIES
# =========================
def clamp_text(s: str) -> str:
    return s[:MAX_TEXT_CHARS] if s else ""


def parse_money(t: str) -> Optional[float]:
    if not t:
        return None
    t = t.replace("£", "").replace("€", "").replace("$", "").replace(",", "").strip()
    try:
        return float(t)
    except:
        return None


def normalize_invoice(inv: str) -> str:
    inv = re.sub(r"[A-Z]", "", inv.upper())
    return inv.lstrip("0")


def find_invoices(text: str):
    raw = re.findall(r"\b(?:INV)?0*\d{5,12}\b", (text or "").upper())
    return [normalize_invoice(r) for r in raw]


def extract_amounts(line: str):
    return re.findall(r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b", line or "")


# =========================
# EXTRACTION
# =========================
def extract_upload(upload) -> Tuple[Any, float, bool, str]:
    """
    Returns: content, confidence, success, failure_reason
    """
    if upload is None:
        return "", 0.0, False, "no file uploaded"

    try:
        data = upload.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            return "", 0.0, False, "file too large"

        name = upload.name.lower()

        with hard_timeout(EXTRACT_TIMEOUT_S, "file extraction"):

            if name.endswith(".pdf"):
                pages = []
                try:
                    with pdfplumber.open(io.BytesIO(data)) as pdf:
                        for p in pdf.pages[:MAX_PDF_PAGES]:
                            pages.append(p.extract_text() or "")
                    text = clamp_text("\n".join(pages))
                    return text, 0.80 if text else 0.0, bool(text), "scanned or unreadable PDF"
                except Exception:
                    return "", 0.0, False, "invalid or scanned PDF"

            if name.endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(io.BytesIO(data))
                text = clamp_text(pytesseract.image_to_string(img))
                return text, 0.60 if text else 0.0, bool(text), "OCR failed"

            if name.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
                df = df.iloc[:MAX_EXCEL_ROWS, :MAX_EXCEL_COLS]
                return df, 0.95, not df.empty, "empty or malformed Excel"

            # fallback text
            text = clamp_text(data.decode("utf-8", errors="ignore"))
            return text, 0.85 if text else 0.0, bool(text), "unreadable text"

    except TimeoutError:
        return "", 0.0, False, "file processing timed out"
    except Exception:
        return "", 0.0, False, "unknown file error"


# =========================
# PARSERS
# =========================
def parse_statement_text(text: str, base_conf: float):
    m, c = {}, {}
    for line in text.splitlines()[:MAX_LINES]:
        invs = find_invoices(line)
        amts = extract_amounts(line)
        if invs and amts:
            val = parse_money(amts[-1])
            for inv in invs:
                m[inv] = val
                c[inv] = max(c.get(inv, 0), base_conf + 0.1)
    return m, c


def parse_remittance_text(text: str, base_conf: float):
    m, c = {}, {}
    for line in text.splitlines()[:MAX_LINES]:
        invs = find_invoices(line)
        amts = extract_amounts(line)
        if invs and amts:
            val = parse_money(amts[-1])
            for inv in invs:
                m[inv] = round(m.get(inv, 0) + val, 2)
                c[inv] = max(c.get(inv, 0), base_conf + 0.1)
    return m, c


def parse_excel(df: pd.DataFrame, base_conf: float):
    m, c = {}, {}
    df = df.fillna("")
    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row.values)
        invs = find_invoices(row_text)
        amts = extract_amounts(row_text)
        if invs and amts:
            val = parse_money(amts[-1])
            for inv in invs:
                m[inv] = round(m.get(inv, 0) + val, 2)
                c[inv] = max(c.get(inv, 0), base_conf + 0.1)
    return m, c


# =========================
# RECONCILIATION
# =========================
def reconcile(stmt, remit, stmt_c, remit_c):
    rows = []
    invoices = sorted(set(stmt) | set(remit))
    for inv in invoices:
        expected = stmt.get(inv)
        paid = remit.get(inv, 0.0)
        conf = min(stmt_c.get(inv, 1), remit_c.get(inv, 1))
        if expected is None:
            rows.append({
                "Invoice_Number": inv,
                "Expected_Total": None,
                "Paid_Total": paid,
                "Difference": None,
                "Confidence": round(conf, 2),
                "Status": "MISSING FROM STATEMENT",
            })
            continue
        diff = round(paid - expected, 2)
        status = (
            "NOT PAID" if paid == 0 else
            "FULLY PAID" if abs(diff) < 0.01 else
            "UNDERPAID" if diff < 0 else
            "OVERPAID"
        )
        rows.append({
            "Invoice_Number": inv,
            "Expected_Total": expected,
            "Paid_Total": paid,
            "Difference": 0.0 if status == "FULLY PAID" else diff,
            "Confidence": round(conf, 2),
            "Status": status,
        })
    return pd.DataFrame(rows)


# =========================
# UI
# =========================
st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")
st.title("Remittance Cleaner — MVP")
st.caption("Paste text or upload files. No data is stored.")

l, r = st.columns(2)

with l:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=200)
    st.markdown("**or**")
    remit_file = st.file_uploader("Upload remittance file", type=["pdf", "png", "jpg", "xlsx", "txt"])

with r:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=200)
    st.markdown("**or**")
    stmt_file = st.file_uploader("Upload statement file", type=["pdf", "png", "jpg", "xlsx", "txt"])

st.divider()
run = st.button("Run Reconciliation", type="primary")


# =========================
# PROCESS (WITH RETRY + FILE IDENTIFICATION)
# =========================
if run:
    try:
        with hard_timeout(PARSE_TIMEOUT_S, "parsing"):

            # --- extract uploads ---
            stmt_content, stmt_conf, stmt_ok, stmt_reason = extract_upload(stmt_file)
            remit_content, remit_conf, remit_ok, remit_reason = extract_upload(remit_file)

            # --- silent retry using pasted text ---
            if not stmt_ok and stmt_text.strip():
                stmt_content, stmt_conf, stmt_ok = stmt_text, 0.95, True

            if not remit_ok and remit_text.strip():
                remit_content, remit_conf, remit_ok = remit_text, 0.95, True

            # --- if still failing, show which file ---
            if not stmt_ok:
                st.error(
                    f"Statement file could not be parsed.\n\n"
                    f"Reason: {stmt_reason}.\n\n"
                    f"Try copy & paste instead."
                )
                st.stop()

            if not remit_ok:
                st.error(
                    f"Remittance file could not be parsed.\n\n"
                    f"Reason: {remit_reason}.\n\n"
                    f"Try copy & paste instead."
                )
                st.stop()

            # --- parse ---
            stmt_map, stmt_c = (
                parse_excel(stmt_content, stmt_conf)
                if isinstance(stmt_content, pd.DataFrame)
                else parse_statement_text(stmt_content, stmt_conf)
            )

            remit_map, remit_c = (
                parse_excel(remit_content, remit_conf)
                if isinstance(remit_content, pd.DataFrame)
                else parse_remittance_text(remit_content, remit_conf)
            )

            if not stmt_map or not remit_map:
                st.warning(
                    "⚠️ We couldn’t reliably read one of the inputs.\n\n"
                    "For best results, copy & paste the text directly."
                )
                st.stop()

            df = reconcile(stmt_map, remit_map, stmt_c, remit_c)

            if df.empty:
                st.warning("No matching invoices found.")
                st.stop()

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
            "Try smaller files or copy & paste instead."
        )
        st.stop()

    except Exception:
        st.error(
            "This app encountered an error while parsing.\n\n"
            "Try copy/paste instead of upload."
        )
        st.stop()
