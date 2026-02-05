import io
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import streamlit as st

import fitz  # PyMuPDF
from pypdf import PdfReader
import pdfplumber


# =========================
# CONFIG / LIMITS
# =========================
MAX_PARSE_SECONDS = 20
MIN_TEXT_LEN_OK = 150


# =========================
# ERRORS (NO DATA LEAK)
# =========================
@dataclass
class ParseFailure(Exception):
    file_label: str
    stage: str
    reason: str

    def __str__(self):
        return f"{self.file_label}: {self.stage} — {self.reason}"


# =========================
# UTILITIES
# =========================
def now_s() -> float:
    return time.time()


def parse_money(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("£", "").replace("€", "").replace("$", "")
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


def is_probably_pdf(file_bytes: bytes) -> bool:
    # Most PDFs begin with %PDF-
    return file_bytes[:5] == b"%PDF-"


# =========================
# PDF TEXT EXTRACTION (ROBUST, NO OCR)
# =========================
def extract_pdf_text_pymupdf(file_bytes: bytes, deadline: float) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            if now_s() > deadline:
                raise TimeoutError
            text_parts.append(page.get_text("text") or "")
    return "\n".join(text_parts).strip()


def extract_pdf_text_pypdf(file_bytes: bytes, deadline: float) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        if now_s() > deadline:
            raise TimeoutError
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_pdf_text_pdfplumber(file_bytes: bytes, deadline: float) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            if now_s() > deadline:
                raise TimeoutError
            parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_pdf_text_safe(file_bytes: bytes, file_label: str) -> Tuple[str, float]:
    """
    Try 3 engines:
      1) PyMuPDF (best on Streamlit Cloud)
      2) pypdf
      3) pdfplumber
    Add silent retry pass.
    """
    if not is_probably_pdf(file_bytes):
        raise ParseFailure(file_label, "PDF header check", "File is not a valid PDF (missing %PDF header)")

    deadline = now_s() + MAX_PARSE_SECONDS

    engines = [
        ("PyMuPDF", extract_pdf_text_pymupdf),
        ("pypdf", extract_pdf_text_pypdf),
        ("pdfplumber", extract_pdf_text_pdfplumber),
    ]

    last_err = None

    for attempt in (1, 2):  # silent retry pass
        for name, fn in engines:
            try:
                txt = fn(file_bytes, deadline)
                if len(txt) >= MIN_TEXT_LEN_OK:
                    # higher confidence if we got decent text length
                    conf = 0.9 if name == "PyMuPDF" else 0.85
                    return txt, conf
                # If tiny text, keep trying other engines (maybe it's scanned)
            except TimeoutError:
                raise
            except Exception as e:
                last_err = f"{name}: {type(e).__name__}"
                continue

    # If we got here: likely scanned PDF (needs OCR) OR weird export
    # We do NOT OCR on Streamlit Cloud without OS deps.
    raise ParseFailure(
        file_label,
        "PDF text extraction",
        f"No readable text extracted (likely scanned / image-only PDF). Last error: {last_err or 'unknown'}"
    )


# =========================
# EXCEL EXTRACTION
# =========================
def extract_excel_safe(file_bytes: bytes, file_label: str) -> Tuple[pd.DataFrame, float]:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        if df is None or df.empty:
            raise ParseFailure(file_label, "Excel read", "Excel loaded but appears empty")
        return df, 0.95
    except ParseFailure:
        raise
    except Exception as e:
        raise ParseFailure(file_label, "Excel read", f"{type(e).__name__}")


# =========================
# GENERIC UPLOAD HANDLER
# =========================
def extract_upload(upload, file_label: str) -> Tuple[Union[str, pd.DataFrame], float, str]:
    if upload is None:
        return "", 0.0, "none"

    name = (upload.name or "").lower()
    file_bytes = upload.read()

    if name.endswith(".xlsx"):
        df, conf = extract_excel_safe(file_bytes, file_label)
        return df, conf, "excel"

    if name.endswith(".pdf"):
        txt, conf = extract_pdf_text_safe(file_bytes, file_label)
        return txt, conf, "pdf"

    # text fallback
    try:
        txt = file_bytes.decode("utf-8", errors="ignore")
        if not txt.strip():
            raise ParseFailure(file_label, "Text decode", "File decoded but empty")
        return txt, 0.8, "text"
    except ParseFailure:
        raise
    except Exception as e:
        raise ParseFailure(file_label, "Text decode", f"{type(e).__name__}")


# =========================
# PARSERS
# =========================
def parse_statement(text: str) -> Dict[str, float]:
    """
    Invoice -> Expected Total (statement)
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
    Invoice -> Paid Total (remittance)
    """
    result = {}
    for line in text.splitlines():
        invoices = find_invoices(line)
        amounts = re.findall(r"\d+\.\d{2}", line)
        if invoices and amounts:
            paid = parse_money(amounts[-1])
            for inv in invoices:
                result[inv] = round(result.get(inv, 0.0) + (paid or 0.0), 2)
    return result


def parse_excel_amounts(df: pd.DataFrame) -> Dict[str, float]:
    """
    Generic Excel parser: find invoice + last numeric amount in row.
    """
    out = {}
    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row.values if x is not None)
        invs = find_invoices(row_text)
        if not invs:
            continue

        nums = []
        for x in row.values:
            val = parse_money(x)
            if val is not None:
                nums.append(val)
        if not nums:
            continue

        amt = float(nums[-1])
        for inv in invs:
            out[inv] = amt
    return out


# =========================
# RECONCILIATION
# =========================
def reconcile(stmt_map: Dict[str, float], remit_map: Dict[str, float], conf: float) -> pd.DataFrame:
    rows = []
    invoices = sorted(set(stmt_map.keys()) | set(remit_map.keys()))

    for inv in invoices:
        expected = stmt_map.get(inv)
        paid = remit_map.get(inv, 0.0)

        if expected is None:
            rows.append({
                "Invoice_Number": inv,
                "Expected_Total": None,
                "Paid_Total": round(paid, 2),
                "Difference": None,
                "Status": "MISSING FROM STATEMENT",
                "Confidence": round(conf, 2),
            })
            continue

        diff = round((paid or 0.0) - (expected or 0.0), 2)

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
            "Expected_Total": round(expected, 2) if expected is not None else None,
            "Paid_Total": round(paid, 2),
            "Difference": diff,
            "Status": status,
            "Confidence": round(conf, 2),
        })

    return pd.DataFrame(rows)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")
st.title("Remittance Cleaner — MVP")
st.caption("Expected Total comes from Statement | Paid Total comes from Remittance | Difference = Paid - Expected")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text", height=180)
    st.markdown("**or**")
    remit_file = st.file_uploader("Upload remittance file (PDF / XLSX / TXT)", type=["pdf", "xlsx", "txt"])

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text", height=180)
    st.markdown("**or**")
    stmt_file = st.file_uploader("Upload statement file (PDF / XLSX / TXT)", type=["pdf", "xlsx", "txt"])

st.divider()

if st.button("Run Reconciliation", type="primary"):
    start = now_s()
    deadline = start + MAX_PARSE_SECONDS

    def enforce_timeout():
        if now_s() > deadline:
            raise TimeoutError

    try:
        enforce_timeout()
        stmt_content, stmt_conf, stmt_type = extract_upload(stmt_file, "Statement file")
        enforce_timeout()
        remit_content, remit_conf, remit_type = extract_upload(remit_file, "Remittance file")

        # Parse maps
        if isinstance(stmt_content, pd.DataFrame):
            stmt_map = parse_excel_amounts(stmt_content)
        else:
            stmt_raw = (stmt_content or "") + "\n" + (stmt_text or "")
            stmt_map = parse_statement(stmt_raw)

        enforce_timeout()

        if isinstance(remit_content, pd.DataFrame):
            remit_map = parse_excel_amounts(remit_content)
        else:
            remit_raw = (remit_content or "") + "\n" + (remit_text or "")
            remit_map = parse_remittance(remit_raw)

        enforce_timeout()

        # If missing/empty, show which failed without data
        failed = []
        if not stmt_map:
            failed.append("Statement")
        if not remit_map:
            failed.append("Remittance")

        if failed:
            st.warning(
                "⚠️ We couldn’t reliably read: **" + ", ".join(failed) + "**.\n\n"
                "Try **copy & paste** instead.\n\n"
                "If it's a **scanned/image-only PDF**, this MVP (cloud) cannot OCR it yet."
            )
            st.stop()

        # Confidence: min of extractors
        confidence = min(stmt_conf or 0.0, remit_conf or 0.0)

        df = reconcile(stmt_map, remit_map, confidence)

        st.success("Reconciliation complete")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="reconciliation_output.csv",
            mime="text/csv"
        )

    except TimeoutError:
        st.error(
            "⏱️ Processing timed out.\n\n"
            "Try smaller files or use copy/paste."
        )

    except ParseFailure as e:
        # This answers your “Which file failed” request without leaking data
        st.error(
            "This app encountered an error while parsing.\n\n"
            f"**Which file failed:** {e.file_label}\n\n"
            f"**Stage:** {e.stage}\n\n"
            f"**Reason:** {e.reason}\n\n"
            "Try copy/paste instead of upload."
        )

    except Exception as e:
        st.error(
            "This app encountered an error while parsing.\n\n"
            "Try copy/paste instead of upload."
        )
