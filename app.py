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
# CONFIG (LIMITS)
# =========================
MAX_UPLOAD_MB = 15  # hard cap for safety (Streamlit also has its own)
MAX_PDF_PAGES = 25
MAX_TEXT_CHARS = 1_500_000  # prevent runaway memory
MAX_LINES_TO_PARSE = 50_000
MAX_EXCEL_ROWS = 50_000
MAX_EXCEL_COLS = 50

# Timeouts (seconds)
EXTRACT_TIMEOUT_S = 10
PARSE_TIMEOUT_S = 10


# =========================
# TIMEOUT UTIL
# =========================
class TimeoutError(Exception):
    pass


@contextmanager
def hard_timeout(seconds: int, label: str = "operation"):
    """
    Hard timeout on Linux/Streamlit Cloud using signal.
    On Windows, signals don't work the same -> we fall back to soft limits only.
    """
    if seconds is None or seconds <= 0:
        yield
        return

    if os.name != "posix":
        # No reliable hard timeout here; rely on soft caps
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out during {label} after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# =========================
# BASIC UTILITIES
# =========================
def clamp_text(s: str) -> str:
    if not s:
        return ""
    if len(s) > MAX_TEXT_CHARS:
        return s[:MAX_TEXT_CHARS]
    return s


def parse_money(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.replace("£", "").replace("€", "").replace("$", "")
    t = t.replace(",", "").strip()
    try:
        return float(t)
    except:
        return None


def normalize_invoice(inv: str) -> str:
    inv = inv.upper().strip()
    inv = re.sub(r"[A-Z]", "", inv)
    return inv.lstrip("0")


def find_invoices(text: str):
    # supports INV0531398, 0531398, etc.
    raw = re.findall(r"\b(?:INV)?0*\d{5,12}\b", (text or "").upper())
    return [normalize_invoice(r) for r in raw]


def safe_read_bytes(upload) -> Tuple[bytes, bool]:
    if upload is None:
        return b"", False
    try:
        data = upload.read()
        if data is None:
            return b"", False
        mb = len(data) / (1024 * 1024)
        if mb > MAX_UPLOAD_MB:
            return b"", False
        return data, True
    except Exception:
        return b"", False


# =========================
# EXTRACTION (TEXT / OCR / EXCEL)
# =========================
def extract_from_pdf(file_bytes: bytes) -> Tuple[str, float]:
    """
    Returns (text, confidence)
    """
    text_chunks = []
    conf = 0.0
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = pdf.pages[:MAX_PDF_PAGES]
            for p in pages:
                t = p.extract_text() or ""
                if t.strip():
                    text_chunks.append(t)
            joined = "\n".join(text_chunks).strip()
            joined = clamp_text(joined)
            if joined:
                conf = 0.80  # text-based PDF extraction is usually decent
                return joined, conf
            return "", 0.0
    except Exception:
        # PDFSyntaxError and friends are caught here -> no crash
        return "", 0.0


def extract_from_image(file_bytes: bytes) -> Tuple[str, float]:
    """
    Returns (text, confidence)
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        # Basic OCR config; more aggressive OCR reduces accuracy sometimes.
        txt = pytesseract.image_to_string(img, config="--oem 3 --psm 6") or ""
        txt = clamp_text(txt.strip())
        if txt:
            return txt, 0.60  # OCR is inherently less reliable
        return "", 0.0
    except Exception:
        return "", 0.0


def extract_from_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, float]:
    """
    Returns (df, confidence)
    Reads first sheet only. Caps rows/cols.
    """
    try:
        with io.BytesIO(file_bytes) as bio:
            # engine=openpyxl is typical; pandas will infer but openpyxl must be installed.
            df = pd.read_excel(bio, sheet_name=0, engine="openpyxl")
        if df is None or df.empty:
            return pd.DataFrame(), 0.0
        # cap shape
        df = df.iloc[:MAX_EXCEL_ROWS, :MAX_EXCEL_COLS].copy()
        return df, 0.95
    except Exception:
        return pd.DataFrame(), 0.0


def extract_upload(upload) -> Tuple[Any, float, str, bool]:
    """
    Returns:
      content: str OR DataFrame
      conf: float
      kind: "paste" | "pdf" | "image" | "text" | "excel" | "unknown"
      ok: bool
    """
    if upload is None:
        return "", 0.0, "none", False

    data, ok = safe_read_bytes(upload)
    if not ok:
        return "", 0.0, "unknown", False

    name = (upload.name or "").lower()

    try:
        with hard_timeout(EXTRACT_TIMEOUT_S, label="file extraction"):
            if name.endswith(".pdf"):
                txt, conf = extract_from_pdf(data)
                return txt, conf, "pdf", bool(txt.strip())

            if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
                txt, conf = extract_from_image(data)
                return txt, conf, "image", bool(txt.strip())

            if name.endswith(".xlsx"):
                df, conf = extract_from_excel(data)
                return df, conf, "excel", not df.empty

            if name.endswith(".txt"):
                txt = data.decode("utf-8", errors="ignore")
                txt = clamp_text(txt)
                return txt, 0.90 if txt.strip() else 0.0, "text", bool(txt.strip())

            # fallback: try decode as text
            txt = data.decode("utf-8", errors="ignore")
            txt = clamp_text(txt)
            return txt, 0.70 if txt.strip() else 0.0, "text", bool(txt.strip())

    except TimeoutError:
        return "", 0.0, "unknown", False
    except Exception:
        return "", 0.0, "unknown", False


# =========================
# PARSERS + CONFIDENCE
# =========================
def _extract_amounts(line: str):
    # money pattern: 123.45 or 1,234.56
    return re.findall(r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b", line or "")


def parse_statement_text(text: str, base_conf: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Statement => Invoice -> Expected Total
    Returns (map, conf_map)
    """
    result: Dict[str, float] = {}
    conf_map: Dict[str, float] = {}

    if not text:
        return result, conf_map

    lines = (text.splitlines() or [])[:MAX_LINES_TO_PARSE]

    for line in lines:
        invs = find_invoices(line)
        amts = _extract_amounts(line)

        if invs and amts:
            expected = parse_money(amts[-1])
            if expected is None:
                continue

            # per-line confidence: base_conf + “clean” match bonus
            line_conf = min(0.99, base_conf + 0.10)

            for inv in invs:
                result[inv] = expected
                conf_map[inv] = max(conf_map.get(inv, 0.0), line_conf)

        elif invs:
            # invoice without amount -> very low confidence (don’t overwrite)
            for inv in invs:
                conf_map[inv] = max(conf_map.get(inv, 0.0), base_conf * 0.40)

    return result, conf_map


def parse_remittance_text(text: str, base_conf: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Remittance => Invoice -> Paid Total (aggregated)
    Returns (map, conf_map)
    """
    result: Dict[str, float] = {}
    conf_map: Dict[str, float] = {}

    if not text:
        return result, conf_map

    lines = (text.splitlines() or [])[:MAX_LINES_TO_PARSE]

    for line in lines:
        invs = find_invoices(line)
        amts = _extract_amounts(line)

        if invs and amts:
            paid = parse_money(amts[-1])
            if paid is None:
                continue

            line_conf = min(0.99, base_conf + 0.10)

            for inv in invs:
                result[inv] = round(result.get(inv, 0.0) + paid, 2)
                conf_map[inv] = max(conf_map.get(inv, 0.0), line_conf)

        elif invs:
            for inv in invs:
                conf_map[inv] = max(conf_map.get(inv, 0.0), base_conf * 0.40)

    return result, conf_map


def parse_excel(df: pd.DataFrame, base_conf: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Very forgiving Excel parser:
      - finds invoice-like values anywhere in a row
      - uses the last money-like value in the row as amount
    Works for both statement and remittance formats when the file is a simple table.
    """
    result: Dict[str, float] = {}
    conf_map: Dict[str, float] = {}

    if df is None or df.empty:
        return result, conf_map

    # Convert to strings safely
    df2 = df.copy()
    df2 = df2.fillna("")

    # cap to prevent runaway
    df2 = df2.iloc[:MAX_EXCEL_ROWS, :MAX_EXCEL_COLS]

    for _, row in df2.iterrows():
        row_text = " | ".join(str(x) for x in row.values)
        invs = find_invoices(row_text)

        # scan row for monetary values
        amts = re.findall(r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b", row_text)
        if not invs or not amts:
            continue

        val = parse_money(amts[-1])
        if val is None:
            continue

        row_conf = min(0.99, base_conf + 0.10)  # Excel is usually high quality

        for inv in invs:
            # Aggregate by default (safe for remittance), statement will overwrite later if needed.
            result[inv] = round(result.get(inv, 0.0) + val, 2)
            conf_map[inv] = max(conf_map.get(inv, 0.0), row_conf)

    return result, conf_map


# =========================
# RECONCILIATION
# =========================
def reconcile(
    stmt_map: Dict[str, float],
    remit_map: Dict[str, float],
    stmt_conf_map: Dict[str, float],
    remit_conf_map: Dict[str, float],
) -> pd.DataFrame:

    invoices = sorted(set(stmt_map.keys()) | set(remit_map.keys()))
    rows = []

    for inv in invoices:
        expected = stmt_map.get(inv)
        paid = remit_map.get(inv, 0.0)

        stmt_c = stmt_conf_map.get(inv, 0.0)
        remit_c = remit_conf_map.get(inv, 0.0)

        # Confidence: if both sides exist, take min (weakest link); else take whichever exists
        if expected is not None and paid is not None:
            conf = min(stmt_c or 0.0, remit_c or 0.0)
        else:
            conf = max(stmt_c or 0.0, remit_c or 0.0)

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
            "Confidence": round(conf, 2),
            "Status": status,
        })

    return pd.DataFrame(rows)


# =========================
# STREAMLIT UI (STABLE)
# =========================
st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")

st.title("Remittance Cleaner — MVP")
st.caption("Statement = Expected | Remittance = Paid | Difference calculated | No data stored")

left, right = st.columns(2)

with left:
    st.subheader("Remittance")
    remit_text = st.text_area("Copy & paste remittance text", height=200, placeholder="Paste remittance content here…")
    st.markdown("**or**")
    remit_file = st.file_uploader("Upload remittance file (PDF / Image / TXT / XLSX)", type=["pdf", "png", "jpg", "jpeg", "txt", "xlsx"])

with right:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Copy & paste statement text", height=200, placeholder="Paste statement content here…")
    st.markdown("**or**")
    stmt_file = st.file_uploader("Upload statement file (PDF / Image / TXT / XLSX)", type=["pdf", "png", "jpg", "jpeg", "txt", "xlsx"])

st.divider()
run = st.button("Run Reconciliation", type="primary")


# =========================
# PROCESS (SAFE, TIME-LIMITED)
# =========================
if run:
    start = time.time()

    # Extract uploads
    stmt_content, stmt_base_conf, stmt_kind, stmt_ok = extract_upload(stmt_file)
    remit_content, remit_base_conf, remit_kind, remit_ok = extract_upload(remit_file)

    # Combine with pasted text (paste gets high confidence)
    stmt_paste = (stmt_text or "").strip()
    remit_paste = (remit_text or "").strip()

    # If paste exists, treat it as reliable input
    if stmt_paste:
        stmt_base_conf = max(stmt_base_conf, 0.95)
    if remit_paste:
        remit_base_conf = max(remit_base_conf, 0.95)

    # Build raw strings/dataframes to parse
    # NOTE: if excel upload exists, we parse the dataframe directly.
    # If both excel + paste exist, we merge results (paste can override statement).
    stmt_map: Dict[str, float] = {}
    remit_map: Dict[str, float] = {}
    stmt_conf_map: Dict[str, float] = {}
    remit_conf_map: Dict[str, float] = {}

    try:
        with hard_timeout(PARSE_TIMEOUT_S, label="parsing"):
            # STATEMENT
            if isinstance(stmt_content, pd.DataFrame) and not stmt_content.empty:
                m, cm = parse_excel(stmt_content, stmt_base_conf)
                # For statement, prefer "latest" value per invoice if duplicates exist; Excel parse aggregates,
                # but statement tables often have one amount per invoice. We'll keep as-is.
                stmt_map.update(m)
                for k, v in cm.items():
                    stmt_conf_map[k] = max(stmt_conf_map.get(k, 0.0), v)

            # Add pasted statement text
            if stmt_paste:
                m, cm = parse_statement_text(stmt_paste, 0.95)
                stmt_map.update(m)  # paste overrides statement amounts
                for k, v in cm.items():
                    stmt_conf_map[k] = max(stmt_conf_map.get(k, 0.0), v)
            else:
                # If no paste, but extracted text exists (pdf/image/txt)
                if isinstance(stmt_content, str) and stmt_content.strip():
                    m, cm = parse_statement_text(stmt_content, stmt_base_conf)
                    stmt_map.update(m)
                    for k, v in cm.items():
                        stmt_conf_map[k] = max(stmt_conf_map.get(k, 0.0), v)

            # REMITTANCE
            if isinstance(remit_content, pd.DataFrame) and not remit_content.empty:
                m, cm = parse_excel(remit_content, remit_base_conf)
                remit_map.update(m)  # excel parse aggregates already
                for k, v in cm.items():
                    remit_conf_map[k] = max(remit_conf_map.get(k, 0.0), v)

            # Add pasted remittance text
            if remit_paste:
                m, cm = parse_remittance_text(remit_paste, 0.95)
                for k, v in m.items():
                    remit_map[k] = round(remit_map.get(k, 0.0) + v, 2)
                for k, v in cm.items():
                    remit_conf_map[k] = max(remit_conf_map.get(k, 0.0), v)
            else:
                if isinstance(remit_content, str) and remit_content.strip():
                    m, cm = parse_remittance_text(remit_content, remit_base_conf)
                    for k, v in m.items():
                        remit_map[k] = round(remit_map.get(k, 0.0) + v, 2)
                    for k, v in cm.items():
                        remit_conf_map[k] = max(remit_conf_map.get(k, 0.0), v)

    except TimeoutError:
        st.error("⏱️ Processing timed out. Try smaller files or copy/paste the text.")
        st.stop()
    except Exception:
        st.error("This app encountered an error while parsing. Try copy/paste instead of upload.")
        st.stop()

    # If either side looks unreliable, show your exact warning
    if not stmt_map or not remit_map:
        st.warning(
            "⚠️ We couldn’t reliably read one of the inputs.\n\n"
            "For best results, **copy & paste the text directly**."
        )
        st.stop()

    df = reconcile(stmt_map, remit_map, stmt_conf_map, remit_conf_map)

    if df.empty:
        st.warning("No matching invoices found between statement and remittance.")
        st.stop()

    elapsed = round(time.time() - start, 2)
    st.success(f"Reconciliation complete (processed in {elapsed}s)")

    # Optional: sort to show low-confidence first (useful in QA)
    df = df.sort_values(by=["Confidence", "Status"], ascending=[True, True])

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "reconciliation_output.csv",
        "text/csv",
    )
