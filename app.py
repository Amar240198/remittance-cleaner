import re
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

# Optional dependencies for file text extraction
import pdfplumber
from PIL import Image
import pytesseract


# ----------------------------
# Helpers: normalization
# ----------------------------
def norm_text(s: str) -> str:
    if not s:
        return ""
    # normalize weird spaces and commas
    s = s.replace("\u00a0", " ")
    return s.strip()


def parse_money(x: str) -> Optional[float]:
    """
    Parses money like:
    £1,234.56  1234.56  1 234,56  etc.
    Returns float or None.
    """
    if not x:
        return None
    t = x.strip()
    # remove currency symbols and spaces
    t = re.sub(r"[£$€]", "", t)
    t = t.replace(" ", "")
    # handle comma decimal (1.234,56) vs comma thousands (1,234.56)
    # heuristic: if both ',' and '.' exist, assume ',' is thousands if '.' appears last
    if "," in t and "." in t:
        if t.rfind(".") > t.rfind(","):
            t = t.replace(",", "")
        else:
            t = t.replace(".", "").replace(",", ".")
    else:
        # if only comma exists, assume comma is decimal when exactly 2 decimals after it
        if "," in t and re.search(r",\d{2}$", t):
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


# ----------------------------
# Text extraction from uploads
# ----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    out = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            out.append(txt)
    return "\n".join(out).strip()


def extract_text_from_image(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)


def extract_text_from_upload(upload) -> str:
    if upload is None:
        return ""
    b = upload.read()
    name = (upload.name or "").lower()

    # PDF
    if name.endswith(".pdf"):
        return extract_text_from_pdf(b)

    # images
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        return extract_text_from_image(b)

    # fallback: treat as text file
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ----------------------------
# Core parsing logic
# ----------------------------
@dataclass
class RemitLine:
    invoice: str
    paid: float


@dataclass
class StatementLine:
    invoice: str
    net: Optional[float]
    vat: Optional[float]
    expected: Optional[float]
    remaining: Optional[float]


def find_invoice_candidates(text: str) -> List[str]:
    """
    Extract invoice numbers. You will likely adjust patterns based on your invoices.
    Patterns included:
      - INV-12345, INV12345
      - 8-12 digit invoice numbers
      - 'Invoice 12345'
    """
    t = text.upper()
    candidates = set()

    # INV-12345 / INV12345 / INVOICE-12345
    for m in re.finditer(r"\b(?:INV|INVOICE)[\s\-#:]*([A-Z0-9]{4,20})\b", t):
        candidates.add(m.group(1))

    # "Invoice 123456"
    for m in re.finditer(r"\bINVOICE[\s#:]*([0-9]{4,12})\b", t):
        candidates.add(m.group(1))

    # raw numeric invoice (be conservative)
    for m in re.finditer(r"\b([0-9]{6,12})\b", t):
        candidates.add(m.group(1))

    return list(candidates)


def parse_remittance(text: str) -> Dict[str, float]:
    """
    Returns dict invoice -> total paid (pivot replacement).
    Assumes remittance text contains invoice number near an amount.
    """
    t = norm_text(text)
    if not t:
        return {}

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    invoice_paid: Dict[str, float] = {}

    # Pattern: invoice + amount on same line (common in remittances)
    amt_pat = re.compile(r"([£$€]?\s*\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d{2})|\b\d+(?:[.,]\d{2})\b)")

    for ln in lines:
        upper = ln.upper()

        invs = find_invoice_candidates(upper)
        if not invs:
            continue

        amts = [parse_money(m.group(1)) for m in amt_pat.finditer(ln)]
        amts = [a for a in amts if a is not None]
        if not amts:
            continue

        # Heuristic: in remittance lines, last amount is often the paid amount
        paid = amts[-1]

        # If multiple invoices on line, we assign the same paid amount to each (rare).
        # Better: split lines earlier; keep MVP simple.
        for inv in invs:
            invoice_paid[inv] = round(invoice_paid.get(inv, 0.0) + paid, 2)

    return invoice_paid


def parse_statement(text: str) -> Dict[str, StatementLine]:
    """
    Extract invoice -> net, vat, expected (net+vat), remaining balance if present.
    We prioritize expected if explicitly found, else compute net+vat.
    """
    t = norm_text(text)
    if not t:
        return {}

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    out: Dict[str, StatementLine] = {}

    money_pat = re.compile(r"([£$€]?\s*\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d{2})|\b\d+(?:[.,]\d{2})\b)")

    for ln in lines:
        upper = ln.upper()
        invs = find_invoice_candidates(upper)
        if not invs:
            continue

        amts = [parse_money(m.group(1)) for m in money_pat.finditer(ln)]
        amts = [a for a in amts if a is not None]

        # Heuristics for statement lines:
        # - Often contains 2 or 3 numbers: Net, VAT, Gross OR Remaining
        net = vat = expected = remaining = None

        # Attempt Net + VAT extraction (optional)
if "VAT" in upper and len(amts) >= 2:
    net, vat = amts[0], amts[1]
    expected = round(net + vat, 2)

# Gross-only line (very common)
elif len(amts) == 1 and any(k in upper for k in ["TOTAL", "AMOUNT", "DUE"]):
    expected = amts[0]

        # Remaining balance hints
        if any(k in upper for k in ["REMAIN", "BALANCE", "OUTSTANDING", "DUE"]) and amts:
            remaining = amts[-1]

        for inv in invs:
            # if multiple lines mention same invoice, last wins (MVP)
            out[inv] = StatementLine(invoice=inv, net=net, vat=vat, expected=expected, remaining=remaining)

    return out


def reconcile(remit: Dict[str, float], stmt: Dict[str, StatementLine]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    alerts = []

    all_invoices = set(remit.keys()) | set(stmt.keys())

    for inv in sorted(all_invoices):
        paid = round(remit.get(inv, 0.0), 2) if inv in remit else 0.0

        st_line = stmt.get(inv)
        net = vat = expected = remaining = None
        if st_line:
            net = st_line.net
            vat = st_line.vat
            expected = st_line.expected
            remaining = st_line.remaining

        status = "INCOMPLETE"
        reason = ""

        if not st_line:
            status = "UNKNOWN INVOICE"
            reason = "Invoice not found in statement"
        else:
           if remaining is not None:
    if abs(remaining) <= 0.01:
        status = "FULLY PAID"
    elif remaining > 0:
        status = "UNDERPAID"
    else:
        status = "OVERPAID"
elif expected is not None:
    diff = round(paid - expected, 2)
    if abs(diff) <= 0.01:
        status = "FULLY PAID"
    elif diff < 0:
        status = "UNDERPAID"
    else:
        status = "OVERPAID"
else:
    status = "INCOMPLETE"
    reason = "No remaining balance or expected amount found"

                diff = round(paid - expected, 2)
                if abs(paid) < 0.01:
                    status = "NOT PAID"
                elif abs(diff) <= 0.01:
                    status = "FULLY PAID"
                elif diff < -0.01:
                    status = "UNDERPAID"
                else:
                    status = "OVERPAID"

        diff_val = None
        if expected is not None:
            diff_val = round(paid - expected, 2)

        rows.append({
            "Invoice_Number": inv,
            "Net_Amount": net,
            "VAT_Amount": vat,
            "Expected_Total": expected,
            "Paid_Total": paid,
            "Difference": diff_val,
            "Payment_Status": status,
            "Remaining_Balance_Statement": remaining,
            "Issue_Reason": reason,
        })

        if status not in ["FULLY PAID"]:
            alerts.append({
                "Invoice_Number": inv,
                "Payment_Status": status,
                "Difference": diff_val,
                "Reason": reason or status,
            })

    return pd.DataFrame(rows), pd.DataFrame(alerts)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Remittance Cleaner (MVP)", layout="wide")

st.title("Remittance Cleaner — MVP")
st.caption("Paste text or upload files. Run reconciliation. Download CSV + alerts.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Remittance")
    remit_text = st.text_area("Paste remittance text (email body)", height=220, placeholder="Paste remittance email text here…")
    remit_upload = st.file_uploader("Upload remittance file (PDF/image/text)", type=["pdf", "png", "jpg", "jpeg", "webp", "txt"])

with col2:
    st.subheader("Statement of Account")
    stmt_text = st.text_area("Paste statement text (copied/exported)", height=220, placeholder="Paste statement text here…")
    stmt_upload = st.file_uploader("Upload statement PDF/image/text", type=["pdf", "png", "jpg", "jpeg", "webp", "txt"])

run = st.button("Run Reconciliation", type="primary")

if run:
    # prefer upload text if provided, else pasted text; if both, combine (upload + paste)
    remit_from_upload = extract_text_from_upload(remit_upload) if remit_upload else ""
    stmt_from_upload = extract_text_from_upload(stmt_upload) if stmt_upload else ""

    remit_combined = "\n".join([x for x in [remit_from_upload, remit_text] if x.strip()])
    stmt_combined = "\n".join([x for x in [stmt_from_upload, stmt_text] if x.strip()])

    if not remit_combined.strip() or not stmt_combined.strip():
        st.error("You must provide BOTH remittance and statement text (paste or upload).")
        st.stop()

    remit_map = parse_remittance(remit_combined)
    stmt_map = parse_statement(stmt_combined)

    df_out, df_alerts = reconcile(remit_map, stmt_map)

    st.success(f"Done. Invoices found — Remittance: {len(remit_map)} | Statement: {len(stmt_map)} | Output: {len(df_out)}")

    st.subheader("Reconciliation Output (preview)")
    st.dataframe(df_out, use_container_width=True)

    st.subheader("Alerts (preview)")
    st.dataframe(df_alerts, use_container_width=True)

    out_csv = df_out.to_csv(index=False).encode("utf-8")
    alerts_csv = df_alerts.to_csv(index=False).encode("utf-8")

    st.download_button("Download reconciliation_output.csv", data=out_csv, file_name="reconciliation_output.csv", mime="text/csv")
    st.download_button("Download alerts.csv", data=alerts_csv, file_name="alerts.csv", mime="text/csv")
