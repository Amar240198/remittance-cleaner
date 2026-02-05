import re
from typing import Dict, Optional

import pandas as pd
import streamlit as st


# =========================
# UTILITIES
# =========================
def parse_money(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).replace("£", "").replace("€", "").replace("$", "")
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


# =========================
# PARSERS (TEXT ONLY)
# =========================
def parse_statement(text: str) -> Dict[str, float]:
    """
    Invoice -> Expected Total (from statement)
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
    Invoice -> Paid Total (from remittance)
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


# =========================
# RECONCILIATION
# =========================
def reconcile(statement: Dict[str, float], remittance: Dict[str, float]) -> pd.DataFrame:
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
            })
            continue

        diff = round(paid - expected, 2)

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
            "Expected_Total": expected,
            "Paid_Total": paid,
            "Difference": diff,
            "Status": status,
        })

    return pd.DataFrame(rows)


# =========================
# STREAMLIT UI (PASTE ONLY)
# =========================
st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")
st.title("Remittance Cleaner — MVP")
st.caption(
    "Paste statement and remittance text below. "
    "This MVP processes text only to ensure maximum reliability."
)

left, right = st.columns(2)

with left:
    st.subheader("Remittance (Paste Text)")
    remit_text = st.text_area(
        "Paste remittance email or text",
        height=220,
        placeholder="Paste remittance text here…"
    )

with right:
    st.subheader("Statement of Account (Paste Text)")
    stmt_text = st.text_area(
        "Paste statement text",
        height=220,
        placeholder="Paste statement text here…"
    )

st.divider()

if st.button("Run Reconciliation", type="primary"):
    if not remit_text.strip() or not stmt_text.strip():
        st.error("Both remittance and statement text are required.")
        st.stop()

    try:
        stmt_map = parse_statement(stmt_text)
        remit_map = parse_remittance(remit_text)

        if not stmt_map or not remit_map:
            st.warning(
                "⚠️ We couldn’t reliably extract invoices and amounts.\n\n"
                "Make sure invoice numbers and amounts appear on the same lines."
            )
            st.stop()

        df = reconcile(stmt_map, remit_map)

        st.success("Reconciliation complete")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "reconciliation_output.csv",
            "text/csv"
        )

    except Exception:
        st.error(
            "This app encountered an error while parsing.\n\n"
            "Please check the pasted text and try again."
        )
