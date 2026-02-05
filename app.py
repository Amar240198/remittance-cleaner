import re
from typing import Dict, Optional

import pandas as pd
import streamlit as st


# =========================
# FORCE UI RENDER FIRST
# =========================
st.set_page_config(page_title="Remittance Cleaner — MVP", layout="wide")
st.title("Remittance Cleaner — MVP")
st.caption(
    "Paste statement and remittance text below. "
    "Statement Balance comes from the statement. Difference = Paid − Statement Balance."
)

# If anything below fails, UI will still render
st.write("")  # defensive render anchor


# =========================
# UTILITIES
# =========================
def parse_money(x: str) -> Optional[float]:
    try:
        s = str(x).replace("£", "").replace("€", "").replace("$", "")
        s = s.replace(",", "").strip()
        return float(s)
    except Exception:
        return None


def normalize_invoice(inv: str) -> str:
    inv = re.sub(r"[A-Z]", "", inv.upper())
    return inv.lstrip("0")


def find_invoices(text: str):
    return [
        normalize_invoice(x)
        for x in re.findall(r"\b(?:INV)?0*\d{5,12}\b", text.upper())
    ]


# =========================
# PARSERS
# =========================
def parse_statement(text: str) -> Dict[str, float]:
    result = {}
    for line in text.splitlines():
        invoices = find_invoices(line)
        amounts = re.findall(r"\d+\.\d{2}", line)
        if invoices and amounts:
            balance = parse_money(amounts[-1])
            for inv in invoices:
                result[inv] = balance
    return result


def parse_remittance(text: str) -> Dict[str, float]:
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
        statement_balance = statement.get(inv)
        paid_total = remittance.get(inv, 0.0)

        if statement_balance is None:
            rows.append({
                "Invoice_Number": inv,
                "Statement_Balance": None,
                "Paid_Total": paid_total,
                "Difference": None,
                "Status": "MISSING FROM STATEMENT",
            })
            continue

        difference = round(paid_total - statement_balance, 2)

        if paid_total == 0:
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
            "Statement_Balance": statement_balance,
            "Paid_Total": paid_total,
            "Difference": difference,
            "Status": status,
        })

    return pd.DataFrame(rows)


# =========================
# UI INPUTS
# =========================
left, right = st.columns(2)

with left:
    st.subheader("Remittance (Paste Text)")
    remit_text = st.text_area("Paste remittance text", height=220)

with right:
    st.subheader("Statement of Account (Paste Text)")
    stmt_text = st.text_area("Paste statement text", height=220)

st.divider()

# =========================
# ACTION
# =========================
if st.button("Run Reconciliation", type="primary"):
    try:
        if not remit_text.strip() or not stmt_text.strip():
            st.error("Both remittance and statement text are required.")
            st.stop()

        stmt_map = parse_statement(stmt_text)
        remit_map = parse_remittance(remit_text)

        if not stmt_map or not remit_map:
            st.warning(
                "⚠️ Unable to extract invoices or balances.\n\n"
                "Ensure invoice numbers and amounts appear on the same lines."
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

    except Exception as e:
        st.error("This app encountered an unexpected error.")
        st.code(str(e))
