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
    Invoice -> Statement Balance (expected total from statement)
    """
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
# STRE
