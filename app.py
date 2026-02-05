import re
from typing import Dict, Optional
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =========================
# APP META
# =========================
st.set_page_config(
    page_title="Smart Remittance Paste",
    layout="wide"
)

st.title("Smart Remittance Paste")
st.markdown("""
**Excel-grade paste for remittance emails.**

Paste directly from Outlook or Gmail.
This app preserves invoice ↔ amount structure automatically,
even when the source email uses HTML tables.

**What this solves**
- Broken copy/paste from remittance emails
- Amounts dropping under invoice numbers
- Manual Excel cleanup
- Reconciliation errors

One paste. No Excel. No formatting fixes.
""")

# =========================
# SMART PASTE (FRONTEND)
# =========================
smart_paste = components.html(
"""
<script>
function sendToStreamlit(rows) {
  const text = rows.map(r => r.join("\\t")).join("\\n");
  window.parent.postMessage(
    { type: "streamlit:setComponentValue", value: text },
    "*"
  );
}

document.addEventListener("paste", (e) => {
  e.preventDefault();

  const html = e.clipboardData.getData("text/html");
  const plain = e.clipboardData.getData("text/plain");
  let rows = [];

  if (html && html.includes("<table")) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");

    doc.querySelectorAll("tr").forEach(tr => {
      const cells = [...tr.querySelectorAll("td, th")]
        .map(td => td.innerText.trim());
      if (cells.length >= 2) {
        rows.push([cells[0], cells[1]]);
      }
    });
  } else {
    const lines = plain.split(/\\r?\\n/)
      .map(l => l.trim())
      .filter(Boolean);

    for (let i = 0; i < lines.length; i += 2) {
      if (lines[i + 1]) {
        rows.push([lines[i], lines[i + 1]]);
      }
    }
  }

  if (rows.length) sendToStreamlit(rows);
});
</script>

<div style="padding:12px;border:1px dashed #999;">
  <strong>Paste remittance here (Ctrl/Cmd + V)</strong><br/>
  Column fidelity is enforced automatically.
</div>
""",
height=120,
)

# =========================
# UTILITIES
# =========================
def parse_money(x: str) -> Optional[float]:
    if not x:
        return None
    s = str(x).replace("£", "").replace("€", "").replace("$", "")
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def normalize_invoice(inv: str) -> str:
    inv = inv.upper().replace("INV", "")
    return inv.lstrip("0")


def find_invoices(text: str):
    raw = re.findall(r"\b(?:INV)?0*\d{5,12}\b", text.upper())
    return [normalize_invoice(r) for r in raw]


# =========================
# PARSER
# =========================
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
# MAIN UI
# =========================
st.subheader("Detected Remittance Lines")

normalized_text = st.text_area(
    "Normalized output (auto-filled)",
    value=smart_paste or "",
    height=200
)

if normalized_text:
    remittance = parse_remittance(normalized_text)

    df = pd.DataFrame(
        [{"Invoice Number": k, "Amount": v} for k, v in remittance.items()]
    )

    st.dataframe(df, use_container_width=True)
