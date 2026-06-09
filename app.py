import streamlit as st
from storage import list_reports

st.set_page_config(page_title="AnnualSight", page_icon="📊", layout="wide")

st.title("📊 AnnualSight")
st.caption("AI-powered analysis of financial annual reports")

st.markdown("""
Use the sidebar to navigate:

- **Upload** — Add a new annual report PDF
- **Library** — View all uploaded reports
- **Analyse** — Ask questions about your reports
""")

reports = list_reports()

if reports:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Reports", len(reports))
    col2.metric("Years covered", len(set(r["year"] for r in reports)))
    col3.metric("Total pages", sum(r["pages"] for r in reports))
