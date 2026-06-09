import streamlit as st
from storage import list_reports

st.set_page_config(page_title="Library — AnnualSight", page_icon="📚", layout="wide")

st.title("📚 Report Library")
st.caption("All annual reports that have been processed and stored.")

reports = list_reports()

if not reports:
    st.info("No reports yet. Go to **Upload** to add your first annual report.")
    st.stop()

# Summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total reports", len(reports))
col2.metric("Years covered", len(set(r["year"] for r in reports)))
col3.metric("Total pages indexed", sum(r["pages"] for r in reports))

st.divider()

# Report cards grouped by year
years = sorted(set(r["year"] for r in reports), reverse=True)

for year in years:
    st.subheader(year)
    year_reports = [r for r in reports if r["year"] == year]

    for r in year_reports:
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"### {r['name']}")
                st.caption(f"{r['pages']} pages · {r['chunks']} chunks · uploaded {r['uploaded_at']}")
            with right:
                st.markdown(f"**{r['year']}**")
