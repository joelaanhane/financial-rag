import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from storage import list_reports, load_report
from rag import search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Analyse — AnnualSight", page_icon="🔍", layout="wide")

st.title("🔍 Analyse Reports")
st.caption("Ask a question about one or more annual reports.")

reports = list_reports()

if not reports:
    st.info("No reports in the library yet. Go to **Upload** to add a report first.")
    st.stop()

# Report selection
report_options = {f"{r['name']} {r['year']}": r for r in reports}
selected_labels = st.multiselect(
    "Select reports to search",
    options=list(report_options.keys()),
    default=list(report_options.keys())[:1],
    help="You can select multiple reports to compare across banks or years.",
)

# Question input
question = st.text_input(
    "Your question",
    placeholder="e.g. What was the net profit? How did capital ratios change?",
)

ask_disabled = not (question and selected_labels)

if st.button("Ask", type="primary", disabled=ask_disabled):
    all_context = ""
    all_sources = []

    with st.spinner(f"Searching {len(selected_labels)} report(s)..."):
        for label in selected_labels:
            r = report_options[label]
            chunks, index = load_report(r["name"], r["year"])
            results = search(question, chunks, index, k=3)
            for c in results:
                all_context += f"\n\n[{r['name']} {r['year']}, Page {c['page']}]: {c['text']}"
                all_sources.append({
                    "label": label,
                    "page": c["page"],
                    "text": c["text"],
                })

    with st.spinner("Generating answer..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst. Answer based only on the provided context. "
                        "Always cite which bank, year, and page number your information comes from. "
                        "If the answer is not in the context, say 'NOT FOUND'."
                    ),
                },
                {"role": "user", "content": f"Context:\n{all_context}\n\nQuestion: {question}"},
            ],
        )
        answer = response.choices[0].message.content

    st.markdown("### Answer")
    st.write(answer)

    with st.expander(f"📄 Source chunks ({len(all_sources)} used)"):
        for s in all_sources:
            st.markdown(f"**{s['label']} — Page {s['page']}**")
            st.caption(s["text"])
            st.divider()
