import streamlit as st
import tempfile
import os
from datetime import date
from rag import load_pdf, chunk_text, get_embeddings
from storage import save_report, report_exists, known_banks, find_canonical_name

st.set_page_config(page_title="Upload — AnnualSight", page_icon="📤", layout="wide")

st.title("📤 Upload Annual Report")
st.caption("Upload a PDF to process and store it in the library.")

ADD_NEW = "+ Add new bank"
current_year = date.today().year
years = [str(y) for y in range(current_year, current_year - 50, -1)]

col1, col2 = st.columns(2)
with col1:
    banks = known_banks()
    if banks:
        choice = st.selectbox("Bank name", banks + [ADD_NEW])
        bank_name = st.text_input("New bank name", placeholder="e.g. Deutsche Bank") if choice == ADD_NEW else choice
    else:
        bank_name = st.text_input("Bank name", placeholder="e.g. ING, ABN AMRO, Rabobank")

    if bank_name:
        canonical = find_canonical_name(bank_name)
        if canonical and canonical != bank_name:
            st.info(f"This matches **{canonical}** already in your library — using that name for consistency.")
            bank_name = canonical
with col2:
    year = st.selectbox("Year", years)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and bank_name:
    if report_exists(bank_name, year):
        st.warning(f"**{bank_name} {year}** is already in the library. Upload a different report or choose another year.")
    else:
        st.info(f"Ready to process **{bank_name} {year}** ({uploaded_file.size // 1024} KB)")

if st.button("Process and save", type="primary", disabled=not (uploaded_file and bank_name)):
    if not bank_name:
        st.error("Please enter a bank name.")
        st.stop()
    if not uploaded_file:
        st.error("Please upload a PDF.")
        st.stop()
    if report_exists(bank_name, year):
        st.error(f"{bank_name} {year} already exists. Remove it from the library first.")
        st.stop()

    # Write the uploaded bytes to a temp file so PyMuPDF can open it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.status("Processing PDF...", expanded=True) as status:
            st.write("Extracting text from pages...")
            pages = load_pdf(tmp_path)
            st.write(f"Found **{len(pages)} pages**.")

            st.write("Splitting into chunks...")
            chunks = chunk_text(pages)
            st.write(f"Created **{len(chunks)} chunks** (500 words, 50 overlap).")

            st.write("Creating embeddings via OpenAI... (this may take a minute)")
            progress_bar = st.progress(0)
            embeddings = get_embeddings(chunks, progress_callback=lambda p: progress_bar.progress(p))
            st.write(f"Embeddings shape: {embeddings.shape}")

            st.write("Saving to library...")
            save_report(bank_name, year, chunks, embeddings, len(pages))

            status.update(label="Done!", state="complete")

        st.success(f"**{bank_name} {year}** saved to the library — {len(pages)} pages, {len(chunks)} chunks.")
        st.balloons()

    finally:
        os.unlink(tmp_path)
