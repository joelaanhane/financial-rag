import os
import pickle
import json
import numpy as np
import faiss
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def safe_key(name):
    """Convert bank name to a safe folder name, e.g. 'ABN AMRO' -> 'abnamro'.

    Used both for the on-disk folder layout and to detect when two differently
    cased/spelled bank names (e.g. 'ABN AMRO' vs 'abn amro') refer to the same bank.
    """
    return name.lower().replace(" ", "").replace("-", "")


def _report_dir(name, year):
    """Return the directory for a specific report, creating it if needed."""
    path = os.path.join(REPORTS_DIR, str(year), safe_key(name))
    os.makedirs(path, exist_ok=True)
    return path


def save_report(name, year, chunks, embeddings, n_pages):
    """Save chunks, embeddings and metadata for a report to disk.

    Args:
        name: Human-readable bank name, e.g. 'ING'.
        year: Report year as string or int.
        chunks: List of chunk dicts from rag.chunk_text().
        embeddings: NumPy array of shape (n_chunks, 1536).
        n_pages: Total number of pages in the source PDF.
    """
    d = _report_dir(name, year)
    np.save(os.path.join(d, "embeddings.npy"), embeddings)
    with open(os.path.join(d, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    meta = {
        "name": name,
        "year": str(year),
        "pages": n_pages,
        "chunks": len(chunks),
        "uploaded_at": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_report(name, year):
    """Load chunks and build a FAISS index for a stored report.

    Args:
        name: Bank name as stored (e.g. 'ING').
        year: Report year as string or int.
    Returns:
        Tuple of (chunks list, faiss IndexFlatL2).
    """
    d = os.path.join(REPORTS_DIR, str(year), safe_key(name))
    embeddings = np.load(os.path.join(d, "embeddings.npy"))
    with open(os.path.join(d, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index


def list_reports():
    """Return a list of metadata dicts for all stored reports, newest first.

    Returns:
        List of dicts with keys: name, year, pages, chunks, uploaded_at.
    """
    reports = []
    if not os.path.exists(REPORTS_DIR):
        return reports
    for year in os.listdir(REPORTS_DIR):
        year_dir = os.path.join(REPORTS_DIR, year)
        if not os.path.isdir(year_dir):
            continue
        for bank in os.listdir(year_dir):
            meta_path = os.path.join(year_dir, bank, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    reports.append(json.load(f))
    return sorted(reports, key=lambda r: (r["year"], r["name"]), reverse=True)


def report_exists(name, year):
    """Check whether a report has already been processed and stored.

    Args:
        name: Bank name.
        year: Report year.
    Returns:
        True if the meta.json for this report exists on disk.
    """
    meta_path = os.path.join(REPORTS_DIR, str(year), safe_key(name), "meta.json")
    return os.path.exists(meta_path)


def known_banks():
    """Return the canonical display name of every distinct bank in the library.

    Reports are deduped by safe_key(), so 'ABN AMRO' and 'abn amro' count as one
    bank; the display name used is whichever came from the most recently uploaded
    report for that bank (list_reports() is already sorted newest-first).

    Returns:
        Sorted list of unique bank display names.
    """
    seen = {}
    for r in list_reports():
        seen.setdefault(safe_key(r["name"]), r["name"])
    return sorted(seen.values(), key=str.lower)


def find_canonical_name(name):
    """Check whether a bank name matches one already in the library.

    Args:
        name: Bank name to check, in any casing/spacing.
    Returns:
        The existing display name if a bank with the same safe_key is already
        stored, otherwise None.
    """
    key = safe_key(name)
    for r in list_reports():
        if safe_key(r["name"]) == key:
            return r["name"]
    return None
