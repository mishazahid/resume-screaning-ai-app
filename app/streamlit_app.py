"""
streamlit_app.py
================
Main entry point for the Resume Screening AI Streamlit application.

Run from the project root with:
    streamlit run app/streamlit_app.py

The app allows users to:
  1. Paste a job description into a text area.
  2. Upload one or more PDF resumes.
  3. Click "Screen Resumes" to trigger the hybrid scoring pipeline.
  4. View a ranked, explainable results panel with score breakdowns,
     matched / missing skills, and a downloadable CSV report.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup — ensures src.* imports work on Streamlit Cloud and locally
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Absolute path to the project root — used for loading sample data files
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.parser import extract_text, normalize_jd
from src.skill_extractor import extract_skills, match_skills
from src.matcher import compute_hybrid_score, get_model, generate_explanation

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Screener",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Model warm-up — cached so the model loads only once per server process.
# Called at module level (outside any if-block) so Streamlit warms it on
# first page load rather than on the first button click.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_screening_model():
    """Load and cache the SentenceTransformer model for the session."""
    return get_model()


# Trigger warm-up immediately
load_screening_model()


# ---------------------------------------------------------------------------
# Helper: build colour for a sub-score bar
# ---------------------------------------------------------------------------
def _bar_colour(score: float) -> str:
    """Return a hex colour string based on score thresholds."""
    if score >= 0.7:
        return "#28a745"   # green
    elif score >= 0.4:
        return "#fd7e14"   # orange
    else:
        return "#dc3545"   # red


# ---------------------------------------------------------------------------
# Helper: render skill pills as HTML badges
# ---------------------------------------------------------------------------
def _skill_pills(skills: list[str], bg: str, fg: str) -> str:
    """
    Build an HTML string of coloured pill badges for a list of skills.

    Parameters
    ----------
    skills : list[str]
        Skill names to render.
    bg : str
        CSS background colour.
    fg : str
        CSS text colour.

    Returns
    -------
    str
        HTML fragment safe to pass to st.markdown(..., unsafe_allow_html=True).
    """
    if not skills:
        return "<em style='color:#888;'>None found</em>"
    pills = "".join(
        f"<span style='"
        f"background:{bg}; color:{fg}; padding:3px 10px; "
        f"border-radius:12px; margin:3px; display:inline-block; font-size:13px"
        f"'>{skill}</span>"
        for skill in skills
    )
    return f"<div style='line-height:2.2;'>{pills}</div>"


# ---------------------------------------------------------------------------
# Helper: build Plotly sub-score bar chart
# ---------------------------------------------------------------------------
def _build_score_chart(scores: dict) -> go.Figure:
    """
    Build a horizontal bar chart showing the four sub-scores.

    Parameters
    ----------
    scores : dict
        Must have keys: semantic_score, skill_score, experience_score,
        education_score.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    labels = ["Semantic\nSimilarity", "Skill\nMatch", "Experience", "Education"]
    values = [
        scores["semantic_score"],
        scores["skill_score"],
        scores["experience_score"],
        scores["education_score"],
    ]
    colours = [_bar_colour(v) for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=10, b=10),
        height=180,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: build downloadable CSV
# ---------------------------------------------------------------------------
def _build_csv(results: list[dict]) -> str:
    """
    Convert the list of screening results into a CSV string for download.

    Parameters
    ----------
    results : list[dict]
        Each element is the merged dict of resume_data + score_data + skill_match.

    Returns
    -------
    str
        UTF-8 CSV string.
    """
    rows = []
    for rank, r in enumerate(results, start=1):
        rows.append(
            {
                "Rank": rank,
                "Filename": r["filename"],
                "Final Score %": r["scores"]["final_score_pct"],
                "Semantic Score": r["scores"]["semantic_score"],
                "Skill Score": r["scores"]["skill_score"],
                "Experience Score": r["scores"]["experience_score"],
                "Education Score": r["scores"]["education_score"],
                "Recommendation": r["scores"]["recommendation"],
                "Matched Skills": ", ".join(r["skill_match"]["matched"]),
                "Missing Skills": ", ".join(r["skill_match"]["missing"]),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# ---------------------------------------------------------------------------
# Helper: detect education level label for display
# ---------------------------------------------------------------------------
def _education_label(text: str) -> str:
    """Return a human-readable education label from resume text."""
    t = text.lower()
    if any(k in t for k in ["phd", "ph.d", "doctorate", "doctoral"]):
        return "PhD / Doctorate"
    if any(k in t for k in ["master", "msc", "m.sc", "mba", "m.eng", "ms "]):
        return "Master's"
    if any(k in t for k in ["bachelor", "bsc", "b.sc", "b.eng", "be ", "b.e", "undergraduate"]):
        return "Bachelor's"
    if any(k in t for k in ["associate", "diploma", "hnd"]):
        return "Associate / Diploma"
    return "Not detected"


# ---------------------------------------------------------------------------
# Helper: detect years from resume for display
# ---------------------------------------------------------------------------
def _experience_label(text: str) -> str:
    """Return a human-readable experience label from resume text."""
    import re
    matches = re.findall(r"(\d+)\+?\s*years?", text, re.IGNORECASE)
    if matches:
        return f"{max(int(m) for m in matches)} years"
    return "Not detected"


# ===========================================================================
# MAIN APP
# ===========================================================================

# ---------------------------------------------------------------------------
# SECTION 1: Header
# ---------------------------------------------------------------------------
st.title("Resume Screening AI")
st.markdown(
    "Paste a job description, upload resumes, get ranked candidates instantly."
)
st.divider()

# ---------------------------------------------------------------------------
# SECTION 2: Input panel
# ---------------------------------------------------------------------------
col_jd, col_upload = st.columns(2)

with col_jd:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the full job description here",
        height=300,
        key="jd_input",
        placeholder="Paste the complete job description here...",
    )
    word_count = len(jd_text.split()) if jd_text.strip() else 0
    st.caption(f"Word count: **{word_count}**")

with col_upload:
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_files",
        label_visibility="collapsed",
    )
    file_count = len(uploaded_files) if uploaded_files else 0
    st.caption(f"{file_count} resume(s) uploaded")
    st.info(
        "Tip: Upload at least 3–5 resumes for a meaningful comparison.",
        icon="💡",
    )
    st.markdown("**Or try with sample data:**")
    load_sample = st.button("Load sample resumes + JD", use_container_width=True)
    # Show persistent success banner after sample data has been loaded
    if st.session_state.get("sample_loaded") and not load_sample:
        st.success("Sample data loaded! Click Screen Resumes to see results.")

# ---------------------------------------------------------------------------
# Handle "Load sample resumes + JD" button click
# ---------------------------------------------------------------------------
if load_sample:
    # Read sample JD and pre-fill the text_area via its session_state key
    sample_jd_path = os.path.join(PROJECT_ROOT, "data", "sample_jd.txt")
    try:
        with open(sample_jd_path, "r", encoding="utf-8") as _f:
            st.session_state["jd_input"] = _f.read()
    except Exception as _e:
        st.error(f"Could not load sample JD: {_e}")

    # Read each sample resume txt and store as dicts
    sample_dir = os.path.join(PROJECT_ROOT, "data", "sample_resumes")
    _sample_fnames = [
        "candidate_alice.txt",
        "candidate_bob.txt",
        "candidate_carol.txt",
    ]
    _loaded_samples: list[dict] = []
    for _fname in _sample_fnames:
        _fpath = os.path.join(sample_dir, _fname)
        try:
            with open(_fpath, "r", encoding="utf-8") as _f:
                _loaded_samples.append({"filename": _fname, "content": _f.read()})
        except Exception as _e:
            st.warning(f"Could not load sample file '{_fname}': {_e}")

    st.session_state["sample_files"] = _loaded_samples
    st.session_state["sample_loaded"] = True
    # Force a rerun so the text_area picks up the new session_state value
    st.rerun()

# ---------------------------------------------------------------------------
# SECTION 3: Action button
# ---------------------------------------------------------------------------
st.write("")  # Vertical spacing
btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
with btn_col2:
    clicked = st.button(
        "Screen Resumes",
        type="primary",
        use_container_width=True,
    )

# Validate inputs
if clicked:
    if not jd_text.strip():
        st.warning("Please paste a job description first.")
        clicked = False
    elif not uploaded_files and not st.session_state.get("sample_files"):
        st.warning("Please upload at least one resume (or use 'Load sample resumes + JD').")
        clicked = False

# ---------------------------------------------------------------------------
# SECTION 4: Results
# ---------------------------------------------------------------------------
if clicked:
    # Clear the sample-loaded banner once the user starts a real screening run
    st.session_state["sample_loaded"] = False

    with st.spinner("Analysing resumes… this may take a moment"):

        # Pre-process the JD once
        jd_data = normalize_jd(jd_text)
        jd_skills = extract_skills(jd_data["cleaned"])

        all_results: list[dict] = []

        # Uploaded PDFs take priority; fall back to session_state sample files
        if uploaded_files:
            _file_items = list(uploaded_files)
            _using_samples = False
        else:
            # Convert sample dicts to BytesIO objects with a .name attribute
            _file_items = []
            for _s in st.session_state.get("sample_files", []):
                _buf = io.BytesIO(_s["content"].encode("utf-8"))
                _buf.name = _s["filename"]
                _file_items.append(_buf)
            _using_samples = True

        for file_item in _file_items:
            # --- Parse the file (PDF or txt) ---
            resume_data = extract_text(file_item)

            if resume_data["error"]:
                # Store a sentinel result so we can display an error card
                all_results.append(
                    {
                        "filename": resume_data["filename"],
                        "resume_data": resume_data,
                        "resume_skills": [],
                        "skill_match": {
                            "matched": [],
                            "missing": jd_skills,
                            "extra": [],
                            "match_ratio": 0.0,
                            "skill_score": 0.0,
                        },
                        "scores": {
                            "final_score": 0.0,
                            "final_score_pct": 0.0,
                            "semantic_score": 0.0,
                            "tfidf_score": 0.0,
                            "skill_score": 0.0,
                            "experience_score": 0.0,
                            "education_score": 0.0,
                            "weights": {},
                            "recommendation": "Weak fit",
                            "detected_resume_years": None,
                            "detected_jd_years": None,
                        },
                        "explanation": "",
                        "parse_error": resume_data["error"],
                    }
                )
                continue

            # --- Extract and match skills ---
            resume_skills = extract_skills(resume_data["cleaned_text"])
            skill_match = match_skills(resume_skills, jd_skills)

            # --- Compute hybrid score ---
            scores = compute_hybrid_score(jd_data, resume_data, skill_match)

            # --- Generate plain-language explanation ---
            explanation = generate_explanation(scores, skill_match)

            all_results.append(
                {
                    "filename": resume_data["filename"],
                    "resume_data": resume_data,
                    "resume_skills": resume_skills,
                    "skill_match": skill_match,
                    "scores": scores,
                    "explanation": explanation,
                    "parse_error": None,
                }
            )

        # Sort by final score descending
        all_results.sort(key=lambda r: r["scores"]["final_score"], reverse=True)

        # Persist results in session_state so they survive reruns
        st.session_state["screening_results"] = all_results
        st.session_state["jd_skills"] = jd_skills


# Display results if available (either from this run or a previous one)
if "screening_results" in st.session_state and st.session_state["screening_results"]:
    results = st.session_state["screening_results"]

    # ── A. Summary metrics ────────────────────────────────────────────────
    st.divider()
    st.subheader("Screening Summary")

    total = len(results)
    top = results[0]
    top_name = top["filename"].replace(".pdf", "").replace("_", " ").replace("-", " ").title()
    top_pct = top["scores"]["final_score_pct"]
    avg_pct = round(
        sum(r["scores"]["final_score_pct"] for r in results) / total, 1
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resumes Screened", total)
    m2.metric("Top Candidate", top_name)
    m3.metric("Top Score", f"{top_pct}%")
    m4.metric("Average Score", f"{avg_pct}%")

    # ── B. Ranked results ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Ranked Candidates")

    for rank, result in enumerate(results, start=1):
        fname = result["filename"]
        score_pct = result["scores"]["final_score_pct"]
        rec = result["scores"]["recommendation"]

        expander_label = f"#{rank} — {fname} — {score_pct}% — {rec}"

        with st.expander(expander_label, expanded=(rank == 1)):

            # Parse error banner
            if result.get("parse_error"):
                st.error(
                    f"Could not parse this resume: {result['parse_error']}"
                )
                st.stop()

            scores = result["scores"]
            skill_match = result["skill_match"]
            resume_data = result["resume_data"]

            col_score, col_skills, col_details = st.columns([1, 1, 1])

            # ── Column 1: Score breakdown ─────────────────────────────
            with col_score:
                st.metric(
                    "Overall Score",
                    f"{score_pct}%",
                    help="Weighted hybrid score: 50% semantic + 30% skills + 15% experience + 5% education",
                )

                # Recommendation badge
                if rec == "Strong fit":
                    st.success(f"**{rec}**")
                elif rec == "Good fit":
                    st.info(f"**{rec}**")
                elif rec == "Partial fit":
                    st.warning(f"**{rec}**")
                else:
                    st.error(f"**{rec}**")

                # AI Insight card — bulleted breakdown for the recruiter
                explanation = result.get("explanation", "")
                if explanation:
                    st.markdown(
                        f"<div style='"
                        f"border-left: 3px solid #4a90d9; "
                        f"background: rgba(74,144,217,0.08); "
                        f"padding: 8px 14px 8px 10px; "
                        f"border-radius: 0 6px 6px 0; "
                        f"margin-top: 10px;'>"
                        f"<span style='font-size:12px; font-weight:600; "
                        f"letter-spacing:.4px; opacity:.7;'>AI INSIGHT</span>"
                        f"{explanation}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Sub-score bar chart
                fig = _build_score_chart(scores)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{rank}")

            # ── Column 2: Skills analysis ─────────────────────────────
            with col_skills:
                st.subheader("Matched Skills")
                st.markdown(
                    _skill_pills(
                        skill_match["matched"],
                        bg="#d4edda",
                        fg="#155724",
                    ),
                    unsafe_allow_html=True,
                )

                st.write("")  # spacing
                st.subheader("Missing Skills")
                st.markdown(
                    _skill_pills(
                        skill_match["missing"],
                        bg="#f8d7da",
                        fg="#721c24",
                    ),
                    unsafe_allow_html=True,
                )

                if skill_match["extra"]:
                    st.write("")
                    st.caption(
                        f"**{len(skill_match['extra'])} additional skills** "
                        "in resume not required by JD: "
                        + ", ".join(skill_match["extra"][:10])
                        + ("…" if len(skill_match["extra"]) > 10 else "")
                    )

            # ── Column 3: Details ─────────────────────────────────────
            with col_details:
                st.subheader("Details")

                tfidf = scores["tfidf_score"]
                semantic = scores["semantic_score"]
                st.write(f"**Semantic score:** {semantic:.2%}")
                st.write(f"**TF-IDF baseline:** {tfidf:.2%}")

                delta = semantic - tfidf
                direction = "above" if delta >= 0 else "below"
                st.caption(
                    f"Semantic is {abs(delta):.2%} {direction} TF-IDF baseline"
                )

                st.divider()

                # Experience — show accurate duration including sub-year
                detected_years = scores.get("detected_resume_years")
                detected_months = scores.get("detected_resume_months", 0)
                if detected_years is None:
                    st.write("**Experience detected:** Not detected")
                    st.caption(
                        "Tip: Years may be written as '2019–present' "
                        "or '5+ years experience'"
                    )
                elif detected_years == 0 and detected_months > 0:
                    # Sub-year: show exact months, don't inflate to 1 year
                    st.write(
                        f"**Experience detected:** {detected_months} months "
                        f"(< 1 year)"
                    )
                else:
                    # Full years — optionally show remaining months too
                    leftover = detected_months % 12 if detected_months else 0
                    if leftover:
                        st.write(
                            f"**Experience detected:** {detected_years} yr "
                            f"{leftover} mo"
                        )
                    else:
                        st.write(
                            f"**Experience detected:** {detected_years} years"
                        )

                # Education
                edu_label = _education_label(resume_data.get("raw_text", ""))
                st.write(f"**Education detected:** {edu_label}")

                st.divider()

                # Section detection summary
                st.write("**Sections found in resume:**")
                sections = resume_data.get("sections", {})
                section_names = ["summary", "skills", "experience", "education", "projects"]
                for sec in section_names:
                    found = bool(sections.get(sec, "").strip())
                    icon = "✓" if found else "–"
                    colour = "green" if found else "#999"
                    st.markdown(
                        f"<span style='color:{colour}'>{icon}</span> {sec.capitalize()}",
                        unsafe_allow_html=True,
                    )

    # ── C. Download CSV ───────────────────────────────────────────────────
    st.divider()
    csv_data = _build_csv(results)
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="resume_screening_results.csv",
        mime="text/csv",
        use_container_width=False,
    )

# ---------------------------------------------------------------------------
# SECTION 5: Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Resume Screening AI — built with Sentence-BERT, scikit-learn, and Streamlit. "
    "Scores are decision-support signals, not automated hiring decisions."
)
