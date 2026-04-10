# Resume Screening AI

A production-ready hybrid resume screener that ranks candidates against a job description using semantic similarity, skill matching, experience alignment, and education relevance — all in a clean Streamlit interface.

## Screenshot

![App Screenshot](screenshot.png)

---

## Features

- **Paste & Upload** — paste any job description and upload multiple PDF resumes in one step
- **Hybrid scoring** — combines four independent signals (semantic, skills, experience, education) into one weighted final score
- **Explainable results** — every candidate gets a score breakdown chart plus matched / missing skill pills
- **Section detection** — automatically locates Summary, Skills, Experience, Education, and Projects sections in each resume
- **Ranked leaderboard** — candidates sorted by score with recommendation labels (Strong fit → Weak fit)
- **CSV export** — download the full ranked results table for further analysis or reporting
- **Fast** — SentenceTransformer model loaded once and cached; subsequent screenings are near-instant

---

## How It Works

```
1. PARSE   — PyMuPDF extracts raw text from each uploaded PDF; sections are
             detected via regex heading patterns

2. EXTRACT — A curated 70+ skill dictionary with aliases identifies canonical
             skills in both the resume and the JD using whole-word regex

3. SCORE   — Four sub-scores are computed independently:
               • Semantic similarity  (Sentence-BERT cosine similarity)
               • Skill match          (Jaccard-style overlap ratio)
               • Experience alignment (years-of-experience regex extraction)
               • Education relevance  (degree-level keyword matching)

4. RANK    — Sub-scores are combined with fixed weights into a final score
             and candidates are sorted from highest to lowest
```

---

## Scoring Formula

```
final_score = (semantic_score  × 0.50)
            + (skill_score     × 0.30)
            + (experience_score× 0.15)
            + (education_score × 0.05)
```

| Component | Weight | Method |
|-----------|--------|--------|
| **Semantic similarity** | 50% | `all-MiniLM-L6-v2` sentence embeddings + cosine similarity |
| **Skill match** | 30% | `matched_skills / jd_skills` ratio |
| **Experience alignment** | 15% | Regex-extracted years compared to JD requirement |
| **Education relevance** | 5% | Degree-level keyword detection (PhD > Master's > Bachelor's > Associate) |

All sub-scores are normalised to **[0.0 – 1.0]** before weighting.

**Recommendation thresholds**

| Score | Label |
|-------|-------|
| ≥ 75% | Strong fit |
| ≥ 55% | Good fit |
| ≥ 35% | Partial fit |
| < 35% | Weak fit |

---

## Setup Instructions

### a. Clone the repository

```bash
git clone https://github.com/your-username/resume-screener.git
cd resume-screener
```

### b. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### c. Install dependencies

```bash
pip install -r requirements.txt
```

### d. Run the app

```bash
streamlit run app/streamlit_app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo, set the **Main file path** to `app/streamlit_app.py`, and click **Deploy**.

Streamlit Cloud will install `requirements.txt` automatically.

> **Note:** The first deploy may take several minutes as PyTorch and the
> SentenceTransformer model (~90 MB) are downloaded.

---

## Project Structure

```
resume-screener/
├── app/
│   └── streamlit_app.py      # Streamlit UI and orchestration logic
├── src/
│   ├── __init__.py
│   ├── parser.py             # PDF text extraction + section detection
│   ├── skill_extractor.py    # Skill dictionary + matching logic
│   └── matcher.py            # Hybrid scoring engine
├── data/
│   └── sample_jd.txt         # Sample Senior Data Scientist JD for testing
├── requirements.txt
└── README.md
```

---

## Limitations & Future Improvements

**Current limitations**

- Scanned / image-only PDFs are not supported (no OCR layer). A banner is shown
  rather than crashing.
- The skill dictionary covers ~70 common tech/soft skills; niche or domain-specific
  skills may be missed.
- Experience years are extracted via regex and may be inaccurate for candidates who
  list cumulative or overlapping roles.
- Education scoring relies on keyword matching, not verified transcripts.

**Possible improvements**

- Add Tesseract OCR fallback for scanned PDFs
- Allow users to upload a custom skill dictionary (CSV / JSON)
- Named-entity recognition (spaCy NER) for richer section and entity extraction
- Candidate clustering to surface diversity within the shortlist
- ATS-style keyword density heatmap
- Multi-language support (non-English resumes)
- API mode (`POST /screen`) so the engine can be called from external systems

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![SentenceTransformers](https://img.shields.io/badge/sentence--transformers-2.6%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e?logo=scikit-learn)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.23%2B-green)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3f4f75?logo=plotly)
![pandas](https://img.shields.io/badge/pandas-2.2%2B-150458?logo=pandas)
