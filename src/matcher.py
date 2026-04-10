"""
matcher.py
==========
Core hybrid scoring engine for the resume-screening pipeline.

Combines four independent signals into one weighted final score:

  final_score = (semantic_score  * 0.50)
              + (skill_score     * 0.30)
              + (experience_score* 0.15)
              + (education_score * 0.05)

Sub-scores are each normalised to [0.0, 1.0] before weighting.

The SentenceTransformer model is loaded once (lazily on first call) and
cached in a module-level variable so subsequent calls are fast.

Public API
----------
get_model()                                  -> SentenceTransformer
compute_semantic_score(jd, resume)           -> float
compute_tfidf_score(jd, resume)              -> float
compute_experience_score(resume, jd)         -> float
compute_education_score(resume, jd)          -> float
compute_hybrid_score(jd_data, resume_data,
                     skill_match)            -> dict
"""

import re
import numpy as np
from typing import Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Model cache — populated on first call to get_model()
# ---------------------------------------------------------------------------
_MODEL: Optional[SentenceTransformer] = None

# Scoring weights (must sum to 1.0)
WEIGHTS: dict[str, float] = {
    "semantic": 0.50,
    "skills": 0.30,
    "experience": 0.15,
    "education": 0.05,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_model() -> SentenceTransformer:
    """
    Return the cached SentenceTransformer model, loading it on first call.

    The model is stored in the module-level ``_MODEL`` variable so it is
    initialised only once per Python process — important for performance
    inside Streamlit which reruns the script on every interaction.

    Returns
    -------
    SentenceTransformer
        Loaded ``all-MiniLM-L6-v2`` model.
    """
    global _MODEL
    if _MODEL is None:
        print("Loading semantic model...")  # visible in server logs
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# ---------------------------------------------------------------------------
# Individual sub-scorers
# ---------------------------------------------------------------------------

def compute_semantic_score(jd_text: str, resume_text: str) -> float:
    """
    Compute semantic similarity between a job description and a resume using
    dense sentence embeddings.

    Both texts are encoded with the cached SentenceTransformer model and their
    cosine similarity is returned.

    Parameters
    ----------
    jd_text : str
        Cleaned job-description text.
    resume_text : str
        Cleaned resume text.

    Returns
    -------
    float
        Cosine similarity in [0.0, 1.0], rounded to 4 decimal places.
        Returns 0.0 if either input is empty.
    """
    if not jd_text or not jd_text.strip():
        return 0.0
    if not resume_text or not resume_text.strip():
        return 0.0

    model = get_model()

    # encode returns a 2-D array of shape (1, embedding_dim) when given a list
    jd_vec = model.encode([jd_text])
    resume_vec = model.encode([resume_text])

    # cosine_similarity returns a 2-D matrix; take the single element
    score = float(cosine_similarity(jd_vec, resume_vec)[0][0])

    # Clip to [0, 1] in case of floating-point overshoot
    score = max(0.0, min(1.0, score))
    return round(score, 4)


def compute_tfidf_score(jd_text: str, resume_text: str) -> float:
    """
    Compute TF-IDF cosine similarity between a job description and a resume.

    Used as a lightweight baseline / comparison signal alongside the semantic
    score.

    Parameters
    ----------
    jd_text : str
        Cleaned job-description text.
    resume_text : str
        Cleaned resume text.

    Returns
    -------
    float
        Cosine similarity in [0.0, 1.0], rounded to 4 decimal places.
        Returns 0.0 if either input is empty.
    """
    if not jd_text or not jd_text.strip():
        return 0.0
    if not resume_text or not resume_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(max_features=5000)

    # Fit on both documents together so the vocabulary is shared
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])

    jd_vec = tfidf_matrix[0]
    resume_vec = tfidf_matrix[1]

    score = float(cosine_similarity(jd_vec, resume_vec)[0][0])
    score = max(0.0, min(1.0, score))
    return round(score, 4)


def _extract_years(text: str) -> Optional[int]:
    """
    Search *text* for any mention of years of experience and return the
    maximum number found, or None if no match is found.

    Patterns matched (case-insensitive):
      - "5 years"
      - "5+ years"
      - "5 yrs"
      - "5+ yrs"

    Parameters
    ----------
    text : str
        Lowercased free-form text.

    Returns
    -------
    int or None
        Maximum years mentioned, or None.
    """
    patterns = [
        r"(\d+)\+?\s*years?",
        r"(\d+)\+?\s*yrs?",
    ]
    years_found: list[int] = []
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            try:
                years_found.append(int(match.group(1)))
            except ValueError:
                pass
    return max(years_found) if years_found else None


def _extract_required_years(jd_text: str) -> Optional[int]:
    """
    Search a job description for an explicit experience requirement and
    return the number of years required, or None if not found.

    Patterns matched (case-insensitive):
      - "4+ years of experience"
      - "minimum 4 years"
      - "at least 4 years"
      - Generic year mentions (fallback)

    Parameters
    ----------
    jd_text : str
        Lowercased job-description text.

    Returns
    -------
    int or None
        Required years, or None.
    """
    specific_patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
        r"minimum\s*(\d+)\s*years?",
        r"at\s*least\s*(\d+)\s*years?",
    ]
    years_found: list[int] = []
    for pat in specific_patterns:
        for match in re.finditer(pat, jd_text, re.IGNORECASE):
            try:
                years_found.append(int(match.group(1)))
            except ValueError:
                pass
    return max(years_found) if years_found else None


def compute_experience_score(resume_text: str, jd_text: str) -> float:
    """
    Estimate how well the candidate's experience level matches the JD
    requirement.

    Scoring rubric
    --------------
    - Required years not found in JD → 0.75 (neutral / no penalty)
    - Candidate years not found in resume → 0.50 (unknown)
    - Candidate years >= required → 1.00
    - Candidate years >= required - 1 → 0.75
    - Candidate years >= required - 2 → 0.50
    - Candidate years  < required - 2 → 0.25

    Parameters
    ----------
    resume_text : str
        Cleaned resume text.
    jd_text : str
        Cleaned job-description text.

    Returns
    -------
    float
        Experience alignment score in [0.0, 1.0].
    """
    required = _extract_required_years(jd_text.lower())

    if required is None:
        # JD does not specify a requirement — no penalty
        return 0.75

    candidate = _extract_years(resume_text.lower())

    if candidate is None:
        # Cannot determine experience from resume — return neutral-low score
        return 0.50

    if candidate >= required:
        return 1.00
    elif candidate >= required - 1:
        return 0.75
    elif candidate >= required - 2:
        return 0.50
    else:
        return 0.25


def _degree_level(text: str) -> int:
    """
    Return a numeric degree level found in *text*.

      4 = PhD / Doctorate
      3 = Master's / MBA / M.Sc / M.Eng
      2 = Bachelor's / B.Sc / B.Eng / Undergraduate
      1 = Associate / Diploma / HND
      0 = No recognisable degree found

    The highest level found is returned (a resume may mention both Bachelor's
    and Master's).

    Parameters
    ----------
    text : str
        Lowercased text to search.

    Returns
    -------
    int
        Highest degree level in {0, 1, 2, 3, 4}.
    """
    phd_keywords = ["phd", "ph.d", "doctorate", "doctoral"]
    masters_keywords = ["master", "msc", "m.sc", "mba", "m.eng", "ms "]
    bachelors_keywords = [
        "bachelor", "bsc", "b.sc", "b.eng", "be ", "b.e", "undergraduate"
    ]
    associate_keywords = ["associate", "diploma", "hnd"]

    level = 0
    for kw in phd_keywords:
        if kw in text:
            level = max(level, 4)
    for kw in masters_keywords:
        if kw in text:
            level = max(level, 3)
    for kw in bachelors_keywords:
        if kw in text:
            level = max(level, 2)
    for kw in associate_keywords:
        if kw in text:
            level = max(level, 1)
    return level


def compute_education_score(resume_text: str, jd_text: str) -> float:
    """
    Estimate how well the candidate's education level meets the JD requirement.

    Scoring rubric
    --------------
    - JD does not mention a degree requirement → 0.80 (don't penalise)
    - Resume degree level >= JD required level → 1.00
    - Resume degree level = JD required - 1    → 0.65
    - Resume degree level <= JD required - 2   → 0.35

    Parameters
    ----------
    resume_text : str
        Cleaned resume text.
    jd_text : str
        Cleaned job-description text.

    Returns
    -------
    float
        Education relevance score in [0.0, 1.0].
    """
    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()

    required_level = _degree_level(jd_lower)

    if required_level == 0:
        # JD has no education requirement — neutral score
        return 0.80

    candidate_level = _degree_level(resume_lower)

    if candidate_level >= required_level:
        return 1.00
    elif candidate_level == required_level - 1:
        return 0.65
    else:
        return 0.35


# ---------------------------------------------------------------------------
# Hybrid scorer — combines all sub-scores
# ---------------------------------------------------------------------------

def compute_hybrid_score(
    jd_data: dict,
    resume_data: dict,
    skill_match: dict,
) -> dict:
    """
    Compute the final weighted hybrid score for one resume against a JD.

    Weights
    -------
    Semantic similarity  : 50 %
    Skill match          : 30 %
    Experience alignment : 15 %
    Education relevance  :  5 %

    Parameters
    ----------
    jd_data : dict
        Output of ``normalize_jd()``.  Uses ``jd_data["cleaned"]``.
    resume_data : dict
        Output of ``extract_text()``.  Uses ``resume_data["cleaned_text"]``.
    skill_match : dict
        Output of ``match_skills()``.  Uses ``skill_match["skill_score"]``.

    Returns
    -------
    dict with keys:
        final_score      : float — weighted composite score in [0.0, 1.0]
        final_score_pct  : float — final_score * 100, rounded to 1 decimal
        semantic_score   : float
        tfidf_score      : float
        skill_score      : float
        experience_score : float
        education_score  : float
        weights          : dict  — the weight mapping used
        recommendation   : str   — "Strong fit" / "Good fit" / "Partial fit" / "Weak fit"
    """
    jd_text = jd_data.get("cleaned", "")
    resume_text = resume_data.get("cleaned_text", "")

    # Compute individual sub-scores
    semantic = compute_semantic_score(jd_text, resume_text)
    tfidf = compute_tfidf_score(jd_text, resume_text)
    skill = float(skill_match.get("skill_score", 0.0))
    experience = compute_experience_score(resume_text, jd_text)
    education = compute_education_score(resume_text, jd_text)

    # Weighted combination
    final = (
        semantic    * WEIGHTS["semantic"]
        + skill     * WEIGHTS["skills"]
        + experience * WEIGHTS["experience"]
        + education * WEIGHTS["education"]
    )
    final = round(max(0.0, min(1.0, final)), 4)

    # Human-readable recommendation
    if final >= 0.75:
        recommendation = "Strong fit"
    elif final >= 0.55:
        recommendation = "Good fit"
    elif final >= 0.35:
        recommendation = "Partial fit"
    else:
        recommendation = "Weak fit"

    return {
        "final_score": final,
        "final_score_pct": round(final * 100, 1),
        "semantic_score": semantic,
        "tfidf_score": tfidf,
        "skill_score": skill,
        "experience_score": experience,
        "education_score": education,
        "weights": dict(WEIGHTS),
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.parser import normalize_jd, extract_text
    from src.skill_extractor import extract_skills, match_skills

    jd_raw = (
        "We need a Senior Data Scientist with 4+ years of experience in Python, "
        "machine learning, scikit-learn, pandas, and SQL. "
        "Master's degree preferred, Bachelor's required."
    )
    resume_raw = (
        "Data Scientist with 5 years of experience. "
        "Proficient in Python, pandas, scikit-learn, TensorFlow, AWS. "
        "M.Sc. in Computer Science from MIT."
    )

    jd_data = normalize_jd(jd_raw)
    # Simulate extract_text output
    resume_data = {
        "filename": "test_resume.pdf",
        "raw_text": resume_raw,
        "cleaned_text": resume_raw.lower(),
        "sections": {},
        "error": None,
    }

    jd_skills = extract_skills(jd_data["cleaned"])
    resume_skills = extract_skills(resume_data["cleaned_text"])
    skill_match = match_skills(resume_skills, jd_skills)

    result = compute_hybrid_score(jd_data, resume_data, skill_match)

    print("Hybrid scoring result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
