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
import datetime
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

    Handles many real-world resume formats including:
      - "5 years of experience", "5+ years of professional experience"
      - "over 5 years", "more than 5 years", "5 + years"
      - "5 years in the industry / field / data"
      - "since 2018"  →  current_year - 2018
      - "2019 - present"  →  current_year - 2019

    Parameters
    ----------
    text : str
        Free-form text (will be lowercased internally).

    Returns
    -------
    int or None
        Maximum years value found (filtered to > 0), or None.
    """
    current_year = datetime.datetime.now().year
    text = text.lower()
    years_found: list[int] = []

    # Patterns where group(1) is directly the integer years value
    direct_patterns = [
        r'(\d+)\+?\s*years?\s*of\s*(?:professional\s*)?(?:experience|exp)',
        r'(\d+)\+?\s*yrs?\s*of\s*(?:professional\s*)?(?:experience|exp)',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in\s*(?:the\s*)?(?:industry|field|domain|software|data)',
        r'over\s*(\d+)\s*years?',
        r'more\s*than\s*(\d+)\s*years?',
        r'(\d+)\s*\+\s*years?',
    ]
    for pat in direct_patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            try:
                val = int(match.group(1))
                if val > 0:
                    years_found.append(val)
            except (ValueError, IndexError):
                pass

    # "since YEAR" → compute elapsed years
    for match in re.finditer(r'since\s*(19|20)(\d{2})', text, re.IGNORECASE):
        try:
            year = int(match.group(1) + match.group(2))
            computed = current_year - year
            if computed > 0:
                years_found.append(computed)
        except (ValueError, IndexError):
            pass

    # "YEAR - present/current/now/today/ongoing" → compute elapsed years
    for match in re.finditer(
        r'(19|20)(\d{2})\s*[-–]\s*(?:present|current|now|today|ongoing)',
        text,
        re.IGNORECASE,
    ):
        try:
            year = int(match.group(1) + match.group(2))
            computed = current_year - year
            if computed > 0:
                years_found.append(computed)
        except (ValueError, IndexError):
            pass

    # "Mon YYYY - Mon YYYY" or "Mon YYYY to Mon YYYY" → sum all durations
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }
    current_month = datetime.datetime.now().month

    month_year_re = (
        r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?'
        r'\s*\d{4})'
        r'\s*(?:[-–—]|to)\s*'
        r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?'
        r'\s*\d{4}|present|current|now|today|ongoing)'
    )
    total_months = 0
    for match in re.finditer(month_year_re, text, re.IGNORECASE):
        try:
            start_str = match.group(1).strip()
            end_str = match.group(2).strip().lower()

            s_tokens = re.split(r'\s+', start_str)
            s_mon = month_map.get(s_tokens[0][:3].lower(), 1)
            s_yr = int(s_tokens[-1])

            if end_str in ('present', 'current', 'now', 'today', 'ongoing'):
                e_mon, e_yr = current_month, current_year
            else:
                e_tokens = re.split(r'\s+', end_str)
                e_mon = month_map.get(e_tokens[0][:3].lower(), 1)
                e_yr = int(e_tokens[-1])

            months = (e_yr - s_yr) * 12 + (e_mon - s_mon)
            if 0 < months < 600:
                total_months += months
        except (ValueError, IndexError):
            pass

    if total_months > 0:
        # Do NOT floor to 1 — return 0 for sub-year so scoring reflects reality
        years_found.append(int(total_months / 12))

    # "YYYY - YYYY" year-only ranges (fallback when no month info is given)
    year_range_re = r'\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})\b'
    year_total_months = 0
    for match in re.finditer(year_range_re, text, re.IGNORECASE):
        try:
            s_yr = int(match.group(1))
            e_yr = int(match.group(2))
            months = (e_yr - s_yr) * 12
            if 0 < months < 600:
                year_total_months += months
        except ValueError:
            pass

    if year_total_months > 0 and total_months == 0:
        years_found.append(int(year_total_months / 12))

    # Filter out negatives; keep 0 (valid: sub-year experience was found)
    years_found = [y for y in years_found if y >= 0]
    return max(years_found) if years_found else None


def _extract_total_experience_months(text: str) -> int:
    """
    Return the total months of experience found in *text* by summing all
    detected date ranges.  Used only for UI display (e.g. "4 months").

    Returns 0 if nothing is detected.
    """
    text = text.lower()
    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }

    month_year_re = (
        r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?'
        r'\s*\d{4})'
        r'\s*(?:[-–—]|to)\s*'
        r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?'
        r'\s*\d{4}|present|current|now|today|ongoing)'
    )
    total = 0
    for match in re.finditer(month_year_re, text, re.IGNORECASE):
        try:
            start_str = match.group(1).strip()
            end_str = match.group(2).strip().lower()
            s_tokens = re.split(r'\s+', start_str)
            s_mon = month_map.get(s_tokens[0][:3].lower(), 1)
            s_yr = int(s_tokens[-1])
            if end_str in ('present', 'current', 'now', 'today', 'ongoing'):
                e_mon, e_yr = current_month, current_year
            else:
                e_tokens = re.split(r'\s+', end_str)
                e_mon = month_map.get(e_tokens[0][:3].lower(), 1)
                e_yr = int(e_tokens[-1])
            months = (e_yr - s_yr) * 12 + (e_mon - s_mon)
            if 0 < months < 600:
                total += months
        except (ValueError, IndexError):
            pass
    return total


def _extract_required_years(jd_text: str) -> Optional[int]:
    """
    Search a job description for an explicit experience requirement and
    return the number of years required, or None if not found.

    Patterns matched (case-insensitive):
      - "4+ years of experience / relevant experience"
      - "minimum of 4 years"
      - "at least 4 years"
      - "3 to 5 years of experience"
      - "4+ years"  (general suffix match)

    Parameters
    ----------
    jd_text : str
        Job-description text (any case; lowercased internally).

    Returns
    -------
    int or None
        Maximum required years found (filtered to > 0), or None.
    """
    jd_text = jd_text.lower()
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*(?:relevant\s*)?(?:experience|exp)',
        r'minimum\s*(?:of\s*)?(\d+)\s*years?',
        r'at\s*least\s*(\d+)\s*years?',
        r'(\d+)\s*to\s*\d+\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\+\s*years?',
    ]
    years_found: list[int] = []
    for pat in patterns:
        for match in re.finditer(pat, jd_text, re.IGNORECASE):
            try:
                val = int(match.group(1))
                if val > 0:
                    years_found.append(val)
            except (ValueError, IndexError):
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

    # Raw text preserves date separators like "Apr 2019 - Dec 2020".
    # clean_text() strips standalone dashes, which breaks date-range detection,
    # so experience extraction must run on raw text.  Fall back to cleaned if absent.
    resume_raw = resume_data.get("raw_text", "") or resume_text

    # Compute individual sub-scores
    semantic = compute_semantic_score(jd_text, resume_text)
    tfidf = compute_tfidf_score(jd_text, resume_text)
    skill = float(skill_match.get("skill_score", 0.0))
    experience = compute_experience_score(resume_raw, jd_text)   # raw text for dates
    education = compute_education_score(resume_text, jd_text)

    # Detected years / months for UI display — raw text preserves date separators
    detected_resume_years = _extract_years(resume_raw.lower())
    detected_resume_months = _extract_total_experience_months(resume_raw.lower())
    detected_jd_years = _extract_required_years(jd_text.lower())

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
        "detected_resume_years": detected_resume_years,
        "detected_resume_months": detected_resume_months,
        "detected_jd_years": detected_jd_years,
    }


# ---------------------------------------------------------------------------
# Plain-language explanation generator
# ---------------------------------------------------------------------------

def generate_explanation(scores: dict, skill_match: dict) -> str:
    """
    Generate a specific, actionable plain-English explanation for a recruiter.

    Builds up to 4 targeted bullets covering: overall verdict, skill gap with
    named missing skills, experience vs requirement, and education fit.
    Each bullet is only included when it adds real information.

    Parameters
    ----------
    scores : dict
        The dict returned by ``compute_hybrid_score()``.
    skill_match : dict
        The dict returned by ``match_skills()``.

    Returns
    -------
    str
        HTML string with 2-4 concise bullet points ready for st.markdown.
    """
    semantic  = scores["semantic_score"]
    skill     = scores["skill_score"]
    exp       = scores["experience_score"]
    edu       = scores["education_score"]
    final     = scores["final_score"]

    matched_list  = skill_match.get("matched", [])
    missing_list  = skill_match.get("missing", [])
    matched       = len(matched_list)
    total_jd      = matched + len(missing_list)

    resume_years  = scores.get("detected_resume_years")
    resume_months = scores.get("detected_resume_months", 0)
    jd_years      = scores.get("detected_jd_years")

    bullets: list[str] = []

    # ── Bullet 1: Overall verdict with semantic context ──────────────────
    if final >= 0.75:
        verdict = "Strong overall match"
    elif final >= 0.55:
        verdict = "Good overall match"
    elif final >= 0.35:
        verdict = "Partial match"
    else:
        verdict = "Weak match"

    if semantic >= 0.65:
        context = "resume language closely mirrors the job description"
    elif semantic >= 0.45:
        context = "moderate semantic overlap with the job description"
    else:
        context = "resume language differs significantly from the job description"

    bullets.append(f"<b>{verdict}</b> — {context} ({semantic:.0%} similarity).")

    # ── Bullet 2: Skill gap — always show named missing skills ───────────
    if total_jd == 0:
        bullets.append("No required skills were detected in the job description.")
    elif skill >= 0.75:
        bullets.append(
            f"<b>Strong skill coverage</b> — {matched} of {total_jd} required "
            f"skills matched ({skill:.0%})."
        )
    else:
        # Name up to 5 missing skills explicitly so recruiter knows exactly what's absent
        top_missing = missing_list[:5]
        more = len(missing_list) - len(top_missing)
        missing_str = ", ".join(f"<i>{s}</i>" for s in top_missing)
        if more > 0:
            missing_str += f" (+{more} more)"
        bullets.append(
            f"<b>Skill gap</b> — {matched}/{total_jd} required skills matched. "
            f"Missing: {missing_str}."
        )

    # ── Bullet 3: Experience — be specific about years detected ─────────
    if jd_years is None:
        # JD has no explicit requirement — skip experience bullet
        pass
    elif resume_years is None and resume_months == 0:
        bullets.append(
            "<b>Experience</b> — could not detect years from resume; "
            "verify manually."
        )
    elif resume_years == 0 and resume_months > 0:
        bullets.append(
            f"<b>Experience</b> — only {resume_months} months detected; "
            f"role requires {jd_years}+ years."
        )
    elif resume_years is not None and resume_years >= jd_years:
        bullets.append(
            f"<b>Experience</b> — {resume_years} yrs detected meets the "
            f"{jd_years}+ yr requirement. ✓"
        )
    elif resume_years is not None:
        gap = jd_years - resume_years
        bullets.append(
            f"<b>Experience</b> — {resume_years} yrs detected vs "
            f"{jd_years}+ yrs required ({gap} yr gap)."
        )

    # ── Bullet 4: Education — only when it meaningfully affects the score ─
    if edu < 0.7:
        bullets.append(
            "<b>Education</b> — degree level appears below the role's "
            "stated requirement."
        )
    elif edu >= 1.0:
        bullets.append("<b>Education</b> — meets or exceeds the degree requirement. ✓")

    # Render as a compact HTML list
    items = "".join(f"<li style='margin:3px 0;'>{b}</li>" for b in bullets)
    return (
        f"<ul style='margin:6px 0 0 0; padding-left:18px; "
        f"font-size:13px; line-height:1.7;'>{items}</ul>"
    )


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
