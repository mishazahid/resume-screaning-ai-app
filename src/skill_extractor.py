"""
skill_extractor.py
==================
Detects technical and soft skills in free-form text using a curated
dictionary with canonical names and aliases.

The matching strategy uses whole-word regex (\\b boundaries) so short tokens
like "r" or "go" are not spuriously matched inside longer words.

Public API
----------
extract_skills(text)         -> list[str]
match_skills(resume, jd)     -> dict
"""

import re
from typing import Union

# ---------------------------------------------------------------------------
# Skill dictionary
# Key   : canonical skill name (how it will appear in the UI and output)
# Value : list of lowercase aliases to search for in text
# ---------------------------------------------------------------------------
SKILLS_DICT: dict[str, list[str]] = {
    # ── Programming Languages ──────────────────────────────────────────────
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "c++": ["c\\+\\+", "cpp", "c plus plus"],
    "c#": ["c#", "c sharp", "csharp"],
    "r": ["\\br\\b"],            # word-boundary pattern stored here; handled specially
    "go": ["\\bgo\\b", "golang"],
    "scala": ["scala"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "ruby": ["ruby"],
    "php": ["php"],
    "bash": ["bash", "shell scripting", "shell script"],
    "matlab": ["matlab"],
    # ── ML & AI ───────────────────────────────────────────────────────────
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing", "natural-language processing"],
    "computer vision": ["computer vision", "cv"],
    "reinforcement learning": ["reinforcement learning", "rl"],
    "transformers": ["transformers", "transformer model"],
    "llm": ["llm", "large language model", "large language models"],
    "generative ai": ["generative ai", "genai", "gen ai", "generative artificial intelligence"],
    "feature engineering": ["feature engineering", "feature extraction"],
    "model deployment": ["model deployment", "model serving", "ml deployment", "mlops"],
    "a/b testing": ["a/b testing", "ab testing", "a b testing", "split testing"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling"],
    "time series": ["time series", "time-series", "forecasting"],
    "anomaly detection": ["anomaly detection", "outlier detection"],
    # ── Frameworks & Libraries ────────────────────────────────────────────
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "keras": ["keras"],
    "hugging face": ["hugging face", "huggingface"],
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgbm"],
    "fastapi": ["fastapi", "fast api"],
    "flask": ["flask"],
    "django": ["django"],
    "spring boot": ["spring boot", "springboot"],
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "nodejs", "node js"],
    "opencv": ["opencv", "open cv"],
    # ── Data Tools ────────────────────────────────────────────────────────
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "spark": ["spark", "apache spark", "pyspark"],
    "hadoop": ["hadoop", "hdfs", "mapreduce"],
    "kafka": ["kafka", "apache kafka"],
    "airflow": ["airflow", "apache airflow"],
    "dbt": ["\\bdbt\\b", "data build tool"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "microsoft excel", "ms excel"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],
    # ── Databases ─────────────────────────────────────────────────────────
    "postgresql": ["postgresql", "postgres", "psql"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "bigquery": ["bigquery", "big query", "google bigquery"],
    "snowflake": ["snowflake"],
    "sqlite": ["sqlite", "sqlite3"],
    "oracle": ["oracle", "oracle db", "oracle database"],
    "dynamodb": ["dynamodb", "dynamo db", "amazon dynamodb"],
    # ── Cloud & DevOps ────────────────────────────────────────────────────
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud platform", "google cloud"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform"],
    "ci/cd": ["ci/cd", "ci cd", "continuous integration", "continuous deployment", "jenkins", "github actions"],
    "git": ["git"],
    "github": ["github"],
    "linux": ["linux", "unix"],
    "rest api": ["rest api", "restful api", "rest apis", "restful", "rest"],
    "graphql": ["graphql"],
    # ── Soft Skills ───────────────────────────────────────────────────────
    "communication": ["communication", "written communication", "verbal communication"],
    "leadership": ["leadership", "team lead", "tech lead"],
    "teamwork": ["teamwork", "collaboration", "team player"],
    "problem solving": ["problem solving", "problem-solving", "analytical thinking"],
    "agile": ["agile", "agile methodology"],
    "scrum": ["scrum"],
    "project management": ["project management", "pm"],
    "mentoring": ["mentoring", "mentorship", "coaching"],
}


def _build_pattern(aliases: list[str]) -> re.Pattern:
    """
    Build a compiled regex pattern that matches any of the given aliases as
    whole words (using \\b word boundaries).

    Aliases that already contain \\b are used verbatim; all others are wrapped
    with \\b..\\b so that short tokens like "r" or "go" do not match inside
    longer words.

    Parameters
    ----------
    aliases : list[str]
        Lowercase alias strings.

    Returns
    -------
    re.Pattern
        Compiled pattern with IGNORECASE flag.
    """
    parts: list[str] = []
    for alias in aliases:
        if "\\b" in alias:
            # Already contains boundary markers — use as-is
            parts.append(alias)
        else:
            # Escape regex metacharacters in the alias EXCEPT for forward slashes
            # (we want "a/b testing" to match literally)
            escaped = re.escape(alias)
            parts.append(r"\b" + escaped + r"\b")
    combined = "|".join(parts)
    return re.compile(combined, re.IGNORECASE)


# Pre-compile all patterns at import time for performance
_COMPILED_PATTERNS: dict[str, re.Pattern] = {
    skill: _build_pattern(aliases) for skill, aliases in SKILLS_DICT.items()
}


def extract_skills(text: str) -> list[str]:
    """
    Identify which canonical skills appear in *text*.

    Matching is performed using pre-compiled whole-word regex patterns so that
    short skill tokens (e.g. "r", "go") are not matched spuriously inside
    longer words.

    Parameters
    ----------
    text : str
        Any free-form text (resume body, job description, etc.).

    Returns
    -------
    list[str]
        Sorted list of canonical skill names found in the text.
        Empty list if no skills are detected or text is empty.
    """
    if not text:
        return []

    lowered = text.lower()
    found: list[str] = []

    for skill, pattern in _COMPILED_PATTERNS.items():
        if pattern.search(lowered):
            found.append(skill)

    return sorted(found)


def match_skills(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compare the skills extracted from a resume against those from a job
    description and compute a skill-match score.

    Parameters
    ----------
    resume_skills : list[str]
        Canonical skill names found in the resume.
    jd_skills : list[str]
        Canonical skill names found in the job description.

    Returns
    -------
    dict with keys:
        matched     : list[str]  — skills present in both resume and JD
        missing     : list[str]  — skills in JD but absent from resume
        extra       : list[str]  — skills in resume but not required by JD
        match_ratio : float      — len(matched) / len(jd_skills); 0.0 if JD empty
        skill_score : float      — same as match_ratio (kept separate for clarity)
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    extra = sorted(resume_set - jd_set)

    if jd_set:
        match_ratio = round(len(matched) / len(jd_set), 4)
    else:
        match_ratio = 0.0

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "match_ratio": match_ratio,
        "skill_score": match_ratio,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_resume = """
    Experienced Data Scientist with 5 years of experience in Python, machine learning,
    and deep learning. Proficient in scikit-learn, pandas, numpy, TensorFlow, and
    PyTorch. Deployed models on AWS using Docker and Kubernetes. Used Airflow for
    pipeline orchestration. Strong communication and teamwork skills.
    """

    sample_jd = """
    We are looking for a Senior Data Scientist with expertise in Python, SQL,
    machine learning, scikit-learn, pandas, statistics. Nice to have: PyTorch,
    Spark, AWS, NLP. 4+ years of experience. Bachelor's degree required.
    """

    resume_skills = extract_skills(sample_resume)
    jd_skills = extract_skills(sample_jd)

    print("Resume skills:", resume_skills)
    print("\nJD skills:", jd_skills)

    match = match_skills(resume_skills, jd_skills)
    print("\nSkill match result:")
    for k, v in match.items():
        print(f"  {k}: {v}")
