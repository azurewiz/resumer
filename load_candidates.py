"""
load_candidates.py
Streams candidates.jsonl and flattens each record into a row matching
candidate_schema.json. Designed to run on the full 100K-record file without
loading the raw text into memory all at once via line-by-line streaming.
"""

import json
import argparse
import pandas as pd


def flatten_candidate(record: dict) -> dict:
    """Flatten one candidate JSON record into a single flat dict for a DataFrame row."""
    profile = record["profile"]
    signals = record["redrob_signals"]
    salary = signals.get("expected_salary_range_inr_lpa", {})

    return {
        "candidate_id": record["candidate_id"],

        # profile
        "years_of_experience": profile.get("years_of_experience"),
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "current_company_size": profile.get("current_company_size"),
        "current_industry": profile.get("current_industry"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", ""),

        # career history (kept as raw list for now -- scoring functions will
        # process this directly, not flattened further here)
        "career_history": record.get("career_history", []),
        "num_jobs": len(record.get("career_history", [])),

        # education
        "education": record.get("education", []),
        "num_degrees": len(record.get("education", [])),

        # skills
        "skills": record.get("skills", []),
        "num_skills": len(record.get("skills", [])),

        # redrob signals
        "profile_completeness_score": signals.get("profile_completeness_score"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "last_active_date": signals.get("last_active_date"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "notice_period_days": signals.get("notice_period_days"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
        "github_activity_score": signals.get("github_activity_score"),
        "salary_min_lpa": salary.get("min"),
        "salary_max_lpa": salary.get("max"),
        "num_skill_assessments": len(signals.get("skill_assessment_scores", {})),
    }


def load_candidates(path: str) -> pd.DataFrame:
    """Stream-load candidates.jsonl into a flat pandas DataFrame."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(flatten_candidate(record))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    args = parser.parse_args()

    df = load_candidates(args.candidates)
    print(f"Loaded {len(df):,} candidates")
    print(df.head(3))
    print("\nColumns:", list(df.columns))
