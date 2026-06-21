"""
tenure_stability_score.py
Penalizes a PATTERN of company-switching every <18 months while titles climb
(Senior -> Staff -> Principal). A single short stint is NOT penalized -- per
the rubric, only a repeated pattern (3+) counts, and only if titles are
visibly climbing through it (matches the JD's exact complaint about
title-chasers, not just people who changed jobs a few times for normal reasons).
"""

import argparse
import pandas as pd

# (keywords, seniority level) -- checked against title text, highest match wins
SENIORITY_LEVELS = [
    (["intern", "trainee"], 0),
    (["junior", "jr."], 1),
    (["associate"], 2),
    (["senior", "sr."], 4),
    (["staff"], 5),
    (["principal"], 6),
    (["lead"], 5),
    (["head of", "head,"], 7),
    (["director"], 8),
    (["vp", "vice president"], 9),
    (["chief", "cto", "ceo", "cxo"], 10),
]


def seniority_level(title: str) -> int:
    t = (title or "").lower()
    levels = [3]  # default mid-level baseline if nothing matches
    for keywords, level in SENIORITY_LEVELS:
        if any(k in t for k in keywords):
            levels.append(level)
    return max(levels)


def tenure_score_one(career_history: list) -> float:
    if len(career_history) < 3:
        return 1.0  # not enough jobs to show a hopping pattern at all

    sorted_jobs = sorted(career_history, key=lambda j: j.get("start_date") or "")
    short_stints = [j for j in sorted_jobs if (j.get("duration_months") or 999) < 18]

    if len(short_stints) < 3:
        return 1.0  # a short stint or two is normal, not penalized

    levels = [seniority_level(j.get("title", "")) for j in sorted_jobs]
    is_climbing = (
        all(levels[i] <= levels[i + 1] for i in range(len(levels) - 1))
        and levels[-1] > levels[0]
    )

    if is_climbing:
        return 0.4  # matches the JD's exact complaint: hopping for title bumps
    return 0.7  # frequent short stints, but not a clear title-chasing pattern


def score_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df["tenure_stability_score"] = df["career_history"].apply(tenure_score_one)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="candidates_with_experience_score.pkl")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_pickle(args.checkpoint)

    df = score_tenure(df)

    print("\ntenure_stability_score distribution:")
    print(df["tenure_stability_score"].value_counts().sort_index())

    print("\nCandidates flagged as title-chasers (score 0.4):")
    chasers = df[df["tenure_stability_score"] == 0.4]
    print(f"Count: {len(chasers)}")
    print(chasers[["candidate_id", "current_title", "num_jobs"]].head(10).to_string(index=False))

    df.to_pickle("candidates_with_tenure_score.pkl")
    print("\nSaved checkpoint to candidates_with_tenure_score.pkl")
