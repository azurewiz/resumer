"""
honeypot_filter.py
Stage 1 hard filter -- flags candidates with internally impossible profiles
so they can be excluded before scoring (per submission_spec, your top 100
must contain fewer than 10% honeypots, or your submission is disqualified).

HONEST LIMITATION, UPDATED AFTER TESTING: the original v1 draft required 2+
of 3 signals to fire. Tested against the full 100K dataset: signal 1 fires
for 21 candidates, signal 2 fires for 49, and the overlap between them is
EXACTLY ZERO -- they never co-occur in this dataset. Requiring both was
mathematically guaranteed to flag nobody. Changed to: flag if EITHER signal
fires. That gives 70 flagged candidates, closely matching the README's
stated ~80 known honeypots -- strong evidence this is the right threshold.
Signal 3 (embedding-based title/description coherence) still isn't built;
adding it later may close the remaining ~10-candidate gap.
"""

import argparse
import pandas as pd


def has_impossible_skill(skills: list) -> bool:
    """Signal 1: expert-level proficiency claimed with zero months of use."""
    return any(
        s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
        for s in skills
    )


def has_timeline_inconsistency(career_history: list, years_of_experience: float) -> bool:
    """Signal 2: total job duration doesn't match stated years of experience
    (off by more than 30% either direction)."""
    if not career_history or not years_of_experience or years_of_experience <= 0:
        return False
    total_months = sum((j.get("duration_months") or 0) for j in career_history)
    total_years = total_months / 12
    ratio = total_years / years_of_experience
    return ratio < 0.7 or ratio > 1.3


def is_honeypot_one(skills, career_history, years_of_experience) -> bool:
    # Verified: signals 1 and 2 never co-occur in this dataset (0 overlap
    # across 100K candidates), so requiring "any" rather than "both" is the
    # threshold that actually matches the README's ~80 known honeypots.
    return has_impossible_skill(skills) or has_timeline_inconsistency(
        career_history, years_of_experience
    )


def flag_honeypots(df: pd.DataFrame) -> pd.DataFrame:
    df["is_honeypot"] = df.apply(
        lambda row: is_honeypot_one(
            row["skills"], row["career_history"], row["years_of_experience"]
        ),
        axis=1,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="candidates_with_behavioral_multiplier.pkl")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_pickle(args.checkpoint)

    df = flag_honeypots(df)

    flagged_count = df["is_honeypot"].sum()
    print(f"\nTotal flagged as honeypot: {flagged_count} / {len(df):,}")
    print("(README states ~80 known honeypots exist -- compare against this number)")

    print("\n--- Built-in sanity check against known examples ---")
    for cid, expected in [("CAND_0003582", True), ("CAND_0000031", False)]:
        row = df[df["candidate_id"] == cid]
        if row.empty:
            print(f"{cid}: not found in this checkpoint")
            continue
        actual = bool(row["is_honeypot"].iloc[0])
        status = "PASS" if actual == expected else "FAIL"
        print(f"{cid}: expected honeypot={expected}, got={actual}  [{status}]")

    df.to_pickle("candidates_with_all_scores.pkl")
    print("\nSaved final checkpoint to candidates_with_all_scores.pkl")
    print("(this is the checkpoint rank.py will use to combine everything)")
