"""
experience_score.py
Scores experience fit against the JD's stated 5-9 year sweet spot.
Soft bell curve, not a hard cutoff -- the JD explicitly treats this as a
range to weigh, not a wall to filter on. Never hits zero; a candidate
outside the band can still be pulled up by strong fit elsewhere.
"""

import argparse
import pandas as pd
import numpy as np


def experience_score_one(yoe: float) -> float:
    """Score one candidate's years_of_experience. Peak 1.0 for 5-9 yrs."""
    if yoe is None:
        return 0.3
    if 5 <= yoe <= 9:
        return 1.0
    if yoe < 5:
        if yoe <= 2:
            return 0.3
        # linear ramp from (2 yrs -> 0.3) up to (5 yrs -> 1.0)
        return 0.3 + (yoe - 2) / (5 - 2) * (1.0 - 0.3)
    else:  # yoe > 9
        if yoe >= 15:
            return 0.3
        # linear taper from (9 yrs -> 1.0) down to (15 yrs -> 0.3)
        return 1.0 - (yoe - 9) / (15 - 9) * (1.0 - 0.3)


def score_experience(df: pd.DataFrame) -> pd.DataFrame:
    df["experience_score"] = df["years_of_experience"].apply(experience_score_one)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="candidates_with_skill_score.pkl")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_pickle(args.checkpoint)

    df = score_experience(df)

    print("\nexperience_score distribution:")
    print(df["experience_score"].describe())

    print("\nSample: years_of_experience vs experience_score (sanity check the curve)")
    sample = df[["years_of_experience", "experience_score"]].drop_duplicates(
        subset="years_of_experience").sort_values("years_of_experience")
    print(sample.to_string(index=False))

    df.to_pickle("candidates_with_experience_score.pkl")
    print("\nSaved checkpoint to candidates_with_experience_score.pkl")
