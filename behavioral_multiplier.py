"""
behavioral_multiplier.py
Applied AFTER the 6-component Fit Score, not blended into it -- this answers
a different question ("is this strong candidate actually reachable?") rather
than "is this a good candidate?" Penalties compound multiplicatively, so a
candidate hitting multiple red flags at once (the "perfect-on-paper ghost"
pattern we found earlier -- 179 real candidates match this) drops hard.
Clamped to [0.3, 1.15] so no single flag fully erases a strong fit, and no
combination of bonuses inflates a weak fit past a genuinely strong one.
"""

import argparse
from datetime import date
import pandas as pd


def behavioral_multiplier_one(
    open_to_work: bool, last_active_date_str: str, response_rate: float, notice_period: int
) -> float:
    mult = 1.0

    if not open_to_work:
        mult *= 0.75

    last_active = date.fromisoformat(last_active_date_str)
    days_inactive = (date.today() - last_active).days
    if days_inactive > 180:
        mult *= 0.65

    if response_rate is not None and response_rate < 0.10:
        mult *= 0.70

    if notice_period is not None:
        if notice_period < 30:
            mult *= 1.10
        elif notice_period > 90:
            mult *= 0.85

    return min(max(mult, 0.3), 1.15)


def score_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    df["behavioral_multiplier"] = df.apply(
        lambda row: behavioral_multiplier_one(
            row["open_to_work_flag"],
            row["last_active_date"],
            row["recruiter_response_rate"],
            row["notice_period_days"],
        ),
        axis=1,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="candidates_with_quality_score.pkl")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_pickle(args.checkpoint)

    df = score_behavioral(df)

    print("\nbehavioral_multiplier distribution:")
    print(df["behavioral_multiplier"].describe())

    print("\nCount at the worst multiplier (~0.46, the 'ghost' pattern):")
    ghosts = df[df["behavioral_multiplier"] <= 0.5]
    print(f"Count: {len(ghosts)}")

    print("\nCount at the best multiplier (1.10+, fast notice + open to work):")
    best = df[df["behavioral_multiplier"] >= 1.10]
    print(f"Count: {len(best)}")

    df.to_pickle("candidates_with_behavioral_multiplier.pkl")
    print("\nSaved checkpoint to candidates_with_behavioral_multiplier.pkl")
