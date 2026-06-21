"""
geo_eligibility_score.py
Scores location fit against the JD's exact text: "Location: Pune/Noida, India
(Hybrid) | Open to relocation candidates from Tier-1 Indian cities."
Soft scoring, not a hard filter -- the JD allows case-by-case exceptions for
strong candidates elsewhere, so no one gets excluded purely on location.
"""

import argparse
import pandas as pd

PUNE_NOIDA = {"pune", "noida"}
OTHER_TIER_1 = {
    "bangalore", "bengaluru", "mumbai", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "chennai", "kolkata",
}


def geo_score_one(location: str, country: str, willing_to_relocate: bool) -> float:
    loc = (location or "").lower()
    is_india = (country or "").strip().lower() == "india"

    if not is_india:
        return 0.1

    if any(city in loc for city in PUNE_NOIDA):
        return 1.0

    if any(city in loc for city in OTHER_TIER_1):
        return 0.8 if willing_to_relocate else 0.4

    return 0.2  # India, but not a Tier-1 city


def score_geo(df: pd.DataFrame) -> pd.DataFrame:
    df["geo_eligibility_score"] = df.apply(
        lambda row: geo_score_one(
            row["location"], row["country"], row["willing_to_relocate"]
        ),
        axis=1,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="candidates_with_tenure_score.pkl")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_pickle(args.checkpoint)

    df = score_geo(df)

    print("\ngeo_eligibility_score distribution:")
    print(df["geo_eligibility_score"].value_counts().sort_index())

    print("\nSample locations per score tier:")
    for score_val in sorted(df["geo_eligibility_score"].unique()):
        sample = df[df["geo_eligibility_score"] == score_val][["location", "country"]].head(3)
        print(f"\n  score={score_val}:")
        print(sample.to_string(index=False))

    df.to_pickle("candidates_with_geo_score.pkl")
    print("\nSaved checkpoint to candidates_with_geo_score.pkl")
