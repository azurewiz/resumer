"""
quality_tiebreaker_score.py
Small-weight signals (7% of total) used to break ties between otherwise
similar candidates: GitHub activity, education tier, and platform engagement
(skill assessments taken). None of these are decisive alone -- a candidate
with no GitHub linked (-1) is scored neutral, NOT penalized, since plenty of
strong engineers don't have public GitHub activity.
"""

import argparse
import pandas as pd

EDU_TIER_SCORE = {
    "tier_1": 1.0,
    "tier_2": 0.7,
    "tier_3": 0.4,
    "tier_4": 0.2,
    "unknown": 0.5,  # neutral -- unknown isn't evidence of anything
}


def github_component(github_score) -> float:
    if github_score is None or github_score == -1:
        return 0.5  # neutral, not penalized for no GitHub linked
    return min(max(github_score / 100, 0.0), 1.0)


def education_component(education_list: list) -> float:
    if not education_list:
        return 0.5
    tiers = [EDU_TIER_SCORE.get(e.get("tier", "unknown"), 0.5) for e in education_list]
    return max(tiers)  # best degree counts, not average


def assessment_component(num_assessments: int) -> float:
    return min(num_assessments / 3, 1.0)  # credit caps at 3 assessments taken


def quality_score_one(github_score, education_list, num_assessments) -> float:
    g = github_component(github_score)
    e = education_component(education_list)
    a = assessment_component(num_assessments)
    return (g + e + a) / 3


def score_quality(df: pd.DataFrame) -> pd.DataFrame:
    df["quality_score"] = df.apply(
        lambda row: quality_score_one(
            row["github_activity_score"], row["education"], row["num_skill_assessments"]
        ),
        axis=1,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="candidates_with_geo_score.pkl")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_pickle(args.checkpoint)

    df = score_quality(df)

    print("\nquality_score distribution:")
    print(df["quality_score"].describe())

    print("\nTop 10 by quality_score:")
    print(df.sort_values("quality_score", ascending=False)
            [["candidate_id", "current_title", "github_activity_score", "quality_score"]]
            .head(10).to_string(index=False))

    df.to_pickle("candidates_with_quality_score.pkl")
    print("\nSaved checkpoint to candidates_with_quality_score.pkl")
