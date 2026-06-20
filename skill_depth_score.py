"""
skill_depth_score.py
Scores each candidate's skill-depth fit against the JD's core skill list.
Weighted by proficiency level and duration_months, NOT just keyword presence --
this is what stops keyword-stuffers (skill listed, zero real depth) from
beating genuine practitioners (skill listed, years of real depth behind it).
"""

import argparse
import pandas as pd
import numpy as np


def normalize(s: str) -> str:
    return s.lower().strip().replace("-", " ").replace("_", " ")


# Core JD skills, pulled directly from job_description.docx (see RUBRIC_v1.md).
# High-trust = specific technologies the JD names explicitly.
# Generic = broader terms a keyword-stuffer is more likely to list without depth.
HIGH_TRUST_SKILLS = {normalize(x) for x in [
    "sentence-transformers", "OpenAI embeddings", "BGE", "E5",
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
    "Elasticsearch", "FAISS",
    "NDCG", "MRR", "MAP", "A/B testing",
    "LoRA", "QLoRA", "PEFT", "fine-tuning",
    "XGBoost", "LightGBM", "learning to rank",
]}

GENERIC_SKILLS = {normalize(x) for x in [
    "embeddings", "retrieval", "ranking", "LLM", "RAG", "hybrid search",
    "machine learning", "deep learning", "NLP",
]}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}

# Raw skill-value sum is capped here before normalizing to 0-1.
# Roughly: 4 strong core skills (expert, 2+ yrs each) maxes the score.
NORMALIZATION_CEILING = 4.0


def match_skill(skill_name: str) -> tuple:
    """Returns (matched: bool, trust_weight: float) for one skill name."""
    name = normalize(skill_name)
    if name in HIGH_TRUST_SKILLS:
        return True, 1.0
    if name in GENERIC_SKILLS:
        return True, 0.5
    return False, 0.0


def score_one_candidate(skills: list) -> float:
    """Raw (uncapped) skill-depth value for one candidate's skill list."""
    total = 0.0
    for s in skills:
        matched, trust_weight = match_skill(s.get("name", ""))
        if not matched:
            continue
        prof = PROFICIENCY_WEIGHT.get(s.get("proficiency", "").lower(), 0.0)
        duration = s.get("duration_months", 0) or 0
        depth_factor = min(duration / 24, 1.0)
        total += prof * depth_factor * trust_weight
    return total


def score_skill_depth(df: pd.DataFrame) -> pd.DataFrame:
    raw_scores = df["skills"].apply(score_one_candidate)
    df["skill_depth_score"] = np.clip(raw_scores / NORMALIZATION_CEILING, 0, 1)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="candidates_with_career_score.pkl")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_pickle(args.checkpoint)

    df = score_skill_depth(df)

    print("\nskill_depth_score distribution:")
    print(df["skill_depth_score"].describe())

    print("\nTop 10 by skill_depth_score:")
    print(df.sort_values("skill_depth_score", ascending=False)
            [["candidate_id", "current_title", "skill_depth_score", "career_context_score"]]
            .head(10).to_string(index=False))

    zero_count = (df["skill_depth_score"] == 0).sum()
    print(f"\nCandidates with skill_depth_score == 0: {zero_count:,}")

    df.to_pickle("candidates_with_skill_score.pkl")
    print("\nSaved checkpoint to candidates_with_skill_score.pkl")
