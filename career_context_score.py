"""
career_context_score.py
Scores each candidate's semantic fit against the JD using sentence embeddings.
This is the component that catches plain-language Tier-5 fits (real work, no
buzzwords) and resists keyword stuffers (buzzwords, no real work) -- because
it compares MEANING of career-history text against the JD, not keyword overlap.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from load_candidates import load_candidates

# This is a condensed representation of the JD's actual requirements
# (see job_description.docx "How to read between the lines" section).
# Built from the doc's own framing of the ideal candidate, not generic
# AI-engineer boilerplate -- the specificity here matters for embedding quality.
JD_ANCHOR_TEXT = """
Senior AI engineer with 6-8 years total experience, 4-5 years in applied
machine learning or AI roles at product companies, not pure IT services or
consulting. Has shipped at least one end-to-end ranking, search, or
recommendation system to real users at meaningful production scale. Has
hands-on experience with embeddings-based retrieval, vector databases,
evaluation frameworks for ranking quality, and informed opinions on when to
fine-tune versus prompt large language models, backed by systems they
actually built and deployed, not just tutorials or demos.
"""


def embed_career_text(record_career_history: list) -> str:
    """Concatenate all job descriptions for one candidate into one text blob."""
    parts = []
    for job in record_career_history:
        title = job.get("title", "")
        desc = job.get("description", "")
        parts.append(f"{title}. {desc}")
    return " ".join(parts)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector `a` and a matrix of vectors `b`."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm


def score_career_context(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    # Embed the JD once
    jd_embedding = model.encode(JD_ANCHOR_TEXT.strip(), convert_to_numpy=True)

    # Build career-history text blobs for every candidate
    career_texts = df["career_history"].apply(embed_career_text).tolist()

    # Embed all candidates in one batched call (much faster than one-by-one)
    print(f"Embedding {len(career_texts):,} candidate career histories...")
    candidate_embeddings = model.encode(
        career_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    similarities = cosine_sim(jd_embedding, candidate_embeddings)

    # Cosine similarity ranges roughly -1 to 1; clip to 0-1 for a clean score
    df["career_context_score"] = np.clip(similarities, 0, 1)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    args = parser.parse_args()

    print("Loading candidates...")
    df = load_candidates(args.candidates)

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = score_career_context(df, model)

    print("\nTop 10 by career_context_score:")
    print(df.sort_values("career_context_score", ascending=False)
            [["candidate_id", "current_title", "current_company", "career_context_score"]]
            .head(10).to_string(index=False))

    print("\nBottom 5 by career_context_score:")
    print(df.sort_values("career_context_score", ascending=True)
            [["candidate_id", "current_title", "current_company", "career_context_score"]]
            .head(5).to_string(index=False))

    # Cache result so we don't re-embed every time while building other scorers
    df.to_pickle("candidates_with_career_score.pkl")
    print("\nSaved checkpoint to candidates_with_career_score.pkl")
