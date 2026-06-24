# Ranking Rubric v1 — ruby_mage / Redrob AI Candidate Ranker

**Status:** First draft. Thresholds/weights below are starting points, to be
tuned once the pipeline is actually running against real output (see
"Known unknowns" at the bottom). Agile-iterate from here, don't treat as final.

---

## Architecture: two stages

**Stage 1 — Hard filter pass** (binary exclude/keep, runs on all candidates)
**Stage 2 — Composite scoring** (on the surviving pool) = Fit Score × Behavioral Multiplier

---

## Stage 1: Hard filters (honeypot / trap exclusion)

**UPDATED after testing against full dataset:** originally specified "2+ of
3 signals." Tested signals 1 and 2 against all 100K candidates — zero
overlap between them (21 candidates trip signal 1, 49 trip signal 2, none
trip both). Requiring both was mathematically guaranteed to flag nobody.
Changed to: candidate excluded if **either** signal fires.

1. **Impossible skill claim** — any skill with `proficiency == "expert"` AND
   `duration_months == 0`
2. **Timeline inconsistency** — total `career_history` duration_months wildly
   mismatched against `years_of_experience` (off by more than ~30%)

Result: 70 candidates flagged, closely matching the README's stated ~80.
Signal 3 (embedding-based title/description coherence) still not built —
may close the remaining ~10-candidate gap later, not currently blocking.

**Validation step (mandatory before submission):** rank everyone, check what
fraction of your top 100 trips 2+ of these flags. Target: 0%. Hard ceiling: <10%.

---

## Stage 2: Fit Score (6 weighted components, sum to 100)

### 1. Career-context fit — 35%
Embedding similarity (sentence-transformers, `all-MiniLM-L6-v2`) between:
- JD's core requirement text (the "what we mean by this role" section)
- Candidate's concatenated `career_history[].description` text

This is the component that catches plain-language Tier 5s (real work,
no buzzwords) and rejects keyword stuffers (buzzwords, no real work) —
it's the single most load-bearing piece of the whole system. Cosine
similarity, normalized 0-1, scaled to this weight.

### 2. Skill-depth fit — 25%
**Verified on full 100K dataset:** 78,884 candidates score 0 (expected for a
narrow profile in a broad pool). Of 204 candidates scoring ≥0.7, 96% are
genuinely AI/ML-titled. 8 (4%) have unrelated titles despite high skill
scores — this component alone doesn't fully resist stuffing, which is why
it's weighted below career-context and cross-validated against it, not
trusted in isolation.

NOT binary keyword presence. For each skill on a curated "core JD skills"
list (sentence-transformers, FAISS, vector DBs, NLP, embeddings, Python,
retrieval, ranking, LLMs, evaluation/NDCG, fine-tuning, LoRA/QLoRA):

```
skill_value = proficiency_weight(beginner=0.25, intermediate=0.5, advanced=0.75, expert=1.0)
              × min(duration_months / 24, 1.0)   # caps credit at 2 years depth
```
Sum across matched skills, normalize, scale to weight.

### 3. Experience-range fit — 15%
Soft bell curve, not a hard cutoff (JD explicitly says "range, not requirement"):
- Peak score (1.0) for 5-9 years
- Linear decay outside the band, floor at 0.3 by year 2 or year 15
- Never hard-zero — strong fit elsewhere can still compensate

### 4. Tenure-stability fit — 10%
Penalize a pattern of company-switching every <18 months WITH escalating
seniority titles (the JD's exact complaint: chasing Senior→Staff→Principal
via hopping). A single short stint isn't penalized; a *pattern* of 3+ is.

### 5. Geo-eligibility fit — 8%
JD text (exact): *"Location: Pune/Noida, India (Hybrid) | Open to relocation
candidates from Tier-1 Indian cities."* Not a hard filter — scored:
- Located in Pune or Noida: 1.0
- Located in another Tier-1 Indian city (Bangalore/Bengaluru, Mumbai, Delhi
  NCR/Gurgaon, Hyderabad, Chennai, Kolkata) AND `willing_to_relocate == True`: 0.8
- Located in a Tier-1 Indian city, NOT willing to relocate: 0.4
- Elsewhere in India: 0.2
- Outside India: 0.1 (JD allows case-by-case for exceptional fit elsewhere
  in the score, never a hard exclusion)

### 6. Quality tiebreakers — 7%
- `github_activity_score` (if > -1, scaled 0-1; if -1, neutral not penalized)
- Education `tier` (tier_1/tier_2 small positive, tier_3/4/unknown neutral)
- `num_skill_assessments` taken on platform (engagement signal)

---

## Core JD Skills List (for skill-depth fit, component 2)

Pulled directly from job_description.docx, not reconstructed from memory:

```python
CORE_JD_SKILLS = {
    # Embeddings-based retrieval (JD explicit examples)
    "sentence-transformers", "openai embeddings", "bge", "e5",
    # Vector DBs / hybrid search infra (JD explicit examples)
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss",
    # Evaluation frameworks
    "ndcg", "mrr", "map", "a/b testing",
    # LLM fine-tuning
    "lora", "qlora", "peft", "fine-tuning",
    # Learning-to-rank
    "xgboost", "lightgbm", "learning to rank",
    # General domain terms (lower weight than the above specifics --
    # these are the ones keyword-stuffers list first)
    "embeddings", "retrieval", "ranking", "llm", "rag", "hybrid search",
}
```
Treat the bolded categories (embeddings-retrieval, vector-DB, evaluation,
fine-tuning, learning-to-rank) as higher-trust signals than the generic
bottom row — a candidate with "FAISS, advanced, 30 months" means more than
one with "embeddings, beginner, 2 months."

---

## Behavioral Multiplier (applied AFTER Fit Score, not blended into it)

Base = 1.0, multiplicative penalties/bonuses:

| Condition | Multiplier effect |
|---|---|
| `open_to_work_flag == False` | ×0.75 |
| `last_active_date` > 180 days ago | ×0.65 |
| `recruiter_response_rate < 0.10` | ×0.70 |
| `notice_period_days < 30` | ×1.10 |
| `notice_period_days > 90` | ×0.85 |

These compound multiplicatively. A candidate hitting the "ghost" pattern
(inactive 180d+ AND response_rate<0.10) lands around ×0.46 — roughly halving
an otherwise-strong fit score. This is a deliberate, data-backed choice: we
measured 179 real candidates who are simultaneously strong-on-paper AND
ghosts (inactive 180d+, response_rate<0.10) — almost a literal copy of the
JD's own example sentence. The multiplier needs to actually move these people
out of the top 10, not just nudge them.

Final cap: multiplier clamped to [0.3, 1.15] so no single behavioral flag can
fully erase a strong fit, and no combination of bonuses inflates a weak fit
past a strong one.

```
Final Score = Fit Score × Behavioral Multiplier
```

---

## Output Format (exact, from validate_submission.py — not optional)
- Filename: `<participant_id>.csv`
- Header exactly: `candidate_id,rank,score,reasoning`
- Exactly 100 data rows, ranks 1-100, each used exactly once
- Scores non-increasing by rank (rank 1 = highest score)
- Tie-break: equal scores → lower candidate_id gets the better (lower) rank

## Reasoning column requirements (per submission_spec)
For each ranked candidate, the reasoning text must cite 2-3 specific facts
that drove the score — not generic praise. Top-10 reasoning should name the
exact company/role/skill that justified the rank. Lower-tier reasoning should
honestly state the gap (e.g., "strong technical fit, but notice period is
120 days and response rate is 0.14").

---

## Architecture: separate pre-computation from ranking step
Confirmed via submission_spec.docx: pre-computation (embeddings) may exceed
5 minutes, but the final ranking step that produces the CSV must not.
Required structure:
- `precompute_embeddings.py` — slow, one-time, saves checkpoint (.pkl/.parquet).
  Document its runtime in submission_metadata.yaml's `pre_computation_time_minutes`.
- `rank.py` — fast, loads the checkpoint, does scoring math + honeypot filter
  + output. This is the one timed at ≤5 min in the Stage 3 sandbox.

## Known unknowns (resolve via testing, not guessing)
- Exact embedding-similarity threshold for honeypot exclusion (need to run
  embeddings against a labeled honeypot sample once code exists)
- Whether the weighting on Fit Score actually produces sane top-10s on real
  output — first full ranking run should be sanity-checked by hand
- **No labeled honeypot list exists anywhere in the provided files.**
  Honeypots must be derived by applying the Stage 1 rule definitions to the
  dataset, then sanity-checking that the resulting count lands near the
  README's stated ~80 — that's the only validation method available.
