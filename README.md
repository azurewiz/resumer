# ruby-mage-candidate-ranker

AI-driven candidate ranking system for the Redrob AI **Intelligent Candidate
Discovery** hackathon (Hack2Skill, team `azurewiz`, solo submission).

**Submission:** `azurewiz.csv` — top 100 candidates ranked for the released
Senior AI Engineer JD.

---

## What this builds

A two-stage pipeline that reads 100,000 candidate profiles and returns a
ranked top-100 shortlist:

**Stage 1 — Honeypot filter:** excludes profiles with internally impossible
facts (expert proficiency at a skill with zero months of use, or career
duration inconsistent with stated experience by >30%).

**Stage 2 — Composite scoring:** six weighted components combined into a
Fit Score, then multiplied by a behavioral-availability modifier.

| Component | Weight | What it measures |
|---|---|---|
| Career-context fit | 35% | Semantic similarity of career history to JD requirements (sentence embeddings) |
| Skill-depth fit | 25% | JD-relevant skills weighted by proficiency × duration_months |
| Experience-range fit | 15% | Soft bell curve peaking at 5–9 years |
| Tenure-stability fit | 10% | Penalizes company-hopping with climbing titles |
| Geo-eligibility fit | 8% | Pune/Noida first, Tier-1 India + relocation second |
| Quality tiebreakers | 7% | GitHub activity, education tier, platform assessments |

The behavioral multiplier (not a component — a post-score modifier) down-weights
candidates who are technically strong but practically unreachable: not open to
work, inactive >180 days, or recruiter response rate <10%.

Full methodology and design rationale: see `RUBRIC_v1.md`.

---

## Repo structure

```
.
├── load_candidates.py          # streams candidates.jsonl into a DataFrame
├── career_context_score.py     # sentence-transformer embeddings (pre-compute step)
├── skill_depth_score.py        # proficiency × duration skill matching
├── experience_score.py         # bell-curve experience fit
├── tenure_stability_score.py   # job-hopping pattern detection
├── geo_eligibility_score.py    # location scoring
├── quality_tiebreaker_score.py # github + education + assessments
├── behavioral_multiplier.py    # availability modifier
├── honeypot_filter.py          # Stage 1 trap exclusion
├── rank.py                     # combines everything → submission CSV
├── requirements.txt
├── submission_metadata.yaml
└── RUBRIC_v1.md                # full scoring design doc
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

# torch CPU build must come from PyTorch's own index, not regular PyPI
pip install torch --index-url https://download.pytorch.org/whl/cpu

# everything else
pip install -r requirements.txt
```

**Python version:** 3.12. **OS tested on:** Pop!_OS 22.04 (Ubuntu-based Linux).

---

## Data

`candidates.jsonl` is provided by the hackathon organizers (not redistributed
here). Place it in the repo root, or symlink it:

```bash
ln -s /path/to/candidates.jsonl ./candidates.jsonl
```

---

## Reproducing the submission

### Step 1 — Pre-computation (one-time, ~80 min on CPU)

```bash
python career_context_score.py --candidates ./candidates.jsonl
```

Embeds all 100K candidate career histories using `all-MiniLM-L6-v2` and saves
a checkpoint (`candidates_with_career_score.pkl`). This step may exceed the
5-minute ranking budget — that is expected and permitted per the submission
spec ("pre-computation may exceed the 5-minute window").

Then run the remaining scorers in order to build the full checkpoint chain:

```bash
python skill_depth_score.py
python experience_score.py
python tenure_stability_score.py
python geo_eligibility_score.py
python quality_tiebreaker_score.py
python behavioral_multiplier.py
python honeypot_filter.py
```

Each reads the previous checkpoint and writes the next one. All complete in
seconds (pure pandas math, no embeddings).

### Step 2 — Ranking (timed step, must complete in ≤5 min)

```bash
python rank.py --checkpoint candidates_with_all_scores.pkl --out azurewiz.csv
```

**Measured runtime: ~17 seconds** on a local CPU machine (16.89s wall-clock,
verified with `time python3 rank.py`). Well within the 5-minute budget.

### Step 3 — Validate output

```bash
python validate_submission.py azurewiz.csv
```

Expected output: `Submission is valid.`

---

## Key design decisions (Stage 5 quick reference)

**Why embeddings for career-context?**
Keyword matching can't catch a candidate who built a real retrieval system
without using the word "RAG." Found this concretely during data exploration
(CAND_0000031: genuine strong fit, no AI keywords in title). Embeddings
capture meaning, not word overlap.

**Why is the behavioral signal a multiplier, not a component?**
It answers a different question. Fit components ask "is this person good?"
The multiplier asks "can we actually hire them right now?" These are
independent dimensions that should not be averaged — they compound.

**Why did the honeypot filter change from AND to OR?**
Tested against the full dataset: signal 1 fires for 21 candidates, signal 2
fires for 49, overlap is exactly zero. AND logic was mathematically
guaranteed to flag nobody. OR logic flags 70 candidates, matching the
README's stated ~80 very closely.

**Why is skill-depth 25% and not higher?**
Skill-depth alone doesn't resist keyword stuffers. A Marketing Manager can
list every JD skill. Career-context (35%) catches them because their actual
career history doesn't semantically describe retrieval work. The two
cross-validate each other.

---

## AI tool usage

Built with Claude as an architecture reviewer and pair-programmer. Full
declaration in `submission_metadata.yaml`. The ranking pipeline makes zero
external API calls — fully offline.
