# LDA and Thematic Analysis of Biblical and Contemporary Literature

This repository contains the Python code used in:

> Alessandra Luce and Paul Robertson, "Latent Dirichlet Allocation (LDA) and Thematic Analysis of Biblical and Contemporary Literature". (forthcoming)

The study applies Latent Dirichlet Allocation to hand-tagged rhetorical features in Paul's letters and a set of roughly contemporary comparanda texts (Epictetus, Seneca, Philodemus, Aelius Aristides, the Letter to the Hebrews, 4 Maccabees, and the Damascus Document), then compares topic distributions using Jensen-Shannon divergence.

---

## Repository Contents

| File | Description |
|------|-------------|
| `verse_refs.py` | Generates a complete verse-by-verse reference table for all texts |
| `decile_ranges.py` | Assigns decile labels to each verse within each letter |
| `deciles.py` | Assigns decile labels to all feature observations across both biblical and non-biblical texts |
| `LDA.py` | Trains the LDA model on Paul's letters, calculates the Paul baseline, and computes Jensen-Shannon divergence scores for each comparandum text |

Scripts should be run in the order listed above.

---

## Input Data Format

The scripts expect hand-tagged feature data in CSV format. Each feature CSV (one per text) should have the following columns:

- `Characteristics` - the name of the rhetorical feature (e.g., `rhetorical questions`, `metaphors`)
- `Appearances` - a comma-separated list of segment references where that feature occurs (e.g., `1:3, 2:7, 4:12` for biblical texts)

The `clean_data/` directory containing all feature CSVs used in the study is included in this repository. The twenty rhetorical features used in this study are:

- universal claims
- appeals to authority
- conversation
- personification
- rhetorical questions
- metaphors
- anecdotes/examples
- imperatives
- exhortation
- caustic injunctions
- pathos
- irony or satire
- hyperbole
- oppositions or choices
- figurations of groupness
- second person addresses
- plural inclusive addresses
- first person reflection
- analysis of questions/objections
- systematic structure

For full discussion of how these features were derived and applied, see the paper and the references therein (especially Robertson 2016 and Robertson 2025).

---

## Directory Structure

Before running the scripts, create the following folders alongside the scripts:

```
/clean_data/       - your input feature CSVs (one per text)
/output/           - generated verse reference files (created by verse_refs.py)
/deciles/          - generated decile-tagged CSVs (created by deciles.py)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy gensim
```

Developed and tested with Python 3.10+, gensim 4.3.3, pandas 2.x, numpy 1.x.

### 2. Generate verse references

```bash
python verse_refs.py
```

Produces `output/verse_references_per_letter.csv`, a flat table of all verses across all texts.

### 3. Assign decile ranges

```bash
python decile_ranges.py
```

Produces `output/verse_references_with_deciles.csv` and `output/decile_verse_ranges.csv`. These are the decile boundary tables reported in Appendix C of the paper.

### 4. Build decile-tagged feature files

```bash
python deciles.py
```

Reads all CSVs from `clean_data/` and writes decile-tagged versions to `deciles/`. Handles biblical texts (verse-based deciles) and non-biblical texts (structure-based deciles) separately.

### 5. Run LDA and compute divergence scores

```bash
python LDA.py
```

- Trains LDA models with 2, 5, and 10 topics on Paul's seven undisputed letters and reports coherence scores for each
- Uses the 5-topic model (selected on interpretive grounds; see §6 of the paper) for all further analysis
- Prints per-decile topic distributions and Jensen-Shannon divergence scores relative to the Paul baseline for each comparandum text

---

## Segmentation Notes

**Biblical texts** (Paul's letters, Hebrews, 4 Maccabees) are segmented by verse, with deciles assigned proportionally across the total verse count of each letter or book.

**Non-biblical texts** (Epictetus, Seneca, Philodemus, Aelius Aristides, Damascus Document) are segmented using the structural divisions of the text (book/chapter/section or column/line, depending on the work). The full section lists used for decile assignment are defined in `deciles.py`. See §5 of the paper for the rationale behind this segmentation approach.

---

## Reproducibility Note

LDA is a probabilistic algorithm. Results may vary slightly across runs due to random initialization. To reproduce the exact results reported in the paper, use `random_state=42` in the `LdaModel` call (this is set by default in `LDA.py`) and gensim version 4.3.3.
