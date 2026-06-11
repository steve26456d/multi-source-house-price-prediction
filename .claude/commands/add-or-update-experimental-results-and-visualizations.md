---
name: add-or-update-experimental-results-and-visualizations
description: Workflow command scaffold for add-or-update-experimental-results-and-visualizations in multi-source-house-price-prediction.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-update-experimental-results-and-visualizations

Use this workflow when working on **add-or-update-experimental-results-and-visualizations** in `multi-source-house-price-prediction`.

## Goal

Adds or updates experiment result files and visualization figures, often in the results/ directory and related notebooks.

## Common Files

- `results/*.png`
- `results/*.csv`
- `notebooks/*.ipynb`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Generate new results or figures (e.g., .png, .csv) in results/
- Update or add to relevant Jupyter notebooks in notebooks/
- Commit all new/changed files together

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.