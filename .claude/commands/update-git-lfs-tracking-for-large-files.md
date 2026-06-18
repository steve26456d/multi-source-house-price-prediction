---
name: update-git-lfs-tracking-for-large-files
description: Workflow command scaffold for update-git-lfs-tracking-for-large-files in multi-source-house-price-prediction.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /update-git-lfs-tracking-for-large-files

Use this workflow when working on **update-git-lfs-tracking-for-large-files** in `multi-source-house-price-prediction`.

## Goal

Updates .gitattributes to add large files (such as .pptx) to Git LFS tracking.

## Common Files

- `.gitattributes`
- `.gitignore`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit .gitattributes to add new file types or patterns
- Commit .gitattributes (and optionally .gitignore) with a docs-related message

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.