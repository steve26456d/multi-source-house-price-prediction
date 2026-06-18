```markdown
# multi-source-house-price-prediction Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill covers the development patterns and workflows used in the `multi-source-house-price-prediction` repository. The project is a Python-based codebase for predicting house prices using data from multiple sources. It includes scripts, Jupyter notebooks, experiment results, and documentation such as reports and presentations. This guide documents the coding conventions, collaborative workflows, and common commands to streamline contributions and maintain consistency.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python files and scripts.
  - Example: `data_loader.py`, `train_model.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .preprocessing import clean_data
    ```

### Export Style
- Use **named exports** (i.e., define functions/classes explicitly for import).
  - Example:
    ```python
    def train_model(...):
        ...
    ```

### Commit Messages
- Freeform, concise commit messages (~17 characters on average).
  - Example: `update results`, `fix data bug`

## Workflows

### Update Documentation and Presentation Materials
**Trigger:** When you need to update the project documentation, report, or presentation slides to reflect recent results or progress.  
**Command:** `/update-docs-presentation`

1. Edit or add to `Report.pdf` or `pre.pptx`.
2. Optionally update supporting files (e.g., `团队分工声明.pdf`, code/model files).
3. Commit changes with a docs-related message.

**Example:**
```bash
# Edit the report or presentation
git add Report.pdf pre.pptx
git commit -m "update docs and slides"
git push
```

---

### Add or Update Experimental Results and Visualizations
**Trigger:** When you want to log new experiment results or update visualizations after running new analyses.  
**Command:** `/add-results-visualizations`

1. Generate new results or figures (e.g., `.png`, `.csv`) in the `results/` directory.
2. Update or add to relevant Jupyter notebooks in `notebooks/`.
3. Commit all new/changed files together.

**Example:**
```bash
# Save new results and update notebook
git add results/new_experiment.png results/metrics.csv notebooks/analysis.ipynb
git commit -m "add new experiment results"
git push
```

---

### Update Git LFS Tracking for Large Files
**Trigger:** When you need to track new large binary files (e.g., presentation files) with Git LFS.  
**Command:** `/track-large-files`

1. Edit `.gitattributes` to add new file types or patterns (e.g., `*.pptx filter=lfs diff=lfs merge=lfs -text`).
2. Commit `.gitattributes` (and optionally `.gitignore`) with a docs-related message.

**Example:**
```bash
# Add .pptx to LFS tracking
echo "*.pptx filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
git add .gitattributes
git commit -m "track pptx files with LFS"
git push
```

## Testing Patterns

- **Framework:** Unknown (not explicitly detected).
- **File Pattern:** Test files typically follow the `*.test.*` naming convention.
  - Example: `model.test.py`
- **Best Practice:** Place test files alongside the modules they test or in a dedicated `tests/` directory.
- **Tip:** Use clear, descriptive function names for tests.

**Example:**
```python
# model.test.py
def test_train_model():
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

## Commands

| Command                       | Purpose                                                      |
|-------------------------------|--------------------------------------------------------------|
| /update-docs-presentation     | Update documentation files and presentation slides           |
| /add-results-visualizations   | Add or update experiment results and visualizations          |
| /track-large-files            | Update Git LFS tracking for new large/binary files           |
```
