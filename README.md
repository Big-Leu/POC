# Personal POC Repository

This repository contains various Proof of Concepts (POCs) for personal learning and experimentation.

## Contents

- Sample code snippets
- Experimentation with new technologies
- Documentation and notes
- **Duplicate Detection App** (Streamlit, Python, NLP)

---

## Duplicate Detection App

This project includes a powerful **Duplicate Detection App** built with Streamlit and Python. It allows you to upload a CSV file and find potential duplicate records using a hybrid approach:

- **Exact Match**: Identifies records that are identical across selected columns.
- **Fuzzy Match**: Uses text similarity (Levenshtein distance) to find near-duplicates (threshold adjustable).
- **Semantic Similarity**: Uses sentence embeddings (via `sentence-transformers`) to find records with similar meaning (threshold adjustable).

### Features

- **Robust CSV Parsing**: Handles malformed or inconsistent CSV files gracefully.
- **Column Selection**: Choose which columns to use for duplicate detection.
- **Threshold Controls**: Adjust fuzzy and semantic similarity thresholds interactively.
- **Progress Bar**: Visual feedback during processing, suitable for large datasets.
- **Prioritized Results**: Shows top 3 duplicates per record, prioritizing Exact > Fuzzy > Semantic matches.
- **Expandable Results**: View details for each record and its top duplicates.

---

## How It Works

1. **Upload CSV**: The app accepts a CSV file with any number of columns. You can select up to 4 columns for analysis (default: first 4).
2. **Data Preview**: See a preview of your data before running analysis.
3. **Set Thresholds**: Use sliders to set the minimum similarity for fuzzy and semantic matches (default: 90%).
4. **Run Detection**: Click 'Detect Duplicates' to start. The app will:
    - Combine selected columns into a single text field per row.
    - Generate sentence embeddings for semantic similarity.
    - Compute a similarity matrix for all records.
    - For each record, compare with all others:
        - If all selected columns match exactly, it's an **Exact** match.
        - If fuzzy text similarity (average across columns) exceeds the threshold, it's a **Fuzzy** match.
        - If semantic similarity exceeds the threshold, it's a **Semantic** match.
    - Only the highest-priority match type is kept for each pair.
    - Top 3 matches per record are shown, sorted by priority and score.
5. **View Results**: Expand each record to see its top potential duplicates, with color-coded match types.

---

## How to Run

### Prerequisites
- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended for dependency management)

### Install Dependencies

If using Poetry:
```bash
poetry install
```


### Start the App
```bash
poetry run streamlit run  backend/app.py
```

The app will open in your browser. Upload your CSV and follow the on-screen instructions.

---

## Example CSV Format

```
name,description,category,notes
Widget A,Small blue widget,Tools,For home use
Widget B,Small blu widget,Tools,For home use
Widget C,Large red widget,Gadgets,Outdoor
```

---

## Project Structure

```
POC/
├── backend/
│   ├── app.py           # Streamlit app for duplicate detection
│   └── __init__.py
├── poetry.lock
├── pyproject.toml
├── README.md
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.