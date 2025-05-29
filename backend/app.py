import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

st.title("Duplicate Detection App")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load data with robust error handling
    try:
        # First attempt - standard settings
        df = pd.read_csv(uploaded_file)
    except pd.errors.ParserError:
        uploaded_file.seek(0)  # Reset file pointer
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines="skip")
            st.warning(
                "Some rows were skipped due to formatting issues."
            )
        except Exception as e:
            # If all else fails, try with the Python engine
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                engine="python",
                on_bad_lines="skip",
            )
            st.warning(f"Loaded CSV with some issues: {str(e)}")

    st.success(f"Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")

    # Data preview
    with st.expander("Data Preview"):
        st.dataframe(df.head())

    # Select columns
    st.subheader("Select Columns for Analysis")
    available_columns = df.columns.tolist()
    if len(available_columns) > 4:
        default_columns = available_columns[:4]
    else:
        default_columns = available_columns

    selected_columns = st.multiselect(
        "Select columns for duplicate detection (default: first 4)",
        available_columns,
        default=default_columns,
    )

    # Set thresholds
    col1, col2 = st.columns(2)
    with col1:
        fuzzy_threshold = st.slider("Fuzzy Match Threshold (%)", 50, 100, 90)
    with col2:
        semantic_threshold = st.slider(
            "Semantic Similarity Threshold (%)", 50, 100, 90
        )

    # Process button
    if st.button("Detect Duplicates"):
        if not selected_columns:
            st.error("Please select at least one column for analysis")
        else:
            # Create combined text field for similarity calculation
            df["combined_text"] = (
                df[selected_columns].astype(str).agg(" ".join, axis=1)
            )

            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Generate embeddings for semantic similarity
            status_text.write("Generating embeddings...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(
                df["combined_text"].tolist(),
                show_progress_bar=True,
            )
            progress_bar.progress(25)

            # Calculate similarity matrix
            status_text.write("Calculating similarity matrix...")
            similarity_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(similarity_matrix, 0)  # Avoid self-matches
            progress_bar.progress(50)

            # Precompute hashes for exact matching
            hash_map = {}
            for idx, row in df[selected_columns].astype(str).iterrows():
                row_tuple = tuple(row)
                row_hash = hash(row_tuple)
                if row_hash not in hash_map:
                    hash_map[row_hash] = []
                hash_map[row_hash].append(idx)

            # Find duplicates for each record
            status_text.write("Finding potential duplicates...")
            results = []
            total_records = len(df)

            for i in range(total_records):
                if i % max(1, total_records // 50) == 0:
                    current_progress = 50 + int((i / total_records) * 50)
                    progress_bar.progress(current_progress)
                    status_text.write(
                        f"Processing record {i+1}/{total_records}"
                    )
                row_data = df.iloc[i]
                row_tuple = tuple(row_data[selected_columns].astype(str))
                row_hash = hash(row_tuple)
                potential_duplicates = []

                # 1. Check for exact matches using hash map
                for j in hash_map[row_hash]:
                    if i == j:
                        continue
                    compare_data = df.iloc[j]
                    if all(
                        row_data[col] == compare_data[col]
                        for col in selected_columns
                    ):
                        potential_duplicates.append(
                            {"index": j, "match_type": "Exact", "score": 100.0}
                        )
                if potential_duplicates:
                    # If exact matches found, skip fuzzy/semantic
                    pass
                else:
                    # 2. Fuzzy and semantic checks for all other records
                    for j in range(total_records):
                        if i == j:
                            continue
                        compare_data = df.iloc[j]
                        # Fuzzy match
                        fuzzy_scores = []
                        for col in selected_columns:
                            if pd.notna(row_data[col]) and \
                                    pd.notna(compare_data[col]):
                                score = fuzz.ratio(
                                    str(row_data[col]), str(compare_data[col])
                                )
                                fuzzy_scores.append(score)
                            else:
                                fuzzy_scores.append(0)
                        avg_fuzzy_score = (
                            sum(fuzzy_scores) / len(fuzzy_scores)
                            if fuzzy_scores else 0
                        )
                        if avg_fuzzy_score >= fuzzy_threshold:
                            potential_duplicates.append(
                                {
                                    "index": j,
                                    "match_type": "Fuzzy",
                                    "score": avg_fuzzy_score,
                                }
                            )
                            continue
                        # Semantic match
                        semantic_score = similarity_matrix[i, j] * 100
                        if semantic_score >= semantic_threshold:
                            potential_duplicates.append(
                                {
                                    "index": j,
                                    "match_type": "Semantic",
                                    "score": semantic_score,
                                }
                            )

                # Sort duplicates by priority and score
                def sort_key(x):
                    type_priority = {"Exact": 0, "Fuzzy": 1, "Semantic": 2}
                    return (type_priority[x["match_type"]], -x["score"])

                potential_duplicates.sort(key=sort_key)

                # Keep top 3 duplicates
                top_duplicates = potential_duplicates[:3]

                if top_duplicates:
                    results.append(
                        {
                            "original_idx": i,
                            "original_data": {
                                col: row_data[col] for col in selected_columns
                            },
                            "duplicates": [
                                {
                                    "idx": dup["index"],
                                    "data": {
                                        col: df.iloc[dup["index"]][col]
                                        for col in selected_columns
                                    },
                                    "match_type": dup["match_type"],
                                    "score": dup["score"],
                                }
                                for dup in top_duplicates
                            ],
                        }
                    )

            # Complete the progress bar
            progress_bar.progress(100)
            status_text.write("Processing complete!")

            # Display results
            st.subheader("Duplicate Detection Results")

            if not results:
                st.info("No duplicates found with the current settings.")
            else:
                st.write(
                    f"Found potential duplicates for {len(results)} records."
                )

                for i, result in enumerate(results):
                    with st.expander(f"Record #{result['original_idx'] + 1}"):
                        # Show original record
                        st.write("**Original Record:**")
                        for col, val in result["original_data"].items():
                            st.write(f"- {col}: {val}")

                        # Show duplicates
                        st.write("**Top Potential Duplicates:**")
                        for j, dup in enumerate(result["duplicates"]):
                            match_color = {
                                "Exact": "red",
                                "Fuzzy": "orange",
                                "Semantic": "green",
                            }[dup["match_type"]]
                            record = f"** Record #{dup['idx'] + 1} -"
                            st.markdown(
                                f"**Duplicate {j+1}:{record} "
                                f"<span style='color:{match_color}'>"
                                f"{dup['match_type']} match ("
                                f"{dup['score']:.2f}%"
                                f")</span>",
                                unsafe_allow_html=True,
                            )

                            for col, val in dup["data"].items():
                                st.write(f"- {col}: {val}")
