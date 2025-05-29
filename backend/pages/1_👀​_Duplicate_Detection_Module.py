"""
Duplicate Detection Module for CSV data.

This module allows users to upload a CSV file and find potential duplicates
using three different matching approaches:
- Exact matching (using hash-based optimization)
- Fuzzy matching (using Levenshtein distance)
- Semantic matching (using sentence embeddings)
"""

import os
import sys

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (calculate_fuzzy_score, create_hash_map, format_match_result,
                   generate_embeddings, load_csv_with_error_handling,
                   load_model, prepare_combined_text, setup_progress_tracking,
                   sort_matches_by_priority, timer_decorator, update_progress)

st.title("Duplicate Detection App")
st.markdown(
    """
This app helps you find potential duplicate records in your CSV data using three different matching techniques:
- **Exact Matching**: Finds records that match exactly on specified columns
- **Fuzzy Matching**: Finds similar records based on text similarity
- **Semantic Matching**: Finds records with similar meaning using NLP
"""
)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load data with robust error handling
    df = load_csv_with_error_handling(uploaded_file)
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
        else:  # Create combined text field for similarity calculation
            df["combined_text"] = prepare_combined_text(df, selected_columns)

            # Setup progress tracking
            progress_bar, status_text = setup_progress_tracking()

            # Generate embeddings for semantic similarity
            update_progress(
                progress_bar,
                status_text,
                "Embeddings",
                0,
                "Generating embeddings...",
            )
            model = load_model()
            embeddings = generate_embeddings(
                df["combined_text"].tolist(),
                model,
                batch_size=64,  # Increased batch size for better performance
            )
            update_progress(progress_bar, status_text, "Embeddings", 25)

            # Calculate similarity matrix
            update_progress(
                progress_bar,
                status_text,
                "Similarity",
                25,
                "Calculating similarity matrix...",
            )
            similarity_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(similarity_matrix, 0)  # Avoid self-matches
            update_progress(progress_bar, status_text, "Similarity", 50)

            # Precompute hashes for exact matching
            hash_map = create_hash_map(df, selected_columns)

            # Find duplicates for each record
            update_progress(
                progress_bar,
                status_text,
                "Processing",
                50,
                "Finding potential duplicates...",
            )
            results = []
            total_records = len(df)

            for i in range(total_records):
                if i % max(1, total_records // 50) == 0:
                    current_progress = 50 + int((i / total_records) * 50)
                    update_progress(
                        progress_bar,
                        status_text,
                        "Processing",
                        current_progress,
                        f"Processing record {i+1}/{total_records}",
                    )

                row_data = df.iloc[i]
                row_tuple = tuple(row_data[selected_columns].astype(str))
                row_hash = hash(row_tuple)
                potential_duplicates = []

                # 1. Check for exact matches using hash map
                if row_hash in hash_map:
                    for j in hash_map[row_hash]:
                        if i == j:
                            continue
                        compare_data = df.iloc[j]
                        if all(
                            row_data[col] == compare_data[col]
                            for col in selected_columns
                        ):
                            potential_duplicates.append(
                                {
                                    "index": j,
                                    "match_type": "Exact",
                                    "score": 100.0,
                                }
                            )

                    # If exact matches found, skip fuzzy/semantic
                    if potential_duplicates:
                        pass
                    else:
                        # 2. Fuzzy and semantic checks
                        for j in range(total_records):
                            if i == j:
                                continue

                            compare_data = df.iloc[j]

                            # Fuzzy match
                            avg_fuzzy_score = calculate_fuzzy_score(
                                row_data, compare_data, selected_columns
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
                potential_duplicates = sort_matches_by_priority(
                    potential_duplicates
                )

                # Keep top 3 duplicates
                top_duplicates = potential_duplicates[:3]

                if top_duplicates:
                    # Prepare original data
                    original_data = {}
                    for col in selected_columns:
                        original_data[col] = row_data[col]

                    # Prepare duplicates data
                    duplicates = []
                    for dup in top_duplicates:
                        dup_data = {}
                        for col in selected_columns:
                            dup_data[col] = df.iloc[dup["index"]][col]

                        duplicates.append(
                            {
                                "idx": dup["index"],
                                "data": dup_data,
                                "match_type": dup["match_type"],
                                "score": dup["score"],
                            }
                        )

                    results.append(
                        {
                            "original_idx": i,
                            "original_data": original_data,
                            "duplicates": duplicates,
                        }
                    )

            # Complete the progress bar
            update_progress(
                progress_bar,
                status_text,
                "Complete",
                100,
                "Processing complete!",
            )

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
                            record = f"** Record #{dup['idx'] + 1}"
                            formatted_match = format_match_result(
                                dup["match_type"], dup["score"]
                            )
                            st.markdown(
                                f"**Duplicate {j+1}:** {record} {formatted_match}",
                                unsafe_allow_html=True,
                            )

                            for col, val in dup["data"].items():
                                st.write(f"- {col}: {val}")
else:
    st.info("Please upload a CSV file to begin analysis.")
