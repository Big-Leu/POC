# ML Clustering Module Documentation

## Overview

The ML Clustering module is a machine learning-based solution for grouping similar records by using semantic similarity. This document explains the approach, methodology, and implementation details of the clustering algorithm and visualization components.

## Table of Contents

1. [Methodology](#methodology)
2. [Technical Implementation](#technical-implementation)
3. [Features](#features)
4. [Visualization Components](#visualization-components)
5. [Usage Guide](#usage-guide)
6. [Advanced Options](#advanced-options)

## Methodology

### Semantic Similarity Approach

The clustering module uses a semantic similarity-based approach to group similar records. Unlike traditional clustering algorithms that rely on exact matches or predefined rules, this approach uses natural language understanding to identify conceptually similar items even when they use different terminology.

### Key Concepts

1. **Embeddings**: Text data is converted into high-dimensional vector representations (embeddings) using a pre-trained transformer model (all-MiniLM-L6-v2). These embeddings capture the semantic meaning of the text.

2. **Similarity Measurement**: Cosine similarity is used to measure how similar two text embeddings are. Cosine similarity values range from 0 to 1, where 1 means identical and 0 means completely different.

3. **Threshold-Based Clustering**: Records are grouped based on a configurable similarity threshold. Higher thresholds create more distinct groups with higher internal similarity.

4. **Centroids**: Each cluster has a centroid (average of all embeddings in the group), which represents the "center" of that group.

5. **Relative Similarity**: Each record's similarity to its group's centroid is calculated, showing how well it fits within its assigned group.

## Technical Implementation

### Embedding Generation

The system uses the `sentence-transformers` library with the "all-MiniLM-L6-v2" model to generate embeddings. This model is optimized for semantic similarity tasks and offers a good balance between accuracy and performance.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
```

### Clustering Algorithm

The clustering algorithm follows these steps:

1. Generate embeddings for unique text values
2. For each embedding:
   - Check if it's similar to any existing cluster's centroid (based on threshold)
   - If yes, add to that cluster
   - If no, create a new cluster
3. Map text values to group IDs
4. Calculate group statistics (percentage of total, size)
5. Calculate relative similarity of each record to its group's centroid

```python
# Pseudo-code for the clustering process
clusters = []
for each embedding:
    found_cluster = False
    for each existing_cluster:
        if similarity(embedding, cluster_centroid) >= threshold:
            add_to_cluster(embedding)
            found_cluster = True
            break
    if not found_cluster:
        create_new_cluster(embedding)
```

### Multi-Column Analysis

The system supports analyzing multiple columns at once by combining them into a single text field before embedding generation. This allows for more comprehensive similarity analysis across different attributes.

## Features

### Key Features

1. **Dynamic Threshold Adjustment**: Users can adjust the similarity threshold to control how items are grouped. Higher thresholds create more groups with higher internal similarity.

2. **Multiple Column Analysis**: The system can analyze a single column or combine multiple columns for more comprehensive grouping.

3. **Interactive Visualizations**:
   - Similarity heatmaps showing relationships between items
   - Group size comparison charts
   - Group-specific heatmaps for detailed analysis

4. **Group Statistics**:
   - Group size and percentage of total records
   - Average similarity within each group
   - Representative samples from each group

5. **Data Export**: Results can be exported as CSV files for further analysis or reporting.

## Visualization Components

### Similarity Heatmaps

The similarity heatmap is an interactive visualization that shows the pairwise similarity between items. Brighter colors indicate higher similarity between items. This helps in understanding why certain records are grouped together.

### Group Size Comparison

The bar chart visualization shows the distribution of records across different groups, helping to identify dominant patterns or outliers in the data.

### Group-Specific Heatmaps

These heatmaps show the internal similarity structure of a specific group, allowing for detailed analysis of why certain records are grouped together and identifying potential sub-groups.

## Usage Guide

### Basic Usage

1. Upload a CSV file containing the data to be clustered (or use the "Generate Example Data" option)
2. Select the column(s) for analysis:
   - Single column mode: Choose one column containing text to analyze
   - Multiple column mode: Select multiple columns to combine for analysis
3. Adjust the similarity threshold (0.5-1.0) based on how strictly you want to group items
4. Process the data and analyze the results

### Using Example Data

For users who want to try the functionality without their own data:

1. Click the "Generate Example Data" button on the main screen
2. The system will create a sample dataset containing financial controls data
3. You can select columns to analyze (the "Control Type" column is recommended)
4. Adjust the threshold and click "Process Example Data"
5. Explore the clustering results and visualizations

### Interpreting Results

- **Group ID**: Unique identifier for each cluster
- **Group Size**: Number of records in each group
- **Group Percentage**: Percentage of total records in each group
- **Relative Similarity**: How similar each record is to its group's centroid (0-100%)

## Advanced Options

### Customizing Visualizations

The heatmaps and charts are customizable by adjusting parameters such as color scales, dimensions, and sampling limits for large datasets.

### Performance Considerations

- For large datasets, the system automatically samples data for visualizations to maintain performance
- Processing time increases with the number of records and the dimensionality of the text data

### Model Selection

The current implementation uses "all-MiniLM-L6-v2", but can be extended to use other embedding models based on specific requirements:
- Larger models for higher accuracy
- Multilingual models for non-English text
- Domain-specific models for specialized terminology

## Technical Requirements

- Python 3.7+
- Libraries: streamlit, pandas, numpy, sentence-transformers, plotly, scikit-learn

## Future Enhancements

Potential future enhancements include:
- Hierarchical clustering for nested grouping
- Anomaly detection to identify outliers
- Interactive cluster editing
- Support for additional embedding models
- Integration with external NLP services
