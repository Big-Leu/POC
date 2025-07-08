import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from logging import getLogger

logger = getLogger(__name__)

# Set up basic Streamlit-friendly logging (prints to Streamlit console)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="ML Clustering", layout="wide")


@st.cache_resource
def get_embedding_model():
    """Get the embedding model for generating embeddings with caching"""
    import torch
    
    # Function to try different model loading approaches
    def try_load_with_approach(approach_name, load_func):
        try:
            st.info(f"Trying {approach_name}...")
            return load_func(), None
        except Exception as e:
            return None, str(e)
    
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Approach 1: Simple default loading
    model, error = try_load_with_approach(
        "default loading",
        lambda: SentenceTransformer("all-MiniLM-L6-v2")
    )
    if model:
        return model
    
    # If we got a meta tensor error, try specialized approaches
    if error and "meta tensor" in error.lower():
        st.warning(f"Meta tensor error detected: {error}")
        
        # Approach 2: Try with device_map="auto"
        model, error = try_load_with_approach(
            "device_map='auto'",
            lambda: SentenceTransformer("all-MiniLM-L6-v2", device_map="auto")
        )
        if model:
            return model
            
        # Approach 3: Try with explicit device
        model, error = try_load_with_approach(
            f"explicit device={device}",
            lambda: SentenceTransformer("all-MiniLM-L6-v2", device=device)
        )
        if model:
            return model
            
        # Approach 4: Try to_empty pattern
        model, error = try_load_with_approach(
            "to_empty() method",
            lambda: SentenceTransformer(
                "all-MiniLM-L6-v2", 
                device="meta"
            ).to_empty(device=device)
        )
        if model:
            return model
            
        # Approach 5: Try smaller model as fallback
        st.warning("All approaches failed. Trying a smaller model as fallback...")
        model, error = try_load_with_approach(
            "smaller model (paraphrase-MiniLM-L3-v2)",
            lambda: SentenceTransformer("paraphrase-MiniLM-L3-v2", device=device)
        )
        if model:
            st.success("Using smaller model as fallback.")
            return model
    
    # If we get here, nothing worked
    st.error(f"Failed to load embedding model: {error}")
    raise RuntimeError(
        f"Could not load embedding model after multiple attempts: {error}"
    )


@st.cache_data(ttl=3600)
def generate_embeddings_robust(_model, texts, initial_batch_size=64):
    """Generate embeddings with retry logic for handling errors and caching"""
    batch_sizes = [initial_batch_size, 32, 16, 8]
    last_error = None
    
    for batch_size in batch_sizes:
        try:
            return _model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True
            )
        except Exception as e:
            last_error = e
            st.warning(
                f"Error with batch_size={batch_size}: {str(e)}. "
                f"Retrying with smaller batch..."
            )
    
    # If we've exhausted all batch sizes, try one last attempt
    # with additional parameters
    try:
        st.warning("Trying final fallback approach...")
        return _model.encode(
            texts,
            batch_size=4,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    except Exception as e:
        st.error(f"All embedding attempts failed: {str(e)}")
        raise RuntimeError(f"Failed to generate embeddings: {last_error}")


def process_clustering(df, text_columns, threshold):
    """
    Process the clustering logic on the given dataframe
    
    Args:
        df: The dataframe containing the data
        text_columns: The column(s) to use for clustering (can be a single column or list)
        threshold: The similarity threshold to use
    
    Returns:
        Tuple containing:
        - The processed dataframe with clustering information
        - Similarity matrix for all items
        - Text mapping for similarity matrix labels
    """
    # Clone the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Handle single column or multiple columns
    is_multi_column = isinstance(text_columns, list) and len(text_columns) > 1
    if is_multi_column:
        # Combine multiple columns into a single text field for embedding
        result_df['combined_text'] = result_df.apply(
            lambda row: " | ".join([
                f"{col}: {str(row[col])}"
                for col in text_columns if pd.notna(row[col])
            ]),
            axis=1
        )
        analysis_column = 'combined_text'
        # Store original columns for reference
        result_df['_original_columns'] = str(text_columns)
    else:
        # If it's a single column or a list with one item
        analysis_column = text_columns[0] if isinstance(text_columns, list) else text_columns
    
    # Generate embeddings for unique text values
    model = get_embedding_model()
    unique_texts = result_df[analysis_column].unique()
    
    # Generate embeddings with robust handling
    with st.spinner("Generating embeddings for unique texts..."):
        embeddings = generate_embeddings_robust(_model=model, texts=unique_texts)
    
    # Compute full similarity matrix for unique texts (for heatmap)
    with st.spinner("Computing similarity matrix..."):
        sim_matrix = cosine_similarity(embeddings)
        text_mapping = {i: text for i, text in enumerate(unique_texts)}
    
    # Clustering logic
    clusters = []
    
    # First pass to form clusters
    for idx, embedding in enumerate(embeddings):
        found_cluster = False
        for cluster in clusters:
            # Check similarity against cluster centroid
            sim_score = cosine_similarity([embedding], [cluster[0]]).flatten()[0]
            if sim_score >= threshold:
                cluster[1].append(unique_texts[idx])
                found_cluster = True
                break
        if not found_cluster:
            clusters.append((embedding, [unique_texts[idx]]))

    # Map text value to group_id
    text_to_group = {}
    for group_id, cluster in enumerate(clusters, start=1):
        for text in cluster[1]:
            text_to_group[text] = group_id

    # Add group_id to dataframe
    result_df["group_id"] = result_df[analysis_column].map(text_to_group)
    
    # Calculate group percentages
    group_counts = result_df["group_id"].value_counts().to_dict()
    total_records = len(result_df)
    result_df["group_percentage"] = result_df["group_id"].map(
        lambda gid: round((group_counts[gid] / total_records) * 100, 2)
    )
    
    # Compute embeddings for ALL rows and centroids
    with st.spinner("Generating embeddings for all rows..."):
        # Use our robust embedding function
        all_embeddings = generate_embeddings_robust(
            _model=model,
            texts=result_df[analysis_column].tolist()
        )
        result_df["embedding"] = list(all_embeddings)
    
    # Compute group centroids
    group_centroids = {}
    for gid in result_df["group_id"].unique():
        # Get embeddings for this group
        group_mask = result_df["group_id"] == gid
        group_vectors = np.vstack(result_df[group_mask]["embedding"].values)
        centroid = np.mean(group_vectors, axis=0)
        group_centroids[gid] = centroid
    
    # Compute relative similarity to centroid
    def compute_relative_similarity(row):
        # Get similarity between this row and its group centroid
        centroid = group_centroids[row["group_id"]]
        sim = cosine_similarity([row["embedding"]], [centroid]).flatten()[0]
        return round(sim * 100, 2)
    
    # Apply similarity calculation to each row
    result_df["relative_similarity"] = result_df.apply(
        compute_relative_similarity, axis=1
    )
    
    # Clean up by removing the embedding column
    result_df = result_df.drop(columns=["embedding"])
    
    return result_df, sim_matrix, text_mapping


def render_similarity_badge(value):
    """Render a colored badge for the similarity percentage"""
    # Since we can't use HTML directly in dataframes, we'll use simple text
    # In a real app, you'd want to use a more sophisticated approach
    # such as custom CSS or a component library
    if value >= 90:
        return f"🟢 {value}%"  # Green circle for high similarity
    elif value >= 80:
        return f"🔵 {value}%"  # Blue circle for good similarity
    elif value >= 70:
        return f"🟡 {value}%"  # Yellow circle for moderate similarity
    else:
        return f"🔴 {value}%"  # Red circle for low similarity


def generate_similarity_heatmap_plotly(similarity_matrix, text_mapping, df, text_column, max_items=40):
    """
    Generate an interactive plotly heatmap of the similarity matrix
    
    Args:
        similarity_matrix: The similarity matrix
        text_mapping: Mapping from index to text
        df: The original dataframe
        text_column: The column with text being analyzed
        max_items: Maximum number of items to include in the heatmap
    
    Returns:
        A plotly figure object
    """
    # If we have too many items, sample them
    n = similarity_matrix.shape[0]
    if n > max_items:
        # Select a subset of the matrix
        indices = np.random.choice(n, size=max_items, replace=False)
        sim_subset = similarity_matrix[np.ix_(indices, indices)]
        labels = [text_mapping[i] for i in indices]
    else:
        sim_subset = similarity_matrix
        labels = [text_mapping[i] for i in range(n)]
    
    # Truncate labels if they're too long
    labels = [label[:30] + "..." if len(label) > 30 else label for label in labels]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sim_subset,
        x=labels,
        y=labels,
        hoverongaps=False,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Similarity')
    ))
    
    fig.update_layout(
        title="Text Similarity Heatmap",
        xaxis_title="Text Items",
        yaxis_title="Text Items",
        width=800,
        height=800
    )
    
    # Make x-axis labels vertical for better readability
    fig.update_xaxes(tickangle=90)
    
    return fig


@st.cache_data(ttl=3600)
def generate_group_heatmap(df, text_column, selected_group):
    """Generate a heatmap for a specific group with robust error handling.
    
    Args:
        df: Dataframe with clustering results
        text_column: Column containing text data
        selected_group: The group ID to visualize
        
    Returns:
        A plotly Figure object or None if error/no data
    """
    try:
        import torch
        
        # Get items in the selected group
        group_df = df[df["group_id"] == selected_group]
        if len(group_df) == 0:
            logger.warning(f"[Group Heatmap] No items in group {selected_group}")
            return None
        elif len(group_df) == 1:
            logger.warning(f"[Group Heatmap] Only 1 item in group {selected_group}")
            return None
            
        # Get unique texts and sample if needed
        texts = group_df[text_column].unique().tolist()
        n_texts = len(texts)
        
        # Sample if too many items
        max_items = 50
        if n_texts > max_items:
            np.random.seed(42)  # For consistent sampling
            indices = np.random.choice(n_texts, max_items, replace=False)
            texts = [texts[i] for i in indices]
            logger.info(f"[Group Heatmap] Sampled {max_items} from {n_texts}")
        
        try:
            # Get model and generate embeddings safely
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[Group Heatmap] Using device: {device}")
            
            # Generate embeddings with retries
            try_count = 0
            max_tries = 3
            last_error = None
            
            while try_count < max_tries:
                try:
                    model = get_embedding_model()
                    logger.info("[Group Heatmap] Generating embeddings...")
                    embeddings = generate_embeddings_robust(
                        _model=model,
                        texts=texts,
                        initial_batch_size=min(16, len(texts))
                    )
                    break
                except Exception as e:
                    try_count += 1
                    last_error = e
                    logger.warning(f"[Group Heatmap] Embedding attempt {try_count} failed: {e}")
                    if try_count < max_tries:
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise RuntimeError(f"Failed after {max_tries} attempts: {last_error}")
            
            # Generate similarity matrix
            logger.info("[Group Heatmap] Computing similarities...")
            sim_matrix = cosine_similarity(embeddings)
            
            # Create labels
            labels = [t[:40] + "..." if len(t) > 40 else t for t in texts]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=sim_matrix,
                x=labels,
                y=labels,
                hoverongaps=False,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Similarity", thickness=15)
            ))
            
            # Configure layout
            height = min(max(400, len(texts) * 25), 800)
            width = min(max(400, len(texts) * 25), 800)
            
            fig.update_layout(
                title=f"Group {selected_group} Similarity ({len(texts)} items)",
                xaxis_title="Text Items",
                yaxis_title="Text Items",
                height=height,
                width=width,
                margin=dict(t=50, l=50, r=50, b=100),
                showlegend=False
            )
            
            # Make labels more readable
            fig.update_xaxes(
                tickangle=45,
                tickfont=dict(size=10),
                showgrid=False
            )
            fig.update_yaxes(
                tickfont=dict(size=10),
                showgrid=False
            )
            
            logger.info(f"[Group Heatmap] Generated plot for group {selected_group}")
            return fig
            
        except Exception as e:
            logger.error(f"[Group Heatmap] Model/embedding error: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
            
    except Exception as e:
        logger.error(f"[Group Heatmap] Unexpected error: {str(e)}")
        return None


def generate_group_comparison_chart(df):
    """
    Generate a bar chart comparing the sizes of different groups
    
    Args:
        df: Dataframe with clustering results
        
    Returns:
        A plotly figure object
    """
    # Get group counts
    group_counts = df.groupby('group_id').size().reset_index(name='count')
    group_counts = group_counts.sort_values('count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        group_counts, 
        x='group_id', 
        y='count',
        labels={'count': 'Number of Records', 'group_id': 'Group ID'},
        title='Group Size Comparison',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis=dict(type='category'),
        width=800,
        height=500
    )
    
    return fig


def render_cluster_visualizations(df, text_column, sim_matrix, text_mapping):
    """
    Render various visualizations for clustering results
    
    Args:
        df: Dataframe with clustering results
        text_column: Column containing text data
        sim_matrix: Similarity matrix for all items
        text_mapping: Text mapping for similarity matrix labels
    """
    # Create a unique ID for this visualization session
    viz_session_id = str(hash(tuple([str(df.shape), str(text_column)])))[:10]
    viz_state_key = f"viz_data_{viz_session_id}"
    
    # Store these values in session state to preserve them
    if viz_state_key not in st.session_state:
        st.session_state[viz_state_key] = {
            'df': df,
            'text_column': text_column,
            'sim_matrix': sim_matrix,
            'text_mapping': text_mapping,
            'active_tab': 0  # Default to first tab
        }
    
    st.markdown("### Cluster Visualizations")
    
    # Create tabs for different visualizations
    tab_labels = ["Overall Similarity", "Group Comparison", "Group Details"]
    
    # Function to handle tab selection
    def on_tab_select(tab_index):
        st.session_state[viz_state_key]['active_tab'] = tab_index
        
    # Determine which tab was previously active
    if "active_tab_index" not in st.session_state:
        st.session_state["active_tab_index"] = 0
        
    # Create the tabs with the active tab preserved
    viz_tabs = st.tabs(tab_labels)
    
    with viz_tabs[0]:
        st.markdown("#### Similarity Heatmap")
        st.markdown("""
        This heatmap shows the similarity between different text items. 
        Brighter colors indicate higher similarity between items.
        """)
        
        # Use a container with a key to avoid recomputing
        with st.container():
            if st.checkbox("Show Overall Similarity Heatmap", value=True, key="show_overall"):
                with st.spinner("Generating overall similarity heatmap..."):
                    overall_fig = generate_similarity_heatmap_plotly(
                        sim_matrix, text_mapping, df, text_column
                    )
                    st.plotly_chart(overall_fig, use_container_width=True)
    
    with viz_tabs[1]:
        st.markdown("#### Group Size Comparison")
        st.markdown("""
        This chart compares the sizes of different groups.
        """)
        
        # Use a container with a key to avoid recomputing
        with st.container():
            group_comparison_fig = generate_group_comparison_chart(df)
            st.plotly_chart(group_comparison_fig, use_container_width=True)
    
    with viz_tabs[2]:
        st.markdown("#### Group-Specific Visualization")
        st.markdown("""
        This visualization shows similarity within specific groups.
        Select a group to see the similarity patterns among its items.
        """)
        
        # Create group options
        group_options = sorted(df["group_id"].unique())
        if group_options:
            # Add group size to the labels
            group_sizes = df["group_id"].value_counts()
            group_labels = [
                f"Group {g} ({group_sizes[g]} items)" 
                for g in group_options
            ]
            
            # Select group using a selectbox
            logger.info(f"[Group Selection] Available groups: {group_options}")
            selected_index = st.selectbox(
                "Select Group to Analyze",
                range(len(group_options)),
                format_func=lambda x: group_labels[x],
                key=f"group_select_{viz_session_id}"
            )
            
            selected_group = group_options[selected_index]
            logger.info(f"[Group Selection] User selected group: {selected_group}")
            
            # Render the group-specific heatmap with the new function
            render_group_specific_heatmap(df, text_column, viz_session_id, selected_group)


def render_group_specific_heatmap(df, text_column, data_id, selected_group):
    """
    Render a heatmap for a specific group with stable state management.
    
    Args:
        df: The dataframe containing the data
        text_column: The column containing the text to analyze
        data_id: Unique identifier for this dataset
        selected_group: The group ID to visualize
    """
    # Get/initialize state
    state = manage_group_state(data_id, selected_group)
    
    try:
        # Create UI layout
        control_col, info_col = st.columns([1, 5])
        viz_container = st.container()
        
        # Add refresh control
        with control_col:
            if st.button("🔄", key=f"refresh_viz_{selected_group}", help="Refresh visualization"):
                state["needs_refresh"] = True
                if selected_group in state["figures"]:
                    del state["figures"][selected_group]
        
        # Show any cached error message
        with info_col:
            if selected_group in state["errors"]:
                st.error(state["errors"][selected_group])
                if st.button("Retry", key=f"retry_{selected_group}"):
                    del state["errors"][selected_group]
                    state["needs_refresh"] = True
        
        # Main visualization area
        with viz_container:
            if state["needs_refresh"] or selected_group not in state["figures"]:
                with st.spinner(f"Analyzing group {selected_group}..."):
                    try:
                        # Generate new visualization
                        if selected_group in state["errors"]:
                            del state["errors"][selected_group]
                            
                        logger.info(f"[Group Heatmap] Generating for group {selected_group}")
                        fig = generate_group_heatmap(df, text_column, selected_group)
                        
                        # Cache the result
                        state["figures"][selected_group] = fig
                        state["needs_refresh"] = False
                        
                        if fig is None:
                            msg = f"Not enough items in group {selected_group}"
                            logger.info(f"[Group Heatmap] {msg}")
                            st.info(msg)
                            
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"[Group Heatmap] Error: {error_msg}")
                        state["errors"][selected_group] = f"Error: {error_msg}"
                        if selected_group in state["figures"]:
                            del state["figures"][selected_group]
            
            # Show visualization if available
            if selected_group in state["figures"]:
                fig = state["figures"][selected_group]
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[Group Heatmap] {error_msg}")
        st.error(error_msg)



def manage_group_state(data_id, selected_group):
    """Manage the state for group visualization
    
    Args:
        data_id: Unique identifier for this dataset
        selected_group: Currently selected group
        
    Returns:
        dict: Current state for this visualization
    """
    state_key = f"group_state_{data_id}"
    
    # Initialize state if needed
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "current_group": None,
            "figures": {},
            "errors": {},
            "needs_refresh": True
        }
    
    state = st.session_state[state_key]
    
    # Check if group changed
    if state["current_group"] != selected_group:
        state["current_group"] = selected_group
        state["needs_refresh"] = True
        # Keep the figures cache to avoid regeneration
        
    return state


def main():
    st.title("ML Clustering")
    
    with st.expander("ℹ️ About this module", expanded=False):
        # Load documentation from file
        try:
            with open("ML_Clustering_Documentation.md", "r") as f:
                documentation = f.read()
                st.markdown(documentation)
        except FileNotFoundError:
            st.error("Documentation file not found: ML_Clustering_Documentation.md")
            st.markdown("""
            ### Semantic Clustering Approach
            
            This module uses natural language understanding to group similar records based on semantic meaning rather than 
            exact keyword matches. The approach works as follows:
            
            1. **Text to Embeddings**: Text is converted to vector embeddings using a transformer model
            2. **Similarity Calculation**: Cosine similarity measures text similarity (0-1 scale)
            3. **Threshold Clustering**: Records are grouped when similarity exceeds your threshold
            
            Please add an ML_Clustering_Documentation.md file for complete documentation.
            """)
    
    st.markdown("""
    This module groups similar records based on semantic similarity.
    Upload your data, select the column to analyze, and adjust the similarity
    threshold to control how items are grouped together.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(
                f"✅ Successfully loaded file with {df.shape[0]} rows and "
                f"{df.shape[1]} columns"
            )
            
            # Display sample of the data
            with st.expander("Preview of uploaded data", expanded=True):
                st.dataframe(df.head(5))
            
            # Column selection for clustering
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if not text_columns:
                st.error("No text columns found in the uploaded data. Please upload a CSV with text columns.")
                return
            
            # Allow for multi-column selection
            col_selection_type = st.radio(
                "Column selection mode:",
                ["Single Column", "Multiple Columns"],
                help="Choose whether to analyze a single column or combine multiple columns"
            )
            
            if col_selection_type == "Single Column":
                selected_columns = [st.selectbox(
                    "Select column for clustering analysis:", 
                    options=text_columns,
                    index=0
                )]
            else:
                selected_columns = st.multiselect(
                    "Select columns for clustering analysis (will be combined):",
                    options=text_columns,
                    default=[text_columns[0]] if text_columns else []
                )
                
                if not selected_columns:
                    st.warning("Please select at least one column for analysis")
                    return
            
            # Threshold slider
            threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.5, 
                max_value=1.0, 
                value=0.75, 
                step=0.01,
                help=(
                    "Higher values create more groups with higher "
                    "similarity within each group"
                )
            )                # Process button
            if st.button("Process Clustering"):
                with st.spinner("Processing clusters..."):
                    # Process the data - now returns similarity matrix and text mapping too
                    result_df, sim_matrix, text_mapping = process_clustering(
                        df, selected_columns, threshold
                    )
                    
                    # Show message about combined columns if multiple were selected
                    if len(selected_columns) > 1:
                        st.info(
                            f"Multiple columns were combined for analysis: "
                            f"{', '.join(selected_columns)}"
                        )
                    
                    # Create tabs for different sections
                    result_tabs = st.tabs(["Results", "Visualizations", "Export"])
                    
                    with result_tabs[0]:  # Results Tab
                        st.subheader("Clustering Results")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", len(result_df))
                        with col2:
                            st.metric("Number of Groups", result_df["group_id"].nunique())
                        with col3:
                            avg_similarity = round(result_df["relative_similarity"].mean(), 2)
                            st.metric("Average Similarity", f"{avg_similarity}%")
                        
                        # Display the results with group info
                        st.markdown("### Clustered Data")
                        
                        # Format the columns for display
                        display_df = result_df.copy()
                        
                        # Format similarity badges
                        display_df["relative_similarity"] = display_df["relative_similarity"].apply(
                            render_similarity_badge
                        )
                        
                        # Add group size column
                        group_sizes = result_df["group_id"].value_counts().to_dict()
                        display_df["group_size"] = display_df["group_id"].map(group_sizes)
                        
                        # Reorder columns for better display
                        key_cols = [
                            "group_id", "group_size", "group_percentage", "relative_similarity"
                        ]
                        other_cols = [
                            col for col in display_df.columns 
                            if col not in key_cols
                        ]
                        display_cols = key_cols + other_cols
                        display_df = display_df[display_cols]
                        
                        # Display interactive table with highlighted groups
                        st.dataframe(
                            display_df.style.apply(
                                lambda x: ["background-color: #f0f8ff" if i % 2 == 0 else "" 
                                         for i in range(len(x))], 
                                axis=0
                            ),
                            use_container_width=True,
                            height=500
                        )
                        
                        # Group Summary
                        st.markdown("### Group Summary")
                        # Use the first column for counting if multiple were selected
                        count_column = selected_columns[0]
                        group_summary = result_df.groupby("group_id").agg(
                            count=pd.NamedAgg(column=count_column, aggfunc="count"),
                            percentage=pd.NamedAgg(column="group_percentage", aggfunc="first"),
                            avg_similarity=pd.NamedAgg(column="relative_similarity", aggfunc="mean")
                        ).sort_values(by="count", ascending=False).reset_index()
                        
                        group_summary["avg_similarity"] = group_summary["avg_similarity"].round(2).apply(
                            lambda x: render_similarity_badge(x)
                        )
                        
                        # Add sample values from each group
                        def get_sample_values(group_id):
                            # Use the analysis column from the result dataframe
                            analysis_col = 'combined_text' if 'combined_text' in result_df.columns else selected_columns[0]
                            values = result_df[result_df["group_id"] == group_id][analysis_col].unique()
                            return ", ".join(values[:3]) + ("..." if len(values) > 3 else "")
                        
                        group_summary["sample_values"] = group_summary["group_id"].apply(get_sample_values)
                        
                        st.dataframe(
                            group_summary.rename(columns={
                                "count": "Group Size",
                                "percentage": "% of Total",
                                "avg_similarity": "Avg Similarity",
                                "sample_values": "Sample Values"
                            }),
                            use_container_width=True,
                            height=400
                        )
                    
                    with result_tabs[1]:  # Visualizations Tab
                        # Store all necessary visualization data in session state
                        # Use a unique hash key for this particular dataset and analysis
                        data_hash = str(hash(tuple([str(df.shape), str(threshold), str(selected_columns)])))
                        viz_data_key = f"clustering_results_{data_hash}"
                        
                        # Always update the session state with the latest results
                        analysis_col = 'combined_text' if 'combined_text' in result_df.columns else selected_columns[0]
                        st.session_state[viz_data_key] = {
                            'result_df': result_df,
                            'sim_matrix': sim_matrix,
                            'text_mapping': text_mapping,
                            'analysis_col': analysis_col,
                            'has_results': True,
                            'selected_columns': selected_columns,
                            'threshold': threshold,
                            'data_hash': data_hash
                        }
                        
                        # We'll also store the current key to track which result we're viewing
                        st.session_state['current_viz_data_key'] = viz_data_key
                        
                        # Render all the visualizations
                        render_cluster_visualizations(
                            result_df, analysis_col, sim_matrix, text_mapping
                        )
                    
                    with result_tabs[2]:  # Export Tab
                        # Export options
                        st.markdown("### Export Results")
                        csv_export = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv_export,
                            file_name="clustering_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Example data when no file is uploaded
        st.markdown("""
        ### Upload a CSV file to begin
        
        Your file should contain at least one text column that you want to cluster.
        
        #### Example format:
        | ID | Description | Category | Value |
        |----|-------------|----------|-------|
        | 1  | KYC Check   | Finance  | 1000  |
        | 2  | Know Your Customer | Finance | 1000 |
        | 3  | AML Check   | Security | 2000  |
        | 4  | Transaction Monitoring | Security | 3000 |
        
        The algorithm will group similar items based on the text column you select.
        """)
        
        # Option to generate example data
        if st.button("Generate Example Data"):
            with st.spinner("Generating example data..."):
                # Create example data similar to what was provided in the request
                example_data = [
                    {"ID": 1, "Control Type": "KYC Check", "Account": 101, "Branch": "A", "Amount": 1000},
                    {"ID": 2, "Control Type": "Know Your Customer", "Account": 101, "Branch": "A", "Amount": 1000},
                    {"ID": 3, "Control Type": "Customer KYC Verification", "Account": 101, "Branch": "A", "Amount": 1000},
                    {"ID": 4, "Control Type": "AML Check", "Account": 102, "Branch": "B", "Amount": 2000},
                    {"ID": 5, "Control Type": "Anti Money Laundering", "Account": 102, "Branch": "B", "Amount": 2000},
                    {"ID": 6, "Control Type": "AML Policy Review", "Account": 102, "Branch": "B", "Amount": 2000},
                    {"ID": 7, "Control Type": "Risk Review", "Account": 103, "Branch": "C", "Amount": 1500},
                    {"ID": 8, "Control Type": "Risk Assessment", "Account": 103, "Branch": "C", "Amount": 1500},
                    {"ID": 9, "Control Type": "Review of Risks", "Account": 103, "Branch": "C", "Amount": 1500},
                    {"ID": 10, "Control Type": "Transaction Monitoring", "Account": 104, "Branch": "D", "Amount": 3000},
                    {"ID": 11, "Control Type": "Txn Monitoring", "Account": 104, "Branch": "D", "Amount": 3000},
                    {"ID": 12, "Control Type": "Transaction Surveillance", "Account": 104, "Branch": "D", "Amount": 3000},
                    {"ID": 13, "Control Type": "Audit Trail Review", "Account": 105, "Branch": "E", "Amount": 1200},
                    {"ID": 14, "Control Type": "Audit Trail Examination", "Account": 105, "Branch": "E", "Amount": 1200},
                    {"ID": 15, "Control Type": "KYC Documentation", "Account": 101, "Branch": "A", "Amount": 1000},
                    {"ID": 16, "Control Type": "Money Laundering Check", "Account": 102, "Branch": "B", "Amount": 2000},
                    {"ID": 17, "Control Type": "Risk Check", "Account": 103, "Branch": "C", "Amount": 1500},
                    {"ID": 18, "Control Type": "Audit Inspection", "Account": 105, "Branch": "E", "Amount": 1200},
                    {"ID": 19, "Control Type": "Customer Verification KYC", "Account": 101, "Branch": "A", "Amount": 1000},
                    {"ID": 20, "Control Type": "Transaction Review", "Account": 104, "Branch": "D", "Amount": 3000}
                ]
                example_df = pd.DataFrame(example_data)
                
                # Display preview and set session state
                st.success("Example data generated successfully!")
                st.session_state['example_df'] = example_df
                
                # Use the example dataframe instead of requiring an upload
                df = example_df
                st.dataframe(df.head())
                
                # Column selection for clustering
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                
                # Allow for multi-column selection
                col_selection_type = st.radio(
                    "Column selection mode:",
                    ["Single Column", "Multiple Columns"],
                    help="Choose whether to analyze a single column or combine multiple columns"
                )
                
                if col_selection_type == "Single Column":
                    # Default to "Control Type" if available
                    default_idx = text_columns.index("Control Type") if "Control Type" in text_columns else 0
                    selected_columns = [st.selectbox(
                        "Select column for clustering analysis:", 
                        options=text_columns,
                        index=default_idx
                    )]
                else:
                    selected_columns = st.multiselect(
                        "Select columns for clustering analysis (will be combined):",
                        options=text_columns,
                        default=["Control Type"] if "Control Type" in text_columns else [text_columns[0]]
                    )
                    
                    if not selected_columns:
                        st.warning("Please select at least one column for analysis")
                        return
                
                # Threshold slider
                threshold = st.slider(
                    "Similarity Threshold", 
                    min_value=0.5, 
                    max_value=1.0, 
                    value=0.75, 
                    step=0.01,
                    help="Higher values create more groups with higher similarity within each group"
                )
                
                # Add process button for the example data
                if st.button("Process Example Data"):
                    with st.spinner("Processing example data..."):
                        # Process the data
                        result_df, sim_matrix, text_mapping = process_clustering(
                            df, selected_columns, threshold
                        )
                        
                        # Show message about combined columns if multiple were selected
                        if len(selected_columns) > 1:
                            st.info(f"Multiple columns were combined for analysis: {', '.join(selected_columns)}")
                        
                        # Create tabs for different sections
                        result_tabs = st.tabs(["Results", "Visualizations", "Export"])
                        
                        # Display results (rest of the code is identical to the regular process flow)
                        with result_tabs[0]:  # Results Tab
                            st.subheader("Clustering Results")
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Records", len(result_df))
                            with col2:
                                st.metric("Number of Groups", result_df["group_id"].nunique())
                            with col3:
                                avg_similarity = round(result_df["relative_similarity"].mean(), 2)
                                st.metric("Average Similarity", f"{avg_similarity}%")
                            
                            # Display the results with group info
                            st.markdown("### Clustered Data")
                            
                            # Format the columns for display
                            display_df = result_df.copy()
                            
                            # Format similarity badges
                            display_df["relative_similarity"] = display_df["relative_similarity"].apply(
                                render_similarity_badge
                            )
                            
                            # Add group size column
                            group_sizes = result_df["group_id"].value_counts().to_dict()
                            display_df["group_size"] = display_df["group_id"].map(group_sizes)
                            
                            # Reorder columns for better display
                            key_cols = [
                                "group_id", "group_size", "group_percentage", "relative_similarity"
                            ]
                            other_cols = [
                                col for col in display_df.columns 
                                if col not in key_cols
                            ]
                            display_cols = key_cols + other_cols
                            display_df = display_df[display_cols]
                            
                            # Display interactive table with HTML content
                            st.markdown("""
                            <style>
                            .similarity-badge {
                                display: inline-block;
                                padding: 3px 8px;
                                border-radius: 10px;
                                color: white;
                                font-weight: bold;
                                text-align: center;
                                min-width: 60px;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            st.dataframe(display_df, use_container_width=True, height=500)
                        
                        with result_tabs[1]:  # Visualizations Tab
                            # Store all necessary visualization data in session state
                            # Use a unique hash key for this particular dataset and analysis
                            data_hash = str(hash(tuple([
                                str(df.shape), str(threshold), str(selected_columns)
                            ])))
                            viz_data_key = f"example_clustering_{data_hash}"
                            
                            # Always update the session state with the latest results
                            analysis_col = 'combined_text' if 'combined_text' in result_df.columns else selected_columns[0]
                            st.session_state[viz_data_key] = {
                                'result_df': result_df,
                                'sim_matrix': sim_matrix,
                                'text_mapping': text_mapping,
                                'analysis_col': analysis_col,
                                'has_results': True,
                                'selected_columns': selected_columns,
                                'threshold': threshold,
                                'data_hash': data_hash
                            }
                            
                            # We'll also store the current key to track which result we're viewing
                            st.session_state['current_viz_data_key'] = viz_data_key
                            
                            # Render all the visualizations
                            render_cluster_visualizations(
                                result_df, analysis_col, sim_matrix, text_mapping
                            )
                        
                        with result_tabs[2]:  # Export Tab
                            # Export options
                            st.markdown("### Export Results")
                            csv_export = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv_export,
                                file_name="example_clustering_results.csv",
                                mime="text/csv"
                            )

if __name__ == "__main__":
    main()
