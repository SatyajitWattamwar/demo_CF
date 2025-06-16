
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.title("Closest Friend (CF) Stability Assessment Tool")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your historical formulation dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df)

    # Select target row
    target_index = st.number_input("Enter the index of the target formulation (row number)", min_value=0, max_value=len(df)-1, step=1)

    # Select features for similarity
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect("Select numeric features for similarity comparison", numeric_columns)

    if selected_features and target_index is not None:
        # Drop rows with missing or infinite values in selected features
        filtered_data = df[selected_features].replace([float('inf'), float('-inf')], pd.NA).dropna()

        # Keep track of the original indices
        valid_indices = filtered_data.index.tolist()

        if target_index not in valid_indices:
            st.error("The selected target row contains missing or infinite values. Please choose another row.")
        else:
            # Standardize and compute similarity only on valid rows
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(filtered_data)
            similarity_matrix = cosine_similarity(scaled_data)

            # Map back to original DataFrame
            target_position = valid_indices.index(target_index)
            similarity_scores = similarity_matrix[target_position]

            # Build results DataFrame
            results_df = df.loc[valid_indices].copy()
            results_df["Similarity Score"] = similarity_scores
            results_df = results_df.drop(index=target_index)
            results_df = results_df.sort_values(by="Similarity Score", ascending=False)

            st.subheader("Top Similar Formulations")
            st.dataframe(results_df)

            # Download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "similarity_results.csv", "text/csv")
