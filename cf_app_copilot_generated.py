
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Closest Friend (CF) Stability Tool", layout="wide")

st.title("🧪 Closest Friend (CF) Stability Assessment Tool")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your historical formulation data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Show preview
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Select target row
    st.subheader("🎯 Select Target Formulation")
    target_index = st.number_input("Enter the index of the target formulation (0-based)", min_value=0, max_value=len(df)-1, value=0)

    # Select numeric columns for comparison
    st.subheader("📊 Select Features for Similarity Comparison")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect("Choose numeric columns to compare", numeric_cols, default=numeric_cols)

    if selected_features:
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[selected_features])

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(scaled_data)
        similarity_scores = similarity_matrix[target_index]

        # Create results DataFrame
        results_df = df.copy()
        results_df["Similarity Score"] = similarity_scores
        results_df = results_df.drop(index=target_index)  # exclude self
        results_df = results_df.sort_values(by="Similarity Score", ascending=False)

        # Show results
        st.subheader("🔍 Closest Formulations")
        st.dataframe(results_df)

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results as CSV", data=csv, file_name="closest_friend_results.csv", mime="text/csv")
    else:
        st.warning("Please select at least one numeric column for comparison.")
else:
    st.info("Please upload a CSV file to begin.")
