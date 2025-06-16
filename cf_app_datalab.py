
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from io import BytesIO

# Title
st.title("🧪 Data Agnostic Closest Friend (CF) Tool")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Show preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Select target row
    st.subheader("Select Target Row")
    target_index = st.number_input("Enter the index of the target row", min_value=0, max_value=len(df)-1, step=1)

    # Identify column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.subheader("Feature Selection and Weighting")

    selected_numeric = st.multiselect("Select numeric features", numeric_cols, default=numeric_cols)
    selected_categorical = st.multiselect("Select categorical features", categorical_cols, default=categorical_cols)

    weight_numeric = st.slider("Weight for numeric similarity", 0.0, 1.0, 0.5)
    weight_categorical = 1.0 - weight_numeric

    if st.button("Find Closest Friends"):
        df_clean = df.copy()

        # Drop rows with missing values in selected columns
        all_selected = selected_numeric + selected_categorical
        df_clean = df_clean.dropna(subset=all_selected)

        # Recompute target index after dropping rows
        if target_index not in df_clean.index:
            st.error("Target row was removed due to missing values. Please choose another index.")
        else:
            # Numeric similarity
            if selected_numeric:
                scaler = StandardScaler()
                numeric_data = scaler.fit_transform(df_clean[selected_numeric])
                numeric_sim = cosine_similarity(numeric_data)
            else:
                numeric_sim = np.zeros((len(df_clean), len(df_clean)))

            # Categorical similarity
            if selected_categorical:
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                cat_data = encoder.fit_transform(df_clean[selected_categorical])
                cat_sim = cosine_similarity(cat_data)
            else:
                cat_sim = np.zeros((len(df_clean), len(df_clean)))

            # Combined similarity
            combined_sim = weight_numeric * numeric_sim + weight_categorical * cat_sim

            # Get similarity scores for target
            target_pos = df_clean.index.get_loc(target_index)
            similarity_scores = combined_sim[target_pos]

            # Build results
            results_df = df_clean.copy()
            results_df["Similarity Score"] = similarity_scores
            results_df = results_df.drop(index=target_index)
            results_df = results_df.sort_values(by="Similarity Score", ascending=False)

            st.subheader("Closest Matches")
            st.dataframe(results_df)

            # Download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="closest_friends_results.csv", mime="text/csv")
