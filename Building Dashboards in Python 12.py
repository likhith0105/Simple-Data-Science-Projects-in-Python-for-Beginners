# ==============================
# PYTHON DASHBOARD PROJECT
# ==============================

# Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Sales Dashboard")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Show Dataset
    st.subheader("Dataset")
    st.write(df)

    # Show Columns
    st.subheader("Columns")
    st.write(df.columns)

    # Statistics
    st.subheader("Statistics")
    st.write(df.describe())

    # Select Column
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_columns) > 0:

        column = st.selectbox("Select Column for Chart", numeric_columns)

        # Plot Graph
        fig, ax = plt.subplots()

        ax.plot(df[column])

        ax.set_title(f"{column} Chart")

        st.pyplot(fig)

    else:
        st.write("No numeric columns found.")

else:
    st.write("Please upload a CSV file.")