import streamlit as st
from dataset_analyzer.dataset_analyzer import DatasetAnalyzer

st.write(
    """
        # Run The Analysis Yourself!

        On this page, you can run the data analysis yourself and view random examples.
    """
)
dataset_analyzer = DatasetAnalyzer()
# Start by reading the labels
for split in ["train", "val"]:
    progress_bar = st.progress(0)
    status_text = st.empty()
    result = dataset_analyzer.read_labels(split, progress_bar, status_text)
    st.write(result)
