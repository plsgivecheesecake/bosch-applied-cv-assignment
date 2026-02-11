""" "
Main entrypoint of streamlit application
"""

import streamlit as st
from analysis.dataset_analyzer import DatasetAnalyzer
from config import CSV_DIR
from pathlib import Path


def process_dataset():
    """
    Sets a session state variable to indicate that processing has begun
    """
    st.session_state.start_processing = True


def main():
    """
    This is the main entrypoint of the streamlit web application, and
    is also responsible for displaying the contents of the home page.
    """
    st.set_page_config(
        page_title="Process Dataset",
        page_icon=":material/sentiment_excited:",
        layout="wide",
    )
    st.write(
        """
            # Hello! :material/sentiment_excited:
            
            Thank you for taking the time to review my submission.
            
            To process the dataset and generate the analysis in the next pages, please press the button below. Please note, that due to the way
            that streamlit manages multipage applications, this page needs to stay open until the processing is complete.
            If you navigate away to another page before it is complete, you will need to restart the processing by clicking the button again.
            Thank you for your understanding!
        """
    )
    if st.button("Process Dataset"):
        process_dataset()

    if st.session_state.get("start_processing"):
        csv_files = sorted(Path(CSV_DIR).glob("*.csv"))
        if not csv_files:
            dataset_analyzer = DatasetAnalyzer()
            # Start by reading the labels
            for split in ["train", "val"]:  # ["train", "val"]:
                progress_bar = st.progress(0)
                status_text = st.empty()
                dataset_analyzer.compute_statistics(split, progress_bar, status_text)

        else:
            st.success(
                "You have already finished processing the dataset. Please navigate to the next pages using the sidebar navigation"
            )
        st.write("Processing Complete!")


if __name__ == "__main__":
    main()
