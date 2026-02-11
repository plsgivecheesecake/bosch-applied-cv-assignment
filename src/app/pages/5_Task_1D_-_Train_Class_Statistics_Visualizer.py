"""
Class visualizer page
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import seaborn as sns
from config import CSV_DIR
import matplotlib.pyplot as plt


def populate_class_visualizer_page():
    """
    Populates the class distribution analysis page
    """
    st.set_page_config(
        page_title="Training Data Visualizer",
        page_icon=":material/sentiment_excited:",
        layout="wide",
    )
    st.write(
        """
            # Task 1D: Interactive Training Data Visualizer

            This section contains a visualizer that lets you filter the train dataset using UI controls
        """
    )
    csv_files = sorted(Path(CSV_DIR).glob("*.csv"))

    if not csv_files:
        st.warning(
            'Please use the sidebar navigation to navigate to the page called "Process Dataset" and process the dataset first. Thank you.'
        )
    else:
        df = load_data()
        st.sidebar.header("Filter Statistics")

        selected_cats = st.sidebar.multiselect(
            "Select Categories",
            options=list(CATEGORIES_MAP.values()),
            default=["traffic light", "car", "person"],  # Default focus
        )

        # 2. Size Group Filter
        selected_sizes = st.sidebar.multiselect(
            "Select COCO Size Groups",
            options=["Small", "Medium", "Large"],
            default=["Small", "Medium", "Large"],
        )

        # 3. Boolean Filters
        filter_occluded = st.sidebar.radio(
            "Occlusion Filter", ["All", "Only Occluded", "Only Non-Occluded"]
        )
        filter_truncated = st.sidebar.radio(
            "Truncation Filter", ["All", "Only Truncated", "Only Non-Truncated"]
        )

        mask = (df["category_name"].isin(selected_cats)) & (
            df["size_name"].isin(selected_sizes)
        )

        if filter_occluded == "Only Occluded":
            mask &= df["occluded"]
        elif filter_occluded == "Only Non-Occluded":
            mask &= not df["occluded"]

        if filter_truncated == "Only Truncated":
            mask &= df["truncated"]
        elif filter_truncated == "Only Non-Truncated":
            mask &= not df["truncated"]

        filtered_df = df[mask]

        if filtered_df.empty:
            st.warning("No data found for the selected filters.")
        else:
            st.subheader(f"Area Distribution for {', '.join(selected_cats)}")

            # Create static plot
            fig, ax = plt.subplots(figsize=(12, 6))

            sns.boxplot(
                data=filtered_df,
                x="category_name",
                y="area",
                hue="size_name",  # Color-code by S/M/L
                hue_order=["Small", "Medium", "Large"],
                ax=ax,
                showfliers=False,  # ESSENTIAL: Keeps memory usage low for 160k+ rows
            )

            num_cats = len(filtered_df["category_name"].unique())

            # Draw a vertical line at every 'half' position between categories
            for i in range(num_cats - 1):
                ax.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.3, lw=1)

            ax.set_yscale("log")
            ax.set_title("Bounding Box Area by Category (S/M/L Groups Separated)")
            ax.set_ylabel("Area (Log Scale)")
            ax.set_xlabel("Object Category")

            # Move legend to the outside so it doesn't overlap data
            ax.legend(title="Size Group", bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.xticks(rotation=45)
            plt.tight_layout()  # Ensures labels aren't cut off

            st.pyplot(fig)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Instances", f"{len(filtered_df):,}")
            col2.metric("Median Area", f"{int(filtered_df['area'].median())} px")
            col3.metric(
                "Occlusion Rate", f"{(filtered_df['occluded'].mean() * 100):.1f}%"
            )
            col4.metric(
                "Truncated Rate", f"{(filtered_df['truncated'].mean() * 100):.1f}%"
            )

            st.write("""
            This is a cool interactive visualizer that I developed and wanted to include in the UI.
            It basically confirms that the dataset is full of small and medium sized images.""")


CATEGORIES_MAP = {
    0: "traffic sign",
    1: "traffic light",
    2: "car",
    3: "rider",
    4: "motor",
    5: "person",
    6: "bus",
    7: "truck",
    8: "bike",
    9: "train",
}
SIZE_NAME_MAP = {0: "Small", 1: "Medium", 2: "Large"}


@st.cache_data
def load_data():
    """
    Reads csv file from path and caches it
    """
    # Update this path to your actual filename in Docker
    records_path = Path(CSV_DIR) / "records_train.csv"
    df = pd.read_csv(records_path)

    # Map the IDs to names for the UI and Plot labels
    df["category_name"] = df["category"].map(CATEGORIES_MAP)
    df["size_name"] = df["size_group"].map(SIZE_NAME_MAP)
    return df


populate_class_visualizer_page()
