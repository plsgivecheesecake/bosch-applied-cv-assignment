"""
Third page of the app showcasing scene statistics
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import CSV_DIR


def populate_scene_statistics_page():
    """
    Populates the "Scene Statistics" page
    """
    st.set_page_config(
        page_title="Scene Statistics",
        page_icon=":material/sentiment_excited:",
        layout="wide",
    )
    st.write(
        """
        # Task 1A: Scene Level Statistical Analysis

        In this page, we explore the time of day and weather conditions across both training and validation splits.
        Examining how images are distributed across different combinations of these two scene parameters will help
        us gain insight into the potential domain biases, and conditions under which a model trained on this
        dataset is expected to generalize. 

        This analysis helps us anticipate performance variations across different lighting and weather scenarios
        and assess whether the model is likely to encounter distribution shifts during evaluation.
        """
    )
    csv_files = sorted(Path(CSV_DIR).glob("*.csv"))

    if not csv_files:
        st.warning(
            'Please use the sidebar navigation to navigate to the page called "Process Dataset" and process the dataset first. Thank you.'
        )
    else:
        st.write(
            """
            Sorry about the long heatmaps below, but I feel that they demonstrate that both dataset splits have the same peaks for weather,
            time of day, and object category combinations. 
            """
        )

        st.write("First, we look at how our dataset is distributed by weather type")
        display_barcharts("weather")
        st.write(
            "We see that most images in our dataset are in the clear weather category"
        )
        st.write(
            "Then we take a look at how the datset is distributed according to scene type"
        )
        display_barcharts("scene")
        st.write(
            'This reveals that most images are from the "City street" scene, meaning that we can expect a wide variety of objects in large numbers throughout the dataset'
        )

        st.write(
            "Finally, we take a look at how the dataset is split according to time of day"
        )
        display_barcharts("timeofday")
        st.write(
            """
        The dataset is primarily bifurcated between **Daytime** and **Nighttime** captures, with only a small representation of **Dawn/Dusk** pictures.
        While this provides a diverse operational range, the high volume of nighttime imagery introduces specific computer vision challenges, like visibility challenges, where low ambient light reduces the signal-to-noise ratio, making it difficult to discern object boundaries in unlit areas.
        Frequent overexposure from car headlights, taillights, and high-intensity street lamps can "wash out" bounding boxes or create artifacts , causing optical interference that might lead to false positives.
            """
        )
        st.write(
            """
            Sorry about the long heatmaps below, but I feel that they demonstrate that both dataset splits have the same peaks for weather,
            time of day, and object category combinations.
            """
        )
        display_heatmaps()


def display_heatmaps() -> None:
    """
    Creates (weather, timeofday) vs category heatmaps for both splits
    """
    train_df = load_data("train")
    val_df = load_data("val")

    train_pivot_df = build_matrix(train_df)
    val_pivot_df = build_matrix(val_df)

    fig_train = plot_heatmap(train_pivot_df)
    fig_val = plot_heatmap(val_pivot_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Split Heatmap")
        st.pyplot(fig_train)
    with col2:
        st.subheader("Validation Split Heatmap")
        st.pyplot(fig_val)


def display_barcharts(attribute_name: str) -> None:
    """
    Displays the attribute specific bar-chart distributions for both splits

    :param attribute_name: Scene attribute for fetching computed CSV file
    :type attribute_name: str
    """
    df_train = pd.read_csv(Path(CSV_DIR) / f"{attribute_name}_train.csv")
    df_val = pd.read_csv(Path(CSV_DIR) / f"{attribute_name}_val.csv")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Split")
        st.bar_chart(data=df_train, x="attribute", y="count")
    with col2:
        st.subheader("Validation Split")
        st.bar_chart(data=df_val, x="attribute", y="count")


def load_data(split: str) -> None:
    """
    Loads csv for categories by scene parameters for given split

    :param split: Name of split, i.e. train or test
    :type split: str
    """
    csv_path = Path(CSV_DIR) / f"categories_by_scene_params_{split}.csv"
    return pd.read_csv(csv_path)


def build_matrix(df: pd.DataFrame) -> None:
    """
    Builds pivot table for a given dataframe for heatmap generation

    :param df: Dataframe for which pivot table should be built
    :type df: pd.DataFrame
    """
    pivot = df.pivot_table(
        index=["weather", "time"], columns="class", values="value", fill_value=0
    )

    return pivot


def plot_heatmap(pivot_df: pd.DataFrame) -> None:
    """
    Plots the heatmap for the scene statistics dataframe onto the UI

    :param pivot_df: Pivot table for scene statistics dataframe
    :type pivot_df: pd.DataFrame
    """
    matrix = pivot_df.values
    fig, ax = plt.subplots(figsize=(8, 12))

    im = ax.imshow(matrix, aspect="auto", cmap="inferno")

    threshold = matrix.max() / 2.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]

            text_color = "white" if val < threshold else "black"

            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                weight="bold",
            )

    fig.colorbar(im, ax=ax, label="Count")

    rows = [f"{w} & {t}" for w, t in pivot_df.index]
    cols = pivot_df.columns
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")

    plt.tight_layout()
    return fig


populate_scene_statistics_page()
