"""
Fourth page of the app showcasing class statistics. This page will show the most important content.
"""

import streamlit as st
from config import CSV_DIR
import pandas as pd
import plotly.express as px
from pathlib import Path


def populate_class_distribution_analysis_page():
    """
    Populates the class distribution analysis page
    """
    st.set_page_config(
        page_title="Class Statistics",
        page_icon=":material/sentiment_excited:",
        layout="wide",
    )
    st.write(
        """
            # Task 1B: Class Distribution Analysis

            This section evaluates the structural characteristics of our dataset across the **Train** and **Val** splits. 
            By analyzing object scale and visibility metadata, we can anticipate potential bottlenecks in model 
            convergence and detection accuracy.
        """
    )
    csv_files = sorted(Path(CSV_DIR).glob("*.csv"))

    if not csv_files:
        st.warning(
            'Please use the sidebar navigation to navigate to the page called "Process Dataset" and process the dataset first. Thank you.'
        )
    else:
        train_df = get_adjusted_df("train")
        val_df = get_adjusted_df("val")

        st.write("""
        One can immediately notice how cars absolutely dominate the entire dataset, followed by traffic signs and traffic lights,
         which are both less than half the number of cars. Trucks and buses have the highest percentage of large and medium objects in their respective categories.
         The smallest number of instances are in the \"train\" category. All visualizations reveal that categories, scene statistics, and even
         anomalies are distributed similarly across both splits.""")
        plot_anomaly_counts(train_df, val_df)
        st.write(
            """
            We categorize objects into **Small** ($< 32^2$ px), **Medium**, and **Large** ($> 96^2$ px) tiers. A balanced distribution 
            is rare in autonomous driving; typically, "Small" objects dominate. If the ratio is too skewed, one may 
            need to implement **Feature Pyramid Networks (FPN)** or specialized tiling strategies. This is also the case in this dataset,
            as small and medium sized objects dominate the dataset. Considering that the images are rather large (1280 x 720), the
            heavy distribution of such small objects across the training set can cause a hindrance to the model for training,
            and also prevent the model from being accurately able to predict the correct class during testing.
            """
        )
        plot_size_distribution(train_df, val_df)
        st.write("""
        Occlusion & Truncation define the "visibility" of our targets. Occlusion measures how much of an object is hidden behind 
        another object, while truncation measures how much of an object is "cut off" by the edge of the image frame. In both cases, this information is
        only available as a boolean flag, and the degree of occlusion or truncation is not available. However, in most classes, a significant number of
        samples are occluded, while the truncation percentage of samples is not that high. Two classes, namely traffic signs and traffic lights seem
        to be the only ones with a low percentage of truncated and occluded samples. This implies that a model trained on this dataset has the potential
        to be good at identifying traffic lights and traffic signs, due to the high number of supposedly good samples. 
        """)
        plot_occlusion_percentage(train_df, val_df)
        plot_truncation_percentage(train_df, val_df)


@st.cache_data
def load_data(split: str) -> pd.DataFrame:
    """
    Loads and caches a csv file as a datframe

    :param split: Name of the dataset split, i.e., train or test
    :type split: str
    :return: A pandas dataframe of the category stands
    :rtype: DataFrame
    """
    file_path = Path(CSV_DIR) / f"category_stats_{split}.csv"
    df = pd.read_csv(file_path)
    return df


def get_adjusted_df(split: str) -> pd.DataFrame:
    """
    Returns a filtered and sorted df for visualization

    :param split: The name of the split, i.e., train or val
    :type split: str
    :return: Returns a pandas datframe to be used by the plotters
    :rtype: DataFrame
    """
    df = load_data(split)
    df["non_anomalies"] = df["total_count"] - df["anomalies"]
    df["occluded_pct"] = (df["occluded"] / df["total_count"]) * 100
    df["non_occluded_pct"] = 100 - df["occluded_pct"]
    df["truncated_pct"] = (df["truncated"] / df["total_count"]) * 100
    df["non_truncated_pct"] = 100 - df["truncated_pct"]

    # Sort values for long tail
    df = df.sort_values(by="total_count", ascending=True)
    return df


def plot_anomaly_counts(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """
    Plotting function for anomaly counts

    :param train_df: Adjusted and sorted dataframe
    :type train_df: pd.DataFrame
    :param val_df: Adjusted and sorted dataframe
    :type val_df: pd.DataFrame
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train - Total Count & Anomalies")
        fig1 = px.bar(
            train_df,
            y="class",
            x=["anomalies", "non_anomalies"],
            orientation="h",
            title="Total Count (Anomalies vs Non-Anomalies)",
            labels={"value": "Count", "class": "Category"},
            color_discrete_map={
                "anomalies": "#FF4B4B",
                "non_anomalies": "#E0E0E0",
            },  # Red & Grey
        )
        fig1.update_layout(
            barmode="stack",
            xaxis_title="Count",
            yaxis_title=None,
            legend_title_text="Type",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Val - Total Count & Anomalies")
        fig1 = px.bar(
            val_df,
            y="class",
            x=["anomalies", "non_anomalies"],
            orientation="h",
            title="Total Count (Anomalies vs Non-Anomalies)",
            labels={"value": "Count", "class": "Category"},
            color_discrete_map={
                "anomalies": "#FF4B4B",
                "normal": "#E0E0E0",
            },  # Red & Grey
        )
        fig1.update_layout(
            barmode="stack",
            xaxis_title="Count",
            yaxis_title=None,
            legend_title_text="Type",
        )
        st.plotly_chart(fig1, use_container_width=True)


def plot_size_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """
    Plotting function for size distribution

    :param train_df: Adjusted and sorted dataframe
    :type train_df: pd.DataFrame
    :param val_df: Adjusted and sorted dataframe
    :type val_df: pd.DataFrame
    """
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train - Size Distribution")
        fig2 = px.bar(
            train_df,
            y="class",
            x=["small", "medium", "large"],
            orientation="h",
            title="Size Breakdown (Small, Medium, Large)",
            labels={"value": "Count", "class": "Category"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_layout(
            barmode="stack",
            xaxis_title="Count",
            yaxis_title=None,
            legend_title_text="Size",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Val - Size Distribution")
        fig2 = px.bar(
            val_df,
            y="class",
            x=["small", "medium", "large"],
            orientation="h",
            title="Size Breakdown (Small, Medium, Large)",
            labels={"value": "Count", "class": "Category"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_layout(
            barmode="stack",
            xaxis_title="Count",
            yaxis_title=None,
            legend_title_text="Size",
        )
        st.plotly_chart(fig2, use_container_width=True)


def plot_occlusion_percentage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """
    Plotting function for occlusion percentage

    :param train_df: Adjusted and sorted dataframe
    :type train_df: pd.DataFrame
    :param val_df: Adjusted and sorted dataframe
    :type val_df: pd.DataFrame
    """
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train - Occlusion Percentage")
        fig3 = px.bar(
            train_df,
            y="class",
            x=["occluded_pct", "non_occluded_pct"],
            orientation="h",
            title="Occlusion %",
            labels={"value": "Percentage (%)", "class": "Category"},
            color_discrete_map={
                "occluded_pct": "#FFA15A",
                "non_occluded_pct": "#F0F2F6",
            },
        )
        fig3.update_layout(
            barmode="stack",
            xaxis_title="Percentage (%)",
            yaxis_title=None,
            legend_title_text="Status",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Val - Occlusion Percentage")
        fig3 = px.bar(
            val_df,
            y="class",
            x=["occluded_pct", "non_occluded_pct"],
            orientation="h",
            title="Occlusion %",
            labels={"value": "Percentage (%)", "class": "Category"},
            color_discrete_map={
                "occluded_pct": "#FFA15A",
                "non_occluded_pct": "#F0F2F6",
            },
        )
        fig3.update_layout(
            barmode="stack",
            xaxis_title="Percentage (%)",
            yaxis_title=None,
            legend_title_text="Status",
        )
        st.plotly_chart(fig3, use_container_width=True)


def plot_truncation_percentage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """
    Plotting function for truncation percentage

    :param train_df: Adjusted and sorted dataframe
    :type train_df: pd.DataFrame
    :param val_df: Adjusted and sorted dataframe
    :type val_df: pd.DataFrame
    """
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train - Truncation Percentage")
        fig4 = px.bar(
            train_df,
            y="class",
            x=["truncated_pct", "non_truncated_pct"],
            orientation="h",
            title="Truncation %",
            labels={"value": "Percentage (%)", "class": "Category"},
            color_discrete_map={"truncated %": "#AA0DFE", "non_truncated %": "#F0F2F6"},
        )
        fig4.update_layout(
            barmode="stack",
            xaxis_title="Percentage (%)",
            yaxis_title=None,
            legend_title_text="Status",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.subheader("Val - Truncation Percentage")
        fig4 = px.bar(
            val_df,
            y="class",
            x=["truncated_pct", "non_truncated_pct"],
            orientation="h",
            title="Truncation %",
            labels={"value": "Percentage (%)", "class": "Category"},
            color_discrete_map={"truncated %": "#AA0DFE", "non_truncated %": "#F0F2F6"},
        )
        fig4.update_layout(
            barmode="stack",
            xaxis_title="Percentage (%)",
            yaxis_title=None,
            legend_title_text="Status",
        )
        st.plotly_chart(fig4, use_container_width=True)


populate_class_distribution_analysis_page()
