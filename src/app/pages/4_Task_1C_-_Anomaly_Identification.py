"""
Fifth page of the app showcasing class statistics. This page will show details of anomalies.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from config import CSV_DIR, NETWORK_MOUNT
from pathlib import Path
from PIL import Image, ImageDraw


def populate_anomaly_identification_page():
    """
    Populates the anomaly identification and analysis page
    """
    st.set_page_config(
        page_title="Anomaly Identification",
        page_icon=":material/sentiment_excited:",
        layout="wide",
    )
    st.write(
        """
        ## Task 1C: Anomaly Identification
        
        To ensure dataset quality, we identify **bounding box anomalies** based on the COCO standard. We flag objects that exhibit extreme physical characteristics in two categories: **Scale** and **Proportion**.

        ### 1. Size Anomalies (Area)
        Anomalies are defined by pixel area thresholds. Objects falling outside these bounds are often noise or poorly labeled instances:
        * **Small Objects:** Area < $32^2$ pixels ($1,024 \\text{ px}^2$)
        * **Large Objects:** Area > $1024 \\times 576$ pixels ($589,824 \\text{ px}^2$)

        ### 2. Shape Anomalies (Aspect Ratio)
        The aspect ratio ($AR = \\frac{width}{height}$) helps identify "sliver" boxes. We flag boxes with:
        * **Ultra-Narrow:** $AR < 0.1$ (Height is more than 10x the width)
        * **Ultra-Wide:** $AR > 10.0$ (Width is more than 10x the height)
        
        > **Note:** These thresholds help isolate potential labeling errors or edge cases that might require specialized augmentation during model training.
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
        ### Distribution Summary
        A comparative analysis reveals that anomalies are **distributed in a consistent fashion** across both the training and validation splits, suggesting a stable data acquisition process. 
        
        Notably, there is a strong correlation between class frequency and anomaly counts: our most prevalent classes—**Cars, Traffic Signs, and Traffic Lights**—account for the highest volume of anomalies. This is statistically expected, as the higher sample size for these categories naturally increases the occurrence of edge cases such as heavily occluded vehicles or distant, sub-threshold traffic lights.
        """
        )
        st.subheader("Visualizing anomalies in the train split")
        render_bar_chart_and_top_images("train")
        st.subheader("Visualizing anomalies in the val split")
        render_bar_chart_and_top_images("val")


@st.cache_data
def load_data(split: str) -> pd.DataFrame:
    """
    Loads a dataframe from a csv file and caches it

    :param split: Denotes the split of the dataset, i.e., train or val
    :type split: str
    :return: Returns a pandas dataframe for the csv file
    :rtype: DataFrame
    """
    # Loading the CSV saved from your specific DataFrame
    df = pd.read_csv(Path(CSV_DIR) / f"anomalies_{split}.csv")
    # Map 'type' 0 and 1 to readable labels
    type_map = {0: "Area Anomaly", 1: "Aspect Ratio Anomaly"}
    df["anomaly_name"] = df["type"].map(type_map)
    return df


def render_bar_chart_and_top_images(split: str) -> None:
    """
    Function that renders bar chart and images

    :param split: The name of the dataset split, i.e., train or val
    :type split: str
    """
    try:
        df = load_data(split)
        top_aspect_images = (
            df[df["type"] == 1]["image_name"].value_counts().head(3).index.tolist()
        )
        # Type 0: Area Anomalies
        top_area_images = (
            df[df["type"] == 0]["image_name"].value_counts().head(3).index.tolist()
        )

        col1, col2 = st.columns(2)
        col1.metric("Total Anomalies", len(df))
        col2.metric("Split", f"{split}")

        # Create the Chart
        # We aggregate by category and the mapped anomaly_name
        chart_df = (
            df.groupby(["category", "anomaly_name"]).size().reset_index(name="count")
        )

        fig = px.bar(
            chart_df,
            x="category",
            y="count",
            color="anomaly_name",
            title="Anomaly Counts per Category",
            labels={
                "category": "Category",
                "count": "Number of Anomalies",
                "anomaly_name": "Anomaly Type",
            },
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader(
            f"Top 3 Images with the most aspect ratio anomalies (in red) - {split}"
        )
        cols1 = st.columns(3)
        for i, img_name in enumerate(top_aspect_images):
            annotated = get_annotated_image(img_name, df)
            with cols1[i]:
                if annotated:
                    st.image(
                        annotated,
                        caption=f"Image: /{split}/{img_name}",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"File {img_name} not found in path.")

        st.subheader(
            f"Top 3 Images with the most BBOX area anomalies (in blue) - {split}"
        )
        cols2 = st.columns(3)
        for i, img_name in enumerate(top_area_images):
            annotated = get_annotated_image(img_name, df)
            with cols2[i]:
                if annotated:
                    st.image(
                        annotated,
                        caption=f"Image: /{split}/{img_name}",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"File {img_name} not found in path.")

    except FileNotFoundError:
        st.error(
            "File 'anomalies.csv' not found. Please ensure you saved the DataFrame correctly."
        )


def get_annotated_image(img_name: str, dataframe: pd.DataFrame):
    """
    Gets an annotated image to render in the UI

    :param img_name: Name of the image
    :type img_name: str
    :param dataframe: Search df for image details
    :type dataframe: pd.DataFrame
    """
    row_info = dataframe[dataframe["image_name"] == img_name].iloc[0]
    # Ok haha I guess I don't need to pass split as param
    split = row_info["split"]
    img_path = (
        Path(NETWORK_MOUNT)
        / "bdd100k_images_100k"
        / "bdd100k"
        / "images"
        / "100k"
        / split
        / img_name
    )
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_annos = dataframe[dataframe["image_name"] == img_name]

        for _, row in img_annos.iterrows():
            # Ensure coordinates are numeric and ordered correctly
            try:
                # Pillow needs [x0, y0, x1, y1] where x0 < x1 and y0 < y1
                coords = [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ]
                # Re-order just in case they are flipped
                shape = [
                    min(coords[0], coords[2]),
                    min(coords[1], coords[3]),
                    max(coords[0], coords[2]),
                    max(coords[1], coords[3]),
                ]

                color = "#FF0000" if row["type"] == 1 else "#0000FF"
                draw.rectangle(shape, outline=color, width=4)
                draw.text((shape[0], shape[1] - 12), str(row["category"]), fill=color)
            except (ValueError, TypeError):
                continue  # Skip rows with bad data

        return img
    except FileNotFoundError:
        return None


populate_anomaly_identification_page()
