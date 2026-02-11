"""
Second page of the app containing initial observations after taking a first look at the data
"""

import streamlit as st
from PIL import Image
from pathlib import Path
from config import PRECOMPUTED_DIR
import json


def populate_initial_observations_page():
    """
    Initializes the "Initial Observations" page
    """
    st.set_page_config(layout="wide")
    st.write(
        """
        # Initial Observations
        
        Counting the images and corresponding labels provided in the zip file reveals
        that 137 images in the train set do not have any labels.

        | Split | Images | Labels | Missing Labels |
        |-------|--------|--------|----------------|
        | train | 70000  | 69863  |       137      |
        | val   | 10000  | 10000  |        0       |

        Below are three such images lacking labels.
        """
    )
    labelless_image1 = Image.open(Path(PRECOMPUTED_DIR) / "6a76c075-d995ef0a.jpg")
    labelless_image2 = Image.open(Path(PRECOMPUTED_DIR) / "5fdc609c-d87d775f.jpg")
    labelless_image3 = Image.open(Path(PRECOMPUTED_DIR) / "6f0cc882-8b3e2238.jpg")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Out of focus")
        st.image(labelless_image1, use_container_width=True)

    with col2:
        st.subheader("Not too bad")
        st.image(labelless_image2, use_container_width=True)

    with col3:
        st.subheader("Quite Good")
        st.image(labelless_image3, use_container_width=True)

    st.write(
        "While the left image is out of focus and rather bad, the image on the right doesn't look too bad."
    )
    st.write(
        """
        Regardless, these 137 images that lack labels will not be used in the analysis, or during training.

        But before proceeding with the analysis, let us take a look at the JSON structure of the labels, to get a
        first impression of what information can be extracted from it.
        """
    )

    with open(Path(PRECOMPUTED_DIR) / "outer_structure.json", "r") as f:
        image_json = json.load(f)

    with open(Path(PRECOMPUTED_DIR) / "label.json", "r") as f:
        label_json = json.load(f)

    st.json(image_json)
    st.write(
        """
        This JSON represents the main structure of each image's label, and denotes the weather and time of day when the image was taken.
        It also denotes the scene of the dataset from which the image was taken, along with the timestamp, which is 10000 for all images.
        Each image's JSON has a list called "labels", which contains the annotations for the objects present in the image.

        The classes that are of most interest in this assignment are: "traffic sign", "traffic light", "car", "rider", "motor", "person", "bus", "truck", "bike", and "train".

        An example of the annotation JSON for each of these objects is shown below:
        """
    )

    st.json(label_json)
    st.write(
        """
        The structure denotes the occlusion and truncation statuses of a label, along with the bottom left and top right coordinates of its bounding box.
        Traffic lights also have a traffic light color, but we will not be using this attribute in this assignment.
        """
    )
    st.warning(
        'In the next pages, we shall proceed with the dataset analysis. Please note, in order for the visuals to render in the next pages, you will need to click the "Process Dataset" button in the previous page.'
    )


populate_initial_observations_page()
