"""
Final page of UI. Contains model evaluation.
"""

import streamlit as st
from pathlib import Path
from config import PRECOMPUTED_DIR
from PIL import Image

st.set_page_config(
    page_title="Model Evaluation",
    page_icon=":material/sentiment_excited:",
    layout="wide",
)

# st.write(
# """
# | Metric        | All    | Small  | Medium | Large  |
# |---------------|--------|--------|--------|--------|
# | mAP @ 50      | 0.43   | 0.02   | 0.5    | 0.55   |
# | mAP @ 75      | 0.21   | 0.16   | 0.23   | 0.77   |
# | mAP @ 50:95   | 0.23   | 0.05   | 0.25   | 0.61   |
# | Peak F1 score | 0.6081 | 0.3972 | 0.6973 | 0.9073 |


# | Category      | AP @ 50 | Peak F1 Score | Confidence Threshold |
# |---------------|---------|---------------|----------------------|
# | traffic sign  | 0.5256  | 0.5659        | 0.3                  |
# | traffic light | 0.4728  | 0.5413        | 0.4                  |
# | car           | 0.6802  | 0.6686        | 0.4                  |
# | rider         | 0.3436  | 0.4227        | 0.3                  |
# | motor         | 0.3621  | 0.4219        | 0.3                  |
# | person        | 0.4404  | 0.4738        | 0.3                  |
# | bus           | 0.5510  | 0.5621        | 0.4                  |
# | truck         | 0.5447  | 0.5513        | 0.4                  |
# | bike          | 0.3525  | 0.4017        | 0.3                  |
# | train         | 0.0124  | 0.0606        | 0.1                  |
# """
# )
st.write(
    """
    # Task 3: Model Evaluation

    Please refer to the notebook called "Task3_Evaluation.ipynb" in the "notebooks" folder in the root of this repository for the code I wrote to generate these metrics and visualizations.

    ## Quantitative Analysis
    
    The following analysis evaluates the model's detection capabilities across different object scales and categories after training for 6 epochs.

"""
)
mapwhole = Image.open(Path(PRECOMPUTED_DIR) / "map_plot.png")
st.image(mapwhole, caption="mAP metrics for the entire dataset")

st.write("""

    ### Performance by Object Size
    The model demonstrates a strong correlation between object size and detection accuracy. For large objects, the model performs exceptionally well, achieving a Peak F1 score of 0.91 and an mAP@50 of 0.55. This indicates that when objects are close and distinct, the model is highly reliable.
    For small objects, there is a critical performance drop-off (Area < $32^2$ px). The mAP@50 is only 0.02, and the mAP@50:95 is 0.05. This confirms the challenges identified in the dataset analysis task, that small, distant objects (like traffic lights or far-away vehicles) are the primary failure mode for this model configuration.

    | Metric | All | Small | Medium | Large |
    | :--- | :--- | :--- | :--- | :--- |
    | mAP @ 50 | 0.43 | 0.02 | 0.50 | 0.55 |
    | mAP @ 50:95 | 0.23 | 0.05 | 0.25 | 0.61 |
    | Peak F1 | 0.61 | 0.40 | 0.70 | 0.91 |

    ### Class-Wise Performance
    The model shows distinct biases towards classes that are more frequent or visually distinct in the dataset.

    Cars are the best-detected class (AP@50: 0.68), likely due to their high frequency and consistent shape.
    
    Large Vehicles (Bus and Truck) also see strong performance, with AP@50 scores above 0.54.
    
    Vulnerable Road Users, i.e., Categories like Rider, Motor, and Bike hover around 0.35 - 0.36 AP. These classes are often thinner (high aspect ratio anomalies) and harder to resolve than boxy vehicles.

    Edge Cases: Trains have a near-zero performance (AP@50: 0.01). This primarily indicates a lack of training samples (class imbalance) rather than a fundamental inability to detect the object type.

    | Category | AP @ 50 | Peak F1 | Conf. Threshold |
    | :--- | :--- | :--- | :--- |
    | Car | 0.68 | 0.67 | 0.4 |
    | Bus | 0.55 | 0.56 | 0.4 |
    | Truck | 0.54 | 0.55 | 0.4 |
    | Traffic Sign | 0.53 | 0.57 | 0.3 |
    | Traffic Light| 0.47 | 0.54 | 0.4 |
    | Person | 0.44 | 0.47 | 0.3 |
    | Motor | 0.36 | 0.42 | 0.3 |
    | Bike | 0.35 | 0.40 | 0.3 |
    | Rider | 0.34 | 0.42 | 0.3 |
    | Train | 0.01 | 0.06 | 0.1 |

    """)

mapnight = Image.open(Path(PRECOMPUTED_DIR) / "day_performance.png")
mapday = Image.open(Path(PRECOMPUTED_DIR) / "night_performance.png")

st.write(
    "As suspected in the data analysis section, the model also performs slightly worse at nighttime, possibly due to lighting inconsistencies and environmental effects. However, the balance of daytime and nighttime images has kept the model from performing too poorly at nighttime."
)

col1, col2 = st.columns(2)


with col1:
    st.header("mAP for daytime images")
    st.image(mapday, use_container_width=True)

with col2:
    st.header("mAP for nighttime images")
    st.image(mapnight, use_container_width=True)

st.write(
    "Our first insights from the metrics are confirmed by the confusion matrix and PR curve"
)

confusion = Image.open(Path(PRECOMPUTED_DIR) / "confusion.png")
prcurve = Image.open(Path(PRECOMPUTED_DIR) / "prcurve.png")
col3, col4 = st.columns(2)


with col3:
    st.header("Confusion Matrix")
    st.image(confusion, use_container_width=True)

with col4:
    st.header("PR Curve")
    st.image(prcurve, use_container_width=True)

st.write("""
    ## Qualitative Analysis
         
    Let us visualize some images with ground truth labels and predictions side by side to see how the model actually did.
""")

densecars = Image.open(Path(PRECOMPUTED_DIR) / "dense_cars.png")
st.write(
    "The fine-tuned model is rather good at detecting even occluded cars in a dense setting with cars"
)
st.image(
    densecars,
)

denseregion = Image.open(Path(PRECOMPUTED_DIR) / "dense_region.png")
st.write("But it misses several detections regularly, especially far away objects")
st.image(denseregion)

denseperson = Image.open(Path(PRECOMPUTED_DIR) / "dense_person.png")
st.write("In this dense person setting it has performed reasonably well")
st.image(denseperson)

big_objects = Image.open(Path(PRECOMPUTED_DIR) / "big_objects.png")
st.write("As is evident from the metrics, it is good at detecting large objects")
st.image(big_objects)

night_performance1 = Image.open(Path(PRECOMPUTED_DIR) / "night_big.png")
st.write(
    "In a nighttime setting, it misses many far away objects, but does well with traffic lights, possibly because the lights are more pronounced due to a lack of background light"
)
st.image(night_performance1)

night_performance2 = Image.open(Path(PRECOMPUTED_DIR) / "night_big2.png")
st.image(night_performance2)


st.write("""
# Conclusion: The model is effective at detecting large or nearby objects in its current state. To improve generalizability, future iterations must utilize a more balanced dataset and train for many more epochs.
""")
