"""
Page for task 2
"""

import streamlit as st
from pathlib import Path
from config import PRECOMPUTED_DIR
from PIL import Image

st.set_page_config(
    page_title="Model Selection and Training",
    page_icon=":material/sentiment_excited:",
    layout="wide",
)
st.write(
    """
    ## Task 2.1: Model Selection

    ### Rationale for Selecting RF-DETR
    For the BDD100K dataset, I propose utilizing a relatively new real-time detection transformer model called RF-DETR. This model is particularly well suited for autonomous driving environments characterized by high variance in lighting (day versus night) and object scale (traffic lights versus vehicles).

    The primary justification for this choice is RF-DETR's ability to bridge the gap between real-time inference speed and the high accuracy typically associated with heavy vision-language models. Since BDD100K contains a significant portion of nighttime images where visibility is compromised, the model relies on the DINOv2 backbone. This backbone provides robust, internet-scale pre-trained features that improve detection accuracy on challenging domains compared to standard CNN backbones.

    Furthermore, RF-DETR is engineered specifically as a real-time detector, modernizing specialist architectures to achieve state-of-the-art inference speeds. Empirical benchmarks demonstrate that its performance parallels and often exceeds the latest YOLO models (such as the latest YOLO 26) in terms of both latency and mean Average Precision (mAP). This ensures the system can process video feeds at the high frame rates required for autonomous navigation without compromising on detection quality.
"""
)
image = Image.open(Path(PRECOMPUTED_DIR) / "rf_detr.png")
st.image(image, caption="RF-DETR performance comparison with latest models")
st.write(
    """
    ### Key Architectural Components
    RF-DETR introduces a "scheduler-free" Neural Architecture Search (NAS) that optimizes the model specifically for the target dataset without the need for repetitive re-training. The following components make it highly effective for BDD100K:

    * Weight-Sharing NAS: This mechanism allows the model to dynamically adjust architectural "knobs" such as patch size and decoder depth. For BDD100K, which contains numerous small anomalies (objects $< 32^2$ pixels), the NAS can automatically select higher input resolutions or smaller patch sizes during training to resolve fine-grained details.
    * Adaptive Query Dropping: The architecture allows for dropping low-confidence query tokens at inference time based on the encoder's output. In sparse highway scenes often found in BDD100K, this reduces computational waste by not processing all 300 standard queries, thereby improving latency without sacrificing accuracy.
    * Consistent Spatial Organization: The model utilizes bilinear interpolation in its detection heads to maintain spatial feature alignment. This is critical for accurately bounding the "sliver" anomalies (extreme aspect ratios) identified in my dataset analysis.

    ### Limitations and Risks
    Despite its advantages, there are specific limitations to consider for deployment:

    * Latency Determinism: Benchmarking reveals that latency measurements can vary by up to 0.1ms due to GPU power throttling and the non-deterministic behavior of TensorRT compilation. This may introduce slight jitter in real-time tracking applications.
    * Quantization Sensitivity: The model is sensitive to FP16 quantization. Naive conversion can degrade performance significantly, in some cases dropping accuracy to near zero, unless specific ONNX opsets (opset 17) are utilized during export.
    * Computational Cost of Search: While the final model is efficient, the initial NAS process is resource-intensive, estimated to require approximately 10,000 GPU hours for a comprehensive search space exploration.
    * Training Convergence & Resource Constraints: RF-DETR is a parameter-heavy model (30.5M parameters for the Nano version compared to 3.2M for YOLOv8 Nano). Consequently, training takes roughly two to four times longer than non-NAS baselines , with full convergence often requiring over 100 epochs. Due to current compute limitations, I will restrict training to 6 epochs, acknowledging this as a potential bottleneck for maximizing model performance.
    """
)

st.write(
    """
    ## Task 2.2: Model Training

    Please refer to the notebook called "Task2_Model_Training.ipynb" in the "notebooks" folder at the root of this repository. 
    """
)
