# My submission for Bosch's Applied CV Coding Assignment
[![Project Validation](https://github.com/plsgivecheesecake/bosch-applied-cv-assignment/actions/workflows/ci.yml/badge.svg)](https://github.com/plsgivecheesecake/bosch-applied-cv-assignment/actions/workflows/ci.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## Setup
Please make sure to do the following
 - Set environment variable called BDD_DATASET_PATH with the directory path where the **EXTRACTED** contents of the assignment_data_bdd.zip file are located on your system. This can be done in one of two ways. You can either set this as an OS variable, or you can use the .env file provided in this repository. If on Windows, please make sure to escape the backslashes or just use a UNIX-style path.
 - Install Docker and Docker Compose

Once the above two steps have been completed, running the project is as simple as running 
``` docker compose up --build ``` from the root of the repository.

When the app is up and running, please navigate to http://localhost:8501/ to access the UI. You will be greeted by a page that looks like this
![Process Dataset](readme_images/process_dataset.jpg)

Please make sure to click the "Process Dataset" button before proceeding to the next pages.
For ease of reading, I have divided task 1 into 5 pages. last two pages are for task 2 and task 3.

## Task 1 : Dataset Analysis

## Task 2.1: Model Selection

### Rationale for Selecting RF-DETR
For the BDD100K dataset, I propose utilizing a relatively new real-time detection transformer model called RF-DETR. This model is particularly well suited for autonomous driving environments characterized by high variance in lighting (day versus night) and object scale (traffic lights versus vehicles).

The primary justification for this choice is RF-DETR's ability to bridge the gap between real-time inference speed and the high accuracy typically associated with heavy vision-language models. Since BDD100K contains a significant portion of nighttime images where visibility is compromised, the model relies on the DINOv2 backbone. This backbone provides robust, internet-scale pre-trained features that improve detection accuracy on challenging domains compared to standard CNN backbones.

Furthermore, RF-DETR is engineered specifically as a real-time detector, modernizing specialist architectures to achieve state-of-the-art inference speeds. Empirical benchmarks demonstrate that its performance parallels and often exceeds the latest YOLO models (such as the latest YOLO 26) in terms of both latency and mean Average Precision (mAP). This ensures the system can process video feeds at the high frame rates required for autonomous navigation without compromising on detection quality.

![RF-DETR Performance](readme_images/rf_detr.png)

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

## Task 2.2: Model Training

Please refer to the notebook called "Task2_Model_Training.ipynb" in the "notebooks" folder at the root of this repository. 