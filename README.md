# Remote Sensing and Machine Learning for Oil Spill Detection

This project provides an automated system to detect and map marine oil spills. By combining satellite imagery with deep learning and news analysis, the platform helps environmental agencies respond to spills faster and more accurately.


## Team Members
* Yerassyl Kaiyrkanov, Sultan Margulan, Nurlan Boranbay, Lazzat Zhengissova, Baglan Zhubatkanov, Ismail Zaukenkhazhy
* **Project Advisor:** Siamac Fazli


## Project Overview

The system processes satellite data to identify oil spills through a multi-step pipeline:

1. **Data Acquisition:** Automatically downloads SAR (Sentinel-1) and Optical (Sentinel-2) images.
2. **Preprocessing:** Cleans raw images using speckle filtering and terrain correction to ensure consistency.
3. **Deep Learning:** Uses ResNet-50 for classification and DeepLabV3+ for semantic segmentation to find exact spill boundaries.
4. **NLP Validation:** Uses Large Language Models (LLMs) to scan global news for oil spill reports, cross-referencing them with satellite detections to reduce false positives.
5. **Visualization:** Displays results on an interactive map with analytical reports and spill quantity estimates.


## Technical Specifications

### Target Metrics
The project focuses on high-precision segmentation to define exact spill boundaries rather than simple bounding boxes.

| Model | Task | Target Metric |
| :--- | :--- | :--- |
| **ResNet-50** | Classification | ~95% Accuracy |
| **DeepLabV3+** | Segmentation | 60-65% IoU |

### Tech Stack
* **Deep Learning:** PyTorch, ResNet, DeepLabV3+
* **Remote Sensing:** Google Earth Engine (GEE), Copernicus Data Space
* **NLP:** GDELT API, LLM (OpenAI/Anthropic/Groq)
* **Web UI:** React, TypeScript, Supabase


## Current Progress

* Created a custom dataset by syncing SAR and Optical images from the DARTIS and CSIRO records.
* Developed an NLP module that converts unstructured news articles into geographic coordinates.
* Built a web dashboard for visualizing detection markers and spill statistics.
* Implemented a standardized preprocessing pipeline for both radar and optical data.
