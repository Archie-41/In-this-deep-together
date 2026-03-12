# Summary: Model Architecture (arXiv:2111.04881)

This paper proposes a **hybrid machine learning + physics framework** for detecting and tracking **dark solitons** in experimental images of Bose–Einstein condensates (BECs).

Instead of using a single deep neural network, the system is structured as a **multi-stage pipeline** that combines computer vision with physics-based validation.

---

# Overview

The architecture consists of four main components:

1. CNN-based object detection
2. Physics-informed classification
3. Physics-based quality scoring
4. Optional multi-frame tracking

Pipeline:

BEC Image
↓
CNN Object Detector
↓
Candidate Soliton Regions
↓
Physics-Informed Classifier
↓
Soliton Type Prediction
↓
Physics Quality Metric
↓
Validated Solitons
↓
(Optional) Multi-frame Tracking

---

# 1. Input

* **Experimental density images** of a Bose–Einstein condensate
* Dark solitons appear as **localized density dips**

Goals:

* Detect solitons
* Classify their type
* Track them across time

---

# 2. Stage 1 — Object Detection Network

**Purpose:** Locate candidate solitons in the image.

**Model type**

* Convolutional Neural Network (CNN) object detector

**Input**

* Full BEC density image

**Output**

* Bounding boxes around candidate soliton regions

Conceptually:

Image → CNN Detector → Candidate Regions

This stage performs the **core computer vision detection task**.

---

# 3. Stage 2 — Physics-Informed Classifier

Each detected region is analyzed with a classifier that combines **image features and physics-based features**.

**Purpose**

* Distinguish true solitons from other density defects.

Possible categories include:

* Longitudinal dark soliton
* Transverse excitation
* Other density perturbations

**Physics-inspired features may include**

* Density minimum depth
* Soliton width
* Orientation
* Contrast relative to background

Classifier structure:

[Image features + Physics features] → Classifier → Soliton Class

This hybrid design helps **reduce false positives** compared to purely image-based models.

---

# 4. Stage 3 — Physics-Based Quality Metric

After classification, a **quality score** is computed to evaluate how well the detected feature matches theoretical soliton expectations.

The score evaluates:

* Expected soliton density profile
* Shape consistency
* Deviation from theoretical behavior

Output:

Quality Score = [0,1]

Uses:

* Filtering incorrect detections
* Increasing reliability of the pipeline

---

# 5. Multi-Frame Tracking

For **time-series image data**, the system tracks solitons across frames.

Tracking relies on:

* Spatial proximity between detections
* Motion consistency
* Classification confidence

This enables the framework to:

* Maintain identity of individual solitons
* Track trajectories
* Observe interactions

---

# Key Idea

The core contribution of the architecture is **integrating machine learning with physics constraints**.

| Component        | Role                            |
| ---------------- | ------------------------------- |
| CNN detector     | Detect candidate structures     |
| Physics features | Encode domain knowledge         |
| Quality metric   | Enforce theoretical constraints |
| Tracking module  | Track solitons over time        |

This design improves robustness compared to **purely data-driven approaches**.

---

# Intuition

Deep learning is used to **detect candidate structures**, while physics-based analysis is used to **verify whether those structures behave like real solitons**.
