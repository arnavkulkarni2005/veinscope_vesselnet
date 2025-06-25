# Deep Learning Hackathon

Welcome to the Deep Learning Hackathon project! This repository contains all the resources, code, and documentation needed to participate in the hackathon.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Contribute](#how-to-contribute)
- [License](#license)

# 🚀 Introduction

**VessenNet** is a deep learning–powered system for the **automated segmentation and analysis of scleral (tear-zone) vein patterns** from high-resolution eye images. It is designed as a two-stage, custom architecture that combines the strengths of U-Net and SegNet into a novel framework called **VesselNet**.

## 🧠 What is VesselNet?

**VesselNet** is our custom hybrid model composed of:

- **U-Net**: First stage used to **crop and isolate the scleral region** from the eye image with precise anatomical localization.
- **SegNet**: Second stage used to **segment the tear vein structures** specifically within the extracted scleral area.

This modular design improves both anatomical accuracy and vascular continuity, making it ideal for segmenting fine, thin, and branching vein structures.

---

## 🔑 Key Features

- CLAHE-based preprocessing to enhance visibility of vascular structures  
- Two-stage architecture for accurate region selection and segmentation  
- Extraction of biologically relevant features: vein density, tortuosity, branching, vessel width, and more  
- Optional health condition mapping (e.g., dehydration, fatigue) using clustering or rule-based heuristics  
- **FastAPI backend** for real-time inference  
- **Flutter app integration** for capturing eye images and displaying segmentation results directly on mobile  

---

## 💡 Use Cases

- Non-invasive wellness screening using ocular biomarkers  
- Real-time vascular analysis from mobile devices  
- Research on scleral vein behavior under stress, fatigue, or hydration changes  

---

**VessenNet** was developed during a hackathon to demonstrate the potential of deep learning and mobile-integrated health analysis from eye-based biometrics.
.

## Getting Started
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/DL_Hack.git
    ```
2. Navigate to the project directory:
    ```bash
    cd DL_Hack
    ```
3. Follow the instructions in the `requirements` section to set up your environment.

## Project Structure
```
/home/teaching/DL_Hack/
├── data/           # Dataset files
├── models/         # Pre-trained and custom models
├── notebooks/      # Jupyter notebooks for experiments
├── scripts/        # Utility scripts
└── README.md       # Project documentation
```

## Requirements
- Python 3.8 or higher
- Required libraries are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
  ```

<!-- ## How to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes. -->

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

<!-- Happy hacking! -->
