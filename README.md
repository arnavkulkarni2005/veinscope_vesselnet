# VesselNet: A Deep Learning Vision for Proactive Eye Health

Welcome to the VesselNet project repository. This document outlines the architecture, features, and future vision of VesselNet, a novel system for non-invasive ocular health monitoring.

---

## Table of Contents
- [Vision & Mission](#vision--mission)
- [The VesselNet Architecture](#the-vesselnet-architecture)
- [Key Features](#key-features)
- [Potential Applications](#potential-applications)
- [From Hackathon to Health-Tech: Our Roadmap](#from-hackathon-to-health-tech-our-roadmap)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Contribute](#how-to-contribute)
- [License](#license)

---

## Vision & Mission
In an era of unprecedented digital screen time, eye strain and related health conditions are on the rise. Our mission is to empower individuals and clinicians with a proactive, non-invasive tool for monitoring ocular biomarkers.  
We envision VesselNet becoming a daily habit for the digitally-strained generation, transforming the smartphone camera into a powerful device for promoting long-term eye health and wellness.

---

## The VesselNet Architecture
VesselNet is a custom, two-stage deep learning pipeline designed for the precise segmentation and analysis of scleral (tear-zone) vein patterns from high-resolution eye images.  
Our hybrid architecture ensures both anatomical accuracy and vascular continuity, which is critical for extracting meaningful biomarkers.

**Pipeline Overview:**
1. **Preprocessing**:  
   Input eye images are enhanced using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve visibility of fine vascular structures.
2. **Stage 1 – Scleral Region Isolation (Attention U-Net)**:  
   A U-Net model isolates and crops the scleral region, removing noise from eyelids and iris.
3. **Stage 2 – Vein Segmentation using MedSegNetNCA (SegNet + MedNCA)**:  
   The isolated scleral ROI is processed through a hybrid SegNet + MedNCA pipeline, where SegNet provides robust structural segmentation while MedNCA (Medical Neural Cellular Automata) refines fine-grained vascular details through iterative, self-organizing updates. This combination enables precise delineation of the thin, intricate, and branching scleral veins, preserving anatomical continuity even in challenging imaging conditions.
4. **Post-processing & Feature Extraction**:  
   - **Vein Density** – vessel area relative to scleral area  
   - **Tortuosity** – measure of vein curvature  
   - **Branching Analysis** – junctions and endpoints count  
   - **Average Vessel Width** – mean vein thickness

This modular design outperforms single-stage models by decoupling localization from fine-grained segmentation.

---

## Key Features
- **High-Precision Two-Stage Architecture** – Attention U-Net + MedSegNetNCA for robust segmentation
- **Advanced Preprocessing** – CLAHE for performance in varied lighting
- **Rich Biomarker Extraction** – density, tortuosity, branching, width
- **Scalable Backend** – FastAPI-powered inference
- **Seamless Mobile Integration** – Flutter app from capture to insights

---

## Potential Applications
### For Personal Wellness
- **Daily Health Tracking** – monitor changes in scleral veins
- **Habit Formation** – promote healthy screen-time behavior
- **Non-invasive Wellness Screening**

### For Clinical & Research Use
- **Clinician’s Assistant** – quantitative supplement to exams
- **Longitudinal Studies** – monitor vascular changes over time
- **Telemedicine** – remote patient monitoring

---

## From Hackathon to Health-Tech: Our Roadmap

**Phase 1: Refinement & Validation (Current)**
- [ ] Expand Dataset  
- [ ] Model Optimization (attention mechanisms, lightweight backbones)  
- [ ] Longitudinal Analysis algorithms  
- [ ] UX/UI Enhancement for mobile app  

**Phase 2: Deployment & Integration**
- [ ] Cloud Deployment  
- [ ] Publish Research Paper  
- [ ] Public API Release  

**Phase 3: Commercialization & Clinical Trials**
- [ ] Clinical Validation with ophthalmologists  
- [ ] Regulatory Compliance (HIPAA, etc.)  
- [ ] Feature Expansion (disease correlation models)

---

## Getting Started
```bash
# Clone this repository
git clone https://github.com/your-username/DL_Hack.git

# Navigate to the project directory
cd DL_Hack

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure
```
/home/teaching/DL_Hack/
├── data/              # Datasets for training and validation
├── models/            # Pre-trained weights (U-Net, SegNet)
├── notebooks/         # Jupyter notebooks for experiments and training
├── app/               # Backend (FastAPI) and Flutter client
│   ├── backend/
│   └── flutter_app/
├── scripts/           # Data processing and feature extraction scripts
└── README.md          # Documentation
```

---

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- Scikit-image
- FastAPI

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Contribute
We welcome contributions to advance accessible health tech.

1. Fork the repository  
2. Create a feature branch  
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes  
4. Submit a Pull Request with a clear description

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
