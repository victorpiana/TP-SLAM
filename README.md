# TP‑SLAM – Visual Simultaneous Localization and Mapping Project

> Practical work for visual SLAM, developed during my engineering studies in the **Monde Virtuel (Virtual Worlds)** specialization.

---

## Table of contents  
- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Technologies & Requirements](#technologies--requirements)  
- [Installation](#installation)  
- [Usage / How to run](#usage--how-to-run)  
- [Expected Results](#expected-results)  
- [Learning Goals](#learning-goals)  
- [Contributions](#contributions)  
- [License](#license)  
- [Author](#author)  

---

## Overview  
This repository contains a practical assignment on Visual SLAM — the problem of estimating camera motion and building a map of the environment simultaneously. It was completed as part of my engineering curriculum in the **Monde Virtuel (Virtual Worlds)** track. The assignment includes implementation code and documentation.

---

## Repository Structure  
Example structure (update according to your files):  

```
.
├── src/                          # Source code folder
│   ├── slam_main.py              # Main SLAM pipeline script
│   ├── mapping.py                # Mapping module
│   ├── tracking.py               # Tracking module
│   ├── visualization.py          # Visualization utilities
│   └── utils.py                  # Helper functions
├── data/                         # Example datasets or input images
│   └── sequences/                # Image sequences used for SLAM testing
├── docs/                         # Reports or documentation
│   └── TP_SLAM_Report.pdf
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Technologies & Requirements  
- Python 3.8+  
- NumPy  
- OpenCV  
- Matplotlib  
- (Optional) SciPy, tqdm, or Open3D for visualization  

Example `requirements.txt`:  
```
numpy
opencv-python
matplotlib
```

---

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/victorpiana/TP-SLAM.git
   cd TP-SLAM
   ```
2. (Recommended) Create a virtual environment:  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage / How to run  
Run the main script with sample data:  
```bash
python src/slam_main.py --input data/sequences/sequence1
```  

Options may include camera intrinsics, dataset paths, or output folders.  
If available, check command-line help:  
```bash
python src/slam_main.py --help
```  

---

## Expected Results  
Running the code should:  
- Track camera poses across image frames.  
- Reconstruct a sparse or dense 3D map of the scene.  
- Display 2D feature matches and 3D point clouds (if visualization is enabled).  

Example outputs could include:  
- Pose trajectories (in `.txt` or `.csv` format).  
- Saved keyframe visualizations.  
- 3D reconstructions (e.g., `.ply` or `.obj` files).  

---

## Learning Goals  
This project aims to:  
- Understand and implement the fundamentals of **Visual SLAM** (Simultaneous Localization and Mapping).  
- Practice **feature extraction, matching, and pose estimation** using real image sequences.  
- Learn how to combine **tracking**, **mapping**, and **visualization** in a unified system.  
- Strengthen Python and computer vision programming skills.  


---

## Author  
**Victor Piana**  
Engineering student — *Monde Virtuel (Virtual Worlds)* specialization
