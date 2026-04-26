# AI Mesh Detection Algorithms

This repository contains advanced 3D metrology tools for detecting and visualizing geometric risks in STL models, specifically optimized for high-precision manufacturing (e.g., hearing aid shells).

## 🚀 Unified 3D Metrology Dashboard
We have unified the detection tools into a single, high-performance web dashboard:
- **Surface Risk Mode**: Detects stair-stepping and contour lines based on slope angles from horizontal.
- **Overhang Mode**: Identifies surfaces facing downward that require support structures.

### Key Features:
- **Geometric Slicing Engine**: Real-time generation of purple contour lines.
- **Dynamic Thresholds**: Interactive sliders to adjust sensitivity based on research baselines.
- **In-Browser Analysis**: No server-side processing required for visualization.

## 🛠️ Included Tools
- `dashboard.html`: The main unified dashboard.
- `overhang_analysis.py`: Geometric normal analysis for overhangs.
- `main.py`: Surface line discretization error analysis.

## 🏃 How to Run
1. Serve the files using a local server:
   ```bash
   python -m http.server 8000
   ```
2. Open `localhost:8000/dashboard.html` in your browser.
3. Drag and drop any STL file for instant analysis.

## Features
- **Curvature Risk Detection**: Identifies near-horizontal surfaces that are prone to stair-stepping based on the angle between surface normals and the Z-axis.
- **Virtual Slicing**: Slices the mesh at a specified layer height (e.g., 0.05 mm) to evaluate discretization error.
- **Deviation Metrics**: Computes RMSE and Hausdorff distance between the original mesh and the sliced contours.
- **Heatmap Visualization**: Generates an interactive 3D heatmap (HTML) showing risk zones:
  - **Red**: High Risk
  - **Yellow**: Medium Risk
  - **Green**: Low Risk
- **Numerical Report**: Outputs max/average deviation and percentage of high-risk zones.

## Installation
Ensure you have Python installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the script with an STL file:
```bash
python main.py --input path/to/your_model.stl
```

If no input is provided, the script will generate a demo sphere.

## Deliverables
- `main.py`: Main script containing the analysis logic.
- `requirements.txt`: Required Python libraries.
- `risk_map.csv`: Detailed per-vertex risk assessment.
- `surface_risk_heatmap.html`: Interactive 3D visualization.
