# Surface Line Risk Analysis

This tool detects surface line risks (stair-stepping) in 3D STL mesh models, which is critical for 3D printing quality assessment.

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
