import numpy as np
import trimesh
import plotly.graph_objects as go
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
import os

def hausdorff_distance(points_a, points_b):
    """
    Computes the symmetric Hausdorff distance between two point sets.
    """
    d1 = directed_hausdorff(points_a, points_b)[0]
    d2 = directed_hausdorff(points_b, points_a)[0]
    return max(d1, d2)

def slice_mesh(mesh, layer_height=0.05):
    """
    Virtually slices the mesh at given layer height.
    Returns a list of 3D points representing the contours of all slices.
    """
    z_min, z_max = mesh.bounds[:, 2]
    z_levels = np.arange(z_min + layer_height, z_max, layer_height)
    
    all_slice_points = []
    
    # Use trimesh to intersect the mesh with planes at each z level
    # trimesh.intersections.mesh_multiplane returns a list of Path3D or Path2D
    # We'll use the 3D version to get coordinates directly
    sections = mesh.section_multiplane(plane_origin=[0, 0, 0], 
                                     plane_normal=[0, 0, 1], 
                                     heights=z_levels)
    
    for i, section in enumerate(sections):
        if section is not None:
            # Each section can be a Path2D if it's a planar section.
            # We need to ensure the points are 3D by adding the Z level.
            vertices = section.vertices
            if vertices.shape[1] == 2:
                z_coord = z_levels[i]
                z_column = np.full((len(vertices), 1), z_coord)
                vertices = np.column_stack((vertices, z_column))
            all_slice_points.append(vertices)
            
    if not all_slice_points:
        return np.array([])
        
    return np.concatenate(all_slice_points, axis=0)

def detect_surface_lines(mesh, layer_height=0.05, threshold=20):
    """
    Detects surface line risks in a 3D STL mesh.
    
    Parameters:
    - mesh: trimesh.Trimesh object or path to the STL file.
    - layer_height: Slicing layer height in mm.
    - threshold: Angle threshold in degrees for curvature risk.
    """
    if isinstance(mesh, str):
        print(f"Loading mesh: {mesh}...")
        mesh = trimesh.load(mesh)
    
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        else:
            raise ValueError("Input must be a trimesh.Trimesh or a valid file path.")
    
    # 1. Curvature Risk (Angle between normal and Z-axis)
    print("Computing curvature risk...")
    normals = mesh.vertex_normals
    
    # Dot product: cos(theta) = N . Z
    cos_theta = np.abs(normals[:, 2]) 
    angles_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # Risk based on angle (Small angle with Z = horizontal surface = high stair-stepping risk)
    angle_risk = np.zeros(len(mesh.vertices))
    angle_risk[angles_deg < threshold] = 1.0
    angle_risk[(angles_deg >= threshold) & (angles_deg < 45)] = 0.5
    
    # 2. Slicing and Deviation Risk
    print(f"Slicing mesh at layer height {layer_height}mm...")
    slice_points = slice_mesh(mesh, layer_height)
    
    distances = np.zeros(len(mesh.vertices))
    if len(slice_points) > 0:
        print("Computing deviation via Nearest Neighbor distance...")
        tree = KDTree(slice_points)
        distances, _ = tree.query(mesh.vertices)
        
        max_dist = layer_height
        deviation_scores = np.clip(distances / max_dist, 0, 1)
        
        print("Calculating overall model metrics...")
        mesh_samples = mesh.sample(min(2000, len(mesh.vertices)))
        h_dist = hausdorff_distance(mesh_samples, slice_points)
        rmse = np.sqrt(np.mean(distances**2))
    else:
        deviation_scores = np.zeros(len(mesh.vertices))
        h_dist = 0
        rmse = 0

    # 3. Combine Risks
    total_risk_score = angle_risk + deviation_scores
    
    risk_levels = []
    for score in total_risk_score:
        if score > 1.2:
            risk_levels.append("High")
        elif score > 0.5:
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")
            
    # Output Report
    high_risk_pct = (risk_levels.count("High") / len(risk_levels)) * 100
    report = {
        "max_deviation": float(np.max(distances)),
        "avg_deviation": float(np.mean(distances)),
        "rmse": float(rmse),
        "hausdorff_distance": float(h_dist),
        "high_risk_zones_pct": high_risk_pct
    }
    
    return mesh, total_risk_score, risk_levels, report

def save_risk_report(vertices, risk_levels, scores, filename="risk_map.csv"):
    """
    Saves the risk map to a CSV file.
    """
    import csv
    print(f"Saving risk map to {filename}...")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vertex_X', 'Vertex_Y', 'Vertex_Z', 'Risk_Level', 'Risk_Score'])
        for i in range(len(vertices)):
            writer.writerow([*vertices[i], risk_levels[i], f"{scores[i]:.4f}"])

def plot_heatmap(mesh, risk_scores, output_path="risk_heatmap.html"):
    """
    Generates a 3D heatmap visualization overlay using Plotly.
    """
    print(f"Generating heatmap: {output_path}...")
    v = mesh.vertices
    f = mesh.faces
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            intensity=risk_scores,
            colorscale=[
                [0, 'rgb(0, 255, 0)'],    # Low: Green
                [0.5, 'rgb(255, 255, 0)'], # Medium: Yellow
                [1.0, 'rgb(255, 0, 0)']    # High: Red
            ],
            colorbar_title="Risk Score",
            name="Risk Map"
        )
    ])
    
    fig.update_layout(
        title="3D Surface Line Risk Heatmap",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(output_path)

def export_to_glb(mesh, risk_scores, filename="risk_model.glb"):
    """
    Exports the mesh to a GLB file with vertex colors mapped to risk scores.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    print(f"Exporting colored GLB to {filename}...")
    
    # Normalize scores for colormap (0-1 range for typical, but we can go up to 2.0 based on logic)
    # Total risk score = angle_risk (0 to 1) + deviation_scores (0 to 1)
    norm = mcolors.Normalize(vmin=0, vmax=2)
    cmap = plt.get_cmap('RdYlGn_r') # Red-Yellow-Green reversed -> Green-Yellow-Red
    
    # Get colors as RGBA, then convert to 0-255 uint8
    colors = cmap(norm(risk_scores))
    colors_uint8 = (colors[:, :3] * 255).astype(np.uint8)
    
    # Assign to trimesh
    mesh.visual.vertex_colors = colors_uint8
    
    # Export as GLB
    mesh.export(filename)

def main():
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to STL file (optional)")
    args = parser.parse_args()

    if args.input:
        mesh_path = args.input
    else:
        print("No input provided, creating a demo sphere...")
        demo_mesh = trimesh.creation.icosphere(subdivisions=3, radius=10)
        mesh_path = "demo_sphere.stl"
        demo_mesh.export(mesh_path)
    
    mesh, scores, levels, report = detect_surface_lines(mesh_path, layer_height=0.1, threshold=20)
    
    print("\n--- Numerical Report ---")
    for k, v in report.items():
        print(f"{k:20}: {v:.4f}")
    
    save_risk_report(mesh.vertices, levels, scores, "risk_map.csv")
    plot_heatmap(mesh, scores, "surface_risk_heatmap.html")
    export_to_glb(mesh, scores, "risk_model.glb")
    print("\nProcess finished.")

if __name__ == "__main__":
    main()
