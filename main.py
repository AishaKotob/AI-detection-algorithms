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

def slice_mesh_v2(mesh, layer_height=0.1):
    """
    Slices the mesh and returns a list of individual contour paths.
    Each element in the returned list is a (N, 3) numpy array of points for a single loop.
    """
    z_min, z_max = mesh.bounds[:, 2]
    z_levels = np.arange(z_min + layer_height, z_max, layer_height)
    
    all_contours = []
    for z in z_levels:
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is not None:
            # discrete returns a list of (N, 3) vertices for each loop
            paths = section.discrete
            for p in paths:
                if len(p) > 1:
                    all_contours.append(p)
                    
    return all_contours

def plot_contours(mesh, contours, output_path="surface_lines_contours.html"):
    """
    Generates Dreve-style purple contour visualization.
    """
    print(f"Generating Dreve-style contours: {output_path}...")
    v = mesh.vertices
    f = mesh.faces
    
    fig = go.Figure()

    # 1. Base Mesh (Semi-transparent gray)
    fig.add_trace(go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color='lightgray',
        opacity=0.2, # As requested
        name="Base Mesh",
        hoverinfo='skip'
    ))

    # 2. Purple Contour Lines
    for i, pts in enumerate(contours):
        # To make it a closed loop, append the first point
        pts_closed = np.vstack([pts, pts[0]])
        fig.add_trace(go.Scatter3d(
            x=pts_closed[:, 0], y=pts_closed[:, 1], z=pts_closed[:, 2],
            mode='lines',
            line=dict(color='purple', width=3),
            name=f"Contour {i}",
            showlegend=False
        ))

    fig.update_layout(
        title="Dreve-Style Purple Contour Lines",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(output_path)

def detect_surface_lines(mesh, layer_height=0.1, threshold=20):
    """
    Detects surface line risks in a 3D STL mesh.
    """
    if isinstance(mesh, str):
        print(f"Loading mesh: {mesh}...")
        mesh = trimesh.load(mesh)
    
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        else:
            raise ValueError("Input must be a trimesh.Trimesh.")
    
    # 1. Curvature Risk
    print("Computing curvature risk...")
    normals = mesh.vertex_normals
    cos_theta = np.abs(normals[:, 2]) 
    angles_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    angle_risk = np.zeros(len(mesh.vertices))
    angle_risk[angles_deg < threshold] = 1.0
    angle_risk[(angles_deg >= threshold) & (angles_deg < 45)] = 0.5
    
    # 2. Slicing and Deviation Risk
    print(f"Slicing mesh at layer height {layer_height}mm...")
    contours = slice_mesh_v2(mesh, layer_height)
    
    distances = np.zeros(len(mesh.vertices))
    # For deviation calculation, we still need flattened points
    if contours:
        flat_pts = np.concatenate(contours, axis=0)
        print("Computing deviation via Nearest Neighbor distance...")
        tree = KDTree(flat_pts)
        distances, _ = tree.query(mesh.vertices)
        deviation_scores = np.clip(distances / layer_height, 0, 1)
    else:
        deviation_scores = np.zeros(len(mesh.vertices))

    # 3. Combine Risks
    total_risk_score = angle_risk + deviation_scores
    risk_levels = ["High" if s > 1.2 else ("Medium" if s > 0.5 else "Low") for s in total_risk_score]
            
    high_risk_pct = (risk_levels.count("High") / len(risk_levels)) * 100
    report = {
        "max_deviation": float(np.max(distances)),
        "high_risk_zones_pct": high_risk_pct
    }
    
    return mesh, total_risk_score, risk_levels, report, contours

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
    import argparse
    parser = argparse.ArgumentParser(description="3D Metrology: Surface Risk & Contour Analysis")
    parser.add_argument("--input", help="Path to STL file (optional)")
    parser.add_argument("--layer_height", type=float, default=0.1, help="Slicing layer height (mm)")
    parser.add_argument("--threshold", type=float, default=20.0, help="Angle threshold from horizontal")
    parser.add_argument("--show_contours", action="store_true", help="Generate Dreve-style purple contour plot")
    args = parser.parse_args()

    if args.input:
        mesh_path = args.input
    else:
        print("No input provided, creating a demo sphere...")
        demo_mesh = trimesh.creation.icosphere(subdivisions=3, radius=10)
        mesh_path = "demo_sphere.stl"
        demo_mesh.export(mesh_path)
    
    mesh, scores, levels, report, contours = detect_surface_lines(
        mesh_path, layer_height=args.layer_height, threshold=args.threshold
    )
    
    print("\n--- Metrology Report ---")
    print(f"Max Deviation: {report['max_deviation']:.4f} mm")
    print(f"High Risk Area: {report['high_risk_zones_pct']:.1f}%")
    
    # 1. Heatmap Visualization
    plot_heatmap(mesh, scores, "surface_risk_heatmap.html")
    
    # 2. Dreve-style Contours (Overlay)
    if args.show_contours:
        plot_contours(mesh, contours, "surface_lines_contours.html")
    
    # 3. Data Outputs
    save_risk_report(mesh.vertices, levels, scores, "risk_map.csv")
    export_to_glb(mesh, scores, "risk_model.glb")
    
    print("\n[SUCCESS] All Dreve-style visualizations and reports generated.")

if __name__ == "__main__":
    main()
