import numpy as np
import trimesh
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def calculate_overhang_angles(mesh):
    """
    Calculates the overhang angle for each face of the mesh.
    Angle is measured from the vertical (Z-axis).
    Vertical Wall = 0 degrees
    Horizontal Ceiling = 90 degrees (facing down)
    Positive vertical surfaces (facing up) are ignored or set to 0.
    """
    normals = mesh.face_normals
    
    # We only care about downward faces (nz < 0) for overhang analysis
    # n_z is the vertical component
    nz = normals[:, 2]
    
    # Calculate angle from vertical
    # Angle = atan2(|nz|, sqrt(nx^2 + ny^2))
    # This gives 0 for vertical walls and 90 for horizontal ceilings
    nx_ny_mag = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2)
    
    # Initialize angles array
    overhang_angles = np.zeros(len(normals))
    
    # Filter for downward faces
    downward_mask = nz < 0
    
    # Calculate for those faces
    # Avoid division by zero if nx_ny_mag is 0 (horizontal faces)
    overhang_angles[downward_mask] = np.degrees(np.arctan2(np.abs(nz[downward_mask]), nx_ny_mag[downward_mask]))
    
    return overhang_angles

def export_overhang_glb(mesh, overhang_angles, filename="overhang_model.glb"):
    """
    Exports a GLB with vertex colors representing overhang angles.
    """
    print(f"Exporting overhang heatmap to {filename}...")
    
    # Map face angles to vertex angles (averaging or simple mapping)
    # trimesh can handle face colors directly if we want, but vertex colors are smoother
    
    # Map angles to colors
    # 0 deg (Vertical) -> Green
    # 45 deg (Warning) -> Yellow
    # 90 deg (Horizontal) -> Red
    norm = mcolors.Normalize(vmin=0, vmax=90)
    cmap = plt.get_cmap('RdYlGn_r') # Green to Red
    
    # Use face colors
    face_colors = (cmap(norm(overhang_angles))[:, :3] * 255).astype(np.uint8)
    
    # Create a copy of the mesh to avoid modifying the original
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = face_colors
    
    # Export
    colored_mesh.export(filename)
    print("Export complete.")

def main():
    parser = argparse.ArgumentParser(description="Overhang Angle Detection")
    parser.add_argument("--input", help="Path to STL file", default="0NXCCBL_SLA.stl")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

    print(f"Loading mesh: {args.input}...")
    mesh = trimesh.load(args.input)
    
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    print("Analyzing overhang angles...")
    angles = calculate_overhang_angles(mesh)
    
    # Print some stats
    max_angle = np.max(angles)
    avg_overhang = np.mean(angles[angles > 0]) if np.any(angles > 0) else 0
    critical_count = np.sum(angles > 45)
    total_faces = len(angles)
    
    print(f"\n--- Overhang Report ---")
    print(f"Max Overhang Angle: {max_angle:.2f}°")
    print(f"Average Overhang:   {avg_overhang:.2f}°")
    print(f"Critical Overhangs (>45°): {critical_count} faces ({(critical_count/total_faces)*100:.1f}%)")
    
    export_overhang_glb(mesh, angles, "overhang_model.glb")
    print("\nProcess finished.")

if __name__ == "__main__":
    main()
