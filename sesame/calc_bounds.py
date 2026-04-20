import struct
import os
from pathlib import Path

def get_mesh_bounds(stl_path):
    if not os.path.exists(stl_path):
        return None
    
    try:
        with open(stl_path, 'rb') as f:
            header = f.read(80)
            if len(header) < 80: return None
            count_bytes = f.read(4)
            if len(count_bytes) < 4: return None
            count = struct.unpack('<I', count_bytes)[0]
            
            # Binary STL has 50 bytes per facet
            # Check if file size matches
            expected_size = 84 + count * 50
            actual_size = os.path.getsize(stl_path)
            
            min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
            max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
            
            if expected_size == actual_size:
                # Binary STL
                for _ in range(count):
                    f.read(12) # Normal
                    for _ in range(3): # 3 vertices
                        v = struct.unpack('<fff', f.read(12))
                        min_x = min(min_x, v[0])
                        max_x = max(max_x, v[0])
                        min_y = min(min_y, v[1])
                        max_y = max(max_y, v[1])
                        min_z = min(min_z, v[2])
                        max_z = max(max_z, v[2])
                    f.read(2) # Attr
            else:
                # Fallback to simple ASCII parsing if needed
                f.seek(0)
                for line in f:
                    line = line.decode('utf-8', errors='ignore').strip().lower()
                    if line.startswith('vertex'):
                        parts = line.split()
                        if len(parts) >= 4:
                            v = [float(parts[1]), float(parts[2]), float(parts[3])]
                            min_x = min(min_x, v[0])
                            max_x = max(max_x, v[0])
                            min_y = min(min_y, v[1])
                            max_y = max(max_y, v[1])
                            min_z = min(min_z, v[2])
                            max_z = max(max_z, v[2])
            
            return (min_x, min_y, min_z), (max_x, max_y, max_z)
    except Exception as e:
        print(f"Error reading {stl_path}: {e}")
        return None

meshes = [
    'base_link.STL',
    'Link_L1.STL', 'Link_L2.STL', 'Link_L3.STL', 'Link_L4.STL',
    'Link_R1.STL', 'Link_R2.STL', 'Link_R3.STL', 'Link_R4.STL'
]

mesh_dir = Path('/home/priamai/mujoco_menagerie/sesame/meshes')

print("| Mesh | Center (xyz) | Size (dim) |")
print("| :--- | :--- | :--- |")

for mesh in meshes:
    path = mesh_dir / mesh
    bounds = get_mesh_bounds(path)
    if bounds:
        min_p, max_p = bounds
        center = [(min_p[i] + max_p[i]) / 2 for i in range(3)]
        size = [max_p[i] - min_p[i] for i in range(3)]
        print(f"| {mesh} | {' '.join(f'{c:.6f}' for c in center)} | {' '.join(f'{s:.6f}' for s in size)} |")
