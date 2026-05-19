import xml.etree.ElementTree as ET
import numpy as np
import copy
import sys

def parse_vec_array(text):
    """Parses an X3D string of floats into a flat 1D numpy array."""
    if not text: return np.array([])
    return np.array([float(x) for x in text.replace(',', ' ').split() if x.strip()])

def parse_int_array(text):
    """Parses an X3D string of integers into a flat 1D numpy array."""
    if not text: return np.array([])
    return np.array([int(x) for x in text.replace(',', ' ').split() if x.strip()])

def format_array(arr, is_int=False):
    """Formats a numpy array back to an X3D compliant string."""
    if is_int:
        return " ".join(str(int(x)) for x in arr)
    else:
        # Use round to 5 decimal places to keep X3D file sizes reasonable
        return " ".join(f"{x:.5f}".rstrip('0').rstrip('.') for x in arr)

def point_to_segment_dist_vec(pts, a, b):
    """
    Vectorized calculation of shortest distance from points 'pts' to a line segment a->b.
    pts: (N, 3) numpy array
    a, b: (3,) numpy arrays
    """
    ab = b - a
    ap = pts - a
    # If bone is just a point (leaf bone or zero length)
    if np.allclose(ab, 0):
        return np.linalg.norm(ap, axis=1)
    
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    
    proj = a + np.outer(t, ab)
    return np.linalg.norm(pts - proj, axis=1)


class X3DAutoRigger:
    def __init__(self, skin_path, skel_path):
        self.skin_path = skin_path
        self.skel_path = skel_path
        self.skin_coords = None
        self.skin_faces = None
        self.skin_shape = None
        
        self.skel_tree = None
        self.hanim_root = None
        
        self.joints = [] # List of ET elements
        self.joint_world_positions = []
        self.joint_bone_segments = []

    def run(self, output_path):
        print("--- Phase 1: Parsing Files ---")
        self.load_skin()
        self.load_skeleton()
        
        print("\n--- Phase 2: Scale & Space Alignment ---")
        self.align_scales()
        
        print("\n--- Phase 3 & 4: Forward Kinematics & Auto-Rigging ---")
        self.compute_forward_kinematics()
        self.calculate_skin_weights()
        
        print("\n--- Phase 5: Merging & Exporting ---")
        self.merge_and_export(output_path)
        print(f"Success! Exported rigged humanoid to {output_path}")

    # ==========================================
    # PHASE 1: Parse & Extract
    # ==========================================
    def load_skin(self):
        tree = ET.parse(self.skin_path)
        root = tree.getroot()
        
        # Find IndexedFaceSet and Coordinate node
        for node in root.iter():
            tag = node.tag.split('}')[-1]
            if tag == 'IndexedFaceSet':
                self.skin_faces = node
                self.skin_shape = self._get_parent(root, node)
                
                # Extract coordinates
                for child in node:
                    if child.tag.split('}')[-1] == 'Coordinate':
                        pts = parse_vec_array(child.attrib.get('point', ''))
                        self.skin_coords = pts.reshape(-1, 3)
                        break
                break

        if self.skin_coords is None:
            raise ValueError("Could not find Coordinate node in the skin X3D.")
        print(f"Extracted Skin: {len(self.skin_coords)} vertices found.")

    def load_skeleton(self):
        self.skel_tree = ET.parse(self.skel_path)
        root = self.skel_tree.getroot()
        
        # Locate HAnimHumanoid
        for node in root.iter():
            if node.tag.split('}')[-1] == 'HAnimHumanoid':
                self.hanim_root = node
                break
                
        if self.hanim_root is None:
            raise ValueError("No HAnimHumanoid found in skeleton file.")

        # Strip Geometry from Skeleton
        self._strip_geometry(self.hanim_root)
        
        # Collect Joints
        for node in self.hanim_root.iter():
            if node.tag.split('}')[-1] == 'HAnimJoint':
                self.joints.append(node)
                
        print(f"Extracted Skeleton: {len(self.joints)} HAnimJoints found. Geometry stripped.")

    def _strip_geometry(self, node):
        tags_to_remove = {'Shape', 'Appearance', 'Material', 'ImageTexture', 'IndexedFaceSet', 'Coordinate', 'Normal'}
        for child in list(node):
            tag = child.tag.split('}')[-1]
            if tag in tags_to_remove:
                node.remove(child)
            else:
                self._strip_geometry(child)

    def _get_parent(self, tree_root, target_node):
        for parent in tree_root.iter():
            for child in parent:
                if child == target_node:
                    return parent
        return None

    # ==========================================
    # PHASE 2: Scale & Space Alignment
    # ==========================================
    def align_scales(self):
        # 1. Skin Bounding Box
        min_y_skin = np.min(self.skin_coords[:, 1])
        max_y_skin = np.max(self.skin_coords[:, 1])
        skin_height = max_y_skin - min_y_skin

        # 2. Skeleton Bounding Box (Approximate based on Joint Translations/Centers)
        skel_y_vals = []
        for j in self.joints:
            # HAnim uses center for world/body position bind pose frequently
            if 'center' in j.attrib:
                skel_y_vals.append(parse_vec_array(j.attrib['center'])[1])
            elif 'translation' in j.attrib:
                skel_y_vals.append(parse_vec_array(j.attrib['translation'])[1])
        
        if not skel_y_vals:
            skel_y_vals = [0, 1] # Fallback
            
        skel_height = max(skel_y_vals) - min(skel_y_vals)
        scale_factor = skin_height / skel_height if skel_height > 0 else 1.0
        
        print(f"Skin height: {skin_height:.3f}, Skel height: {skel_height:.3f}. Applying Scale Factor: {scale_factor:.3f}")

        # 3. Apply Scale to Skeleton
        for j in self.joints:
            for field in ['translation', 'center', 'scale']:
                if field in j.attrib:
                    val = parse_vec_array(j.attrib[field])
                    j.attrib[field] = format_array(val * scale_factor)

    # ==========================================
    # PHASE 3 & 4: Forward Kinematics & Rigging
    # ==========================================
    def compute_forward_kinematics(self):
        # Build hierarchy map to calculate absolute positions
        parent_map = {c: p for p in self.hanim_root.iter() for c in p}
        
        self.joint_world_positions = []
        
        for j in self.joints:
            # Reconstruct world position. For standard HAnim bind pose, 
            # 'center' often holds the body-space coordinate, or we accumulate translations.
            if 'center' in j.attrib and parse_vec_array(j.attrib['center']).any():
                world_pos = parse_vec_array(j.attrib['center'])
            else:
                # Accumulate translation up to root
                world_pos = np.array([0.0, 0.0, 0.0])
                curr = j
                while curr is not None and curr.tag.split('}')[-1] == 'HAnimJoint':
                    if 'translation' in curr.attrib:
                        world_pos += parse_vec_array(curr.attrib['translation'])
                    curr = parent_map.get(curr)
                    
            self.joint_world_positions.append(world_pos)
            
        # Determine Bone Line Segments (Joint to its first child)
        self.joint_bone_segments = []
        for i, j in enumerate(self.joints):
            pos_a = self.joint_world_positions[i]
            
            # Find first child joint
            child_joint = None
            for child in j:
                if child.tag.split('}')[-1] == 'HAnimJoint':
                    child_joint = child
                    break
            
            if child_joint is not None:
                child_idx = self.joints.index(child_joint)
                pos_b = self.joint_world_positions[child_idx]
            else:
                # Leaf joint: point bone
                pos_b = pos_a 
                
            self.joint_bone_segments.append((pos_a, pos_b))

    def calculate_skin_weights(self):
        num_verts = len(self.skin_coords)
        num_joints = len(self.joints)
        
        weights = np.zeros((num_verts, num_joints))
        
        # Calculate Bounding Box Diagonal for dynamic falloff radius
        bb_min = np.min(self.skin_coords, axis=0)
        bb_max = np.max(self.skin_coords, axis=0)
        body_height = bb_max[1] - bb_min[1]
        
        # Tuning: Envelope radius (15% of body height is a standard falloff starting point)
        envelope_radius = body_height * 0.15 
        
        # A. Calculate distances and quadratic falloff weight
        for j_idx in range(num_joints):
            pos_a, pos_b = self.joint_bone_segments[j_idx]
            
            # Vectorized point-to-line distance
            distances = point_to_segment_dist_vec(self.skin_coords, pos_a, pos_b)
            
            # Quadratic falloff: max(0, 1 - (dist / radius)^2)
            w = np.maximum(0, 1.0 - (distances / envelope_radius)**2)
            weights[:, j_idx] = w

        # B. Normalize and Prune to Top 4 Weights
        for i in range(num_verts):
            row = weights[i]
            total = np.sum(row)
            
            if total == 0:
                # Fallback: if vertex is completely outside all envelopes, bind to nearest joint
                pos_a_all = np.array([s[0] for s in self.joint_bone_segments])
                closest_joint = np.argmin(np.linalg.norm(pos_a_all - self.skin_coords[i], axis=1))
                weights[i, closest_joint] = 1.0
            else:
                # Keep top 4
                top_4_indices = np.argsort(row)[-4:]
                mask = np.ones(num_joints, dtype=bool)
                mask[top_4_indices] = False
                row[mask] = 0.0 # Zero out everything except top 4
                
                # Renormalize
                row_sum = np.sum(row)
                if row_sum > 0:
                    weights[i] = row / row_sum
                    
        # C. Invert the map: Assign weights to HAnimJoints
        for j_idx, joint in enumerate(self.joints):
            vert_weights = weights[:, j_idx]
            
            # Find vertices with weight > 0 for this joint
            active_verts = np.where(vert_weights > 0.001)[0]
            active_weights = vert_weights[active_verts]
            
            if len(active_verts) > 0:
                joint.attrib['skinCoordIndex'] = format_array(active_verts, is_int=True)
                joint.attrib['skinCoordWeight'] = format_array(active_weights, is_int=False)
    # ==========================================
    # PHASE 5: Validation, Merging & Cleanup
    # ==========================================
    def merge_and_export(self, output_path):
        # 1. Ensure HAnim Spec 2.0 (X3D 4.0 compliant)
        self.hanim_root.attrib['version'] = "2.0"
        
        # 2. Setup SkinCoord Node using containerField
        skin_coord_elem = ET.Element("Coordinate")
        skin_coord_elem.attrib['DEF'] = "SkinCoord"
        skin_coord_elem.attrib['containerField'] = "skinCoord"
        skin_coord_elem.attrib['point'] = format_array(self.skin_coords.flatten())
        
        self.hanim_root.append(skin_coord_elem)
        
        # 3. Update IndexedFaceSet to USE the shared Coordinate
        for child in list(self.skin_faces):
            if child.tag.split('}')[-1] == 'Coordinate':
                self.skin_faces.remove(child)
                
        use_coord = ET.Element("Coordinate")
        use_coord.attrib['USE'] = "SkinCoord"
        self.skin_faces.append(use_coord)
        
        # 4. Attach Skin Shape into HAnimHumanoid using containerField
        if self.skin_shape is not None and self.skin_shape.tag.split('}')[-1] == 'Shape':
            skin_shape_elem = copy.deepcopy(self.skin_shape)
        else:
            skin_shape_elem = ET.Element("Shape")
            skin_shape_elem.append(copy.deepcopy(self.skin_faces))
            
        skin_shape_elem.attrib['containerField'] = "skin"
        self.hanim_root.append(skin_shape_elem)
        
        # 5. Populate Flat Arrays (joints, segments, sites) using containerField
        for tag_name in ['HAnimJoint', 'HAnimSegment', 'HAnimSite']:
            container_field_name = tag_name.replace('HAnim', '').lower() + "s"
            
            # ONLY grab actual definition nodes, skip existing USE references!
            found_nodes = [n for n in self.hanim_root.iter(tag_name) if 'USE' not in n.attrib]
            
            for node in found_nodes:
                if 'DEF' not in node.attrib:
                    # Auto-generate DEF if missing
                    node.attrib['DEF'] = node.attrib.get('name', f"{tag_name}_{np.random.randint(1000,9999)}")
                
                use_ref = ET.Element(tag_name)
                use_ref.attrib['USE'] = node.attrib['DEF']
                use_ref.attrib['containerField'] = container_field_name
                
                self.hanim_root.append(use_ref)

        # 6. Save Final XML
        new_tree_root = ET.Element("X3D", {"profile": "Immersive", "version": "4.0"})
        scene = ET.Element("Scene")
        scene.append(self.hanim_root)
        new_tree_root.append(scene)
        
        # Run the final cleanup routine before writing
        self._cleanup_x3d_tree(new_tree_root)
        
        final_tree = ET.ElementTree(new_tree_root)
        ET.indent(final_tree, space="  ", level=0)
        final_tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def _cleanup_x3d_tree(self, root_node):
        """
        Cleans the X3D tree of mutually exclusive DEF/USE conflicts 
        and removes any orphaned USE nodes (especially Transforms).
        """
        # Pass 1: Resolve any node that has BOTH DEF and USE
        for el in root_node.iter():
            if 'DEF' in el.attrib and 'USE' in el.attrib:
                # If element has children, it's functioning as a definition. Drop USE.
                # If it's empty, it's functioning as a reference. Drop DEF.
                if len(list(el)) > 0:
                    del el.attrib['USE']
                else:
                    del el.attrib['DEF']

        # Pass 2: Catalog all valid DEF declarations currently in the tree
        valid_defs = {el.attrib['DEF'] for el in root_node.iter() if 'DEF' in el.attrib}

        # Pass 3: Recursively find and eliminate orphaned USE nodes
        def remove_dead_uses(parent):
            for child in list(parent): # iterate over a copy of the list to allow safe deletion
                if 'USE' in child.attrib:
                    use_val = child.attrib['USE']
                    # If the DEF this points to doesn't exist, delete this node
                    if use_val not in valid_defs:
                        parent.remove(child)
                        continue 
                
                # Continue recursively down the tree
                remove_dead_uses(child)

        # Kick off recursive orphan removal
        remove_dead_uses(root_node)

if __name__ == "__main__":
    # Example usage: 
    # python x3d_autorigger.py skin.x3d skeleton.x3d output_rigged.x3d
    
    if len(sys.argv) < 4:
        print("Usage: python x3d_autorigger.py <skin_mesh.x3d> <skeleton.x3d> <output.x3d>")
        sys.exit(1)
        
    skin_file = sys.argv[1]
    skel_file = sys.argv[2]
    out_file = sys.argv[3]
    
    rigger = X3DAutoRigger(skin_file, skel_file)
    rigger.run(out_file)
