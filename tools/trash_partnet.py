import xml.etree.ElementTree as ET
import copy
import trimesh

import os


def rename_links_and_joints(urdf_path, output_path):
    """
    Rename links and joints in the URDF:
    1. Change `link_XX` to `drawer_XX`.
    2. Change `joint_XX` to `drawer_XX_<type>_joint` based on the joint type.
    3. Rename links to include "main_frame".
    Handles missing <parent> or <child> gracefully.
    """
    # Parse the URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Step 1: Rename links
    for link in root.findall("link"):
        name = link.get("name")
        if name and name.startswith("link_"):
            # Generate the new name

            index = name.split("_")[-1].zfill(2)
            new_name = f"drawer_{index}"
            link.set("name", new_name)

    # # Step 2: Rename joints
    for joint in root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get(
            "type")  # Get the joint type (e.g., "revolute", "fixed")
        if name and name.startswith("joint_"):
            # Extract the index and rename based on the corresponding link
            link_index = name.split("_")[-1].zfill(2)
            new_name = f"drawer_{link_index}_{joint_type}_joint"
            joint.set("name", new_name)

            # Update parent/child links in the joint
            parent = joint.find("parent")
            child = joint.find("child")

            if parent is not None and parent.get("link").startswith("link_"):
                name = parent.get("link")

                index = name.split("_")[-1].zfill(2)
                new_name = f"drawer_{index}"
                parent.set("link", new_name)

            if child is not None and child.get("link").startswith("link_"):
                name = child.get("link")

                index = name.split("_")[-1].zfill(2)
                new_name = f"drawer_{index}"
                child.set("link", new_name)

    # Save the modified URDF
    tree.write(output_path, xml_declaration=True)
    print(f"Modified URDF saved to: {output_path}")


def generate_drawer_handle_pairs(urdf_path):
    """
    Dynamically generate drawer-handle pairs based on the number of links in the URDF.
    Args:
        urdf_path (str): Path to the input URDF file.
    Returns:
        list of tuples: List of (drawer_name, handle_name).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    drawer_handle_pairs = []
    for link in root.findall("link"):
        name = link.get("name")
        if name and name.startswith("drawer_"):
            index = name.split("_")[-1]  # Extract the index from the link name
            drawer_name = f"drawer_{index}"
            handle_name = f"drawer_{index}_handle"
            drawer_handle_pairs.append((drawer_name, handle_name))

    return drawer_handle_pairs


def merge_collision_meshes(drawer_name, handle_collision_meshes):
    # Determine the directory of the raw URDF file
    mesh_dir = os.path.dirname(raw_urdf_path)
    combined_mesh_dir = os.path.join(mesh_dir, "combined_mesh")
    os.makedirs(combined_mesh_dir, exist_ok=True)

    collision_mesh_buffer = []

    # Load each collision mesh and append it to the buffer
    for coll in handle_collision_meshes:
        mesh_element = coll.find("geometry").find("mesh")
        filename = mesh_element.get("filename")
        mesh_path = os.path.join(mesh_dir, filename)
        trimesh_mesh = trimesh.load(mesh_path)
        collision_mesh_buffer.append(trimesh_mesh)

    # Combine all meshes into a single mesh
    combined_mesh = trimesh.util.concatenate(collision_mesh_buffer)
    combined_mesh_filename = f"{drawer_name}_handle.stl"
    combined_mesh_path = os.path.join(combined_mesh_dir,
                                      combined_mesh_filename)

    # Export the combined mesh to the specified directory
    combined_mesh.export(combined_mesh_path)

    coll = handle_collision_meshes[0]
    mesh_element = coll.find("geometry").find("mesh")
    mesh_element.set("filename", f"combined_mesh/{combined_mesh_filename}")

    # Return the updated handle_collision_meshes
    return [coll]


def modify_urdf_for_drawers(raw_file_path, output_file_path):
    """
    Modify URDF to:
    1. Remove handle visual/collision meshes from drawers.
    2. Add a fixed joint connecting each drawer to its handle.
    3. Create individual links for handles.

    Args:
        raw_file_path (str): Path to the raw URDF file.
        output_file_path (str): Path to save the modified URDF.
    """
    # Generate drawer-handle pairs based on the number of links
    drawer_handle_pairs = generate_drawer_handle_pairs(raw_file_path)

    # Parse the raw URDF
    tree = ET.parse(raw_file_path)
    root = tree.getroot()

    for drawer_name, handle_name in drawer_handle_pairs:
        # Find and store handle visual/collision meshes and remove them from the drawer
        handle_visual_meshes = []
        handle_collision_meshes = []
        handle_visual_meshes_name = []

        for link in root.findall("link"):
            if link.get("name") == drawer_name:

                for visual in list(
                        link.findall("visual")
                ):  # Use list to avoid issues while removing elements
                    mesh = visual.find("geometry").find("mesh")

                    if mesh is not None and "handle" in visual.get("name"):
                        handle_visual_meshes.append(visual)
                        handle_visual_meshes_name.append(mesh.get("filename"))
                        link.remove(visual)

                for collision in list(link.findall("collision")):
                    mesh = collision.find("geometry").find("mesh")

                    if mesh is not None and mesh.get(
                            "filename") in handle_visual_meshes_name:
                        handle_collision_meshes.append(collision)

                        link.remove(collision)

                handle_collision_meshes = merge_collision_meshes(
                    drawer_name, handle_collision_meshes)

        # Create a new link for the handle
        link_index = drawer_name.split("_")[1]
        handle_link_name = f"{drawer_name.split('_')[0]}_{link_index}_handle"

        # Create a new link element for the handle
        handle_link = ET.Element("link", {"name": handle_link_name})

        for collision_elem in handle_visual_meshes:
            # Append the entire collision element
            collision = ET.SubElement(handle_link, "visual")
            for sub_elem in collision_elem:  # Copy all sub-elements

                collision.append(copy.deepcopy(sub_elem))

        for collision_elem in handle_collision_meshes:
            # Append the entire collision element
            collision = ET.SubElement(handle_link, "collision")
            for index, sub_elem in enumerate(
                    collision_elem):  # Copy all sub-elements

                collision.append(copy.deepcopy(sub_elem))

        # Add the handle link to the root
        root.append(handle_link)

        # Add a fixed joint between the drawer and the handle
        fixed_joint = ET.Element("joint", {
            "name": f"{drawer_name}_fixed_joint",
            "type": "fixed"
        })

        # Add <parent> and <child> with 'link' attributes
        ET.SubElement(fixed_joint, "parent", {"link": f"{drawer_name}"})
        ET.SubElement(fixed_joint, "child", {"link": f"{handle_name}"})

        # Add <origin> with 'xyz' and 'rpy' attributes
        ET.SubElement(
            fixed_joint,
            "origin",
            {
                "xyz": "0 0 0",  # Modify if the handle has an offset position
                "rpy": "0 0 0"  # Modify if the handle has an offset rotation
            })

        # Add the joint to the root
        root.append(fixed_joint)

    # Save the modified URDF
    tree.write(output_file_path, xml_declaration=True, encoding='utf-8')
    print(f"Modified URDF saved to: {output_file_path}")


# File paths
raw_urdf_path = "source/assets/kitchen/usd/oven01/raw_data/mobility.urdf"
output_urdf_path = "source/assets/kitchen/usd/oven01/raw_data/modified_mobility.urdf"

# Modify the URDF dynamically
rename_links_and_joints(raw_urdf_path, output_urdf_path)
modify_urdf_for_drawers(output_urdf_path, output_urdf_path)
