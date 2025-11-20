from plyfile import PlyData


def inspect_ply(file_path):
    # Load the PLY file
    ply_data = PlyData.read(file_path)

    # Print header information
    print("PLY Header:", ply_data)

    # List all elements (vertices, faces, etc.)
    print("\nElements in the PLY file:")
    import pdb
    pdb.set_trace()
    for element in ply_data.elements:
        print(f"- {element.name} ({len(element.data)} entries)")

    # Show properties of the first element (usually "vertex" in Gaussian splatting)
    if "vertex" in ply_data:
        print("\nVertex Properties:")
        for prop in ply_data["vertex"].data.dtype.names:
            print(f"- {prop}")

    # Preview first 5 entries
    print("\nFirst 5 vertices (example):")
    print(ply_data["vertex"].data[:5])


# Example Usage
inspect_ply("/home/ensu/Downloads/2_3_2025.ply")
