"""
Converts mesh from pts/elem format to XDMF format for use with cardioEMI.

pts format:
    - First line: number of points
    - Subsequent lines: x y z (3D coordinates)

elem format:
    - First line: number of elements
    - Subsequent lines: Tt v1 v2 v3 v4 tag (tetrahedral elements with subdomain tag)

The script generates:
    - An XDMF file with mesh, cell_tags, and facet_tags
    - An HDF5 file with the actual data
    - A pickle file with the connectivity dictionary for cardioEMI

Usage:
    python convert_pts_elem.py <pts_file> <elem_file> <output_prefix>
    python convert_pts_elem.py <pts_file> <elem_file> <output_prefix> --color-intracellular

Options:
    --color-intracellular: Map even tags to 0 (ECS), then use graph coloring
                           to assign minimal tags to odd (intracellular) tags
                           such that no neighboring cells share the same tag.

Example:
    python convert_pts_elem.py pepe36-sep-domi.pts pepe36-sep-domi.elem pepe36
    python convert_pts_elem.py pepe36-sep-domi.pts pepe36-sep-domi.elem pepe36 --color-intracellular
"""

import numpy as np
import h5py
from lxml import etree
from collections import defaultdict
import pickle
import argparse
import os


def read_pts_file(filename):
    """Read point coordinates from pts file."""
    with open(filename, 'r') as f:
        num_points = int(f.readline().strip())
        points = np.zeros((num_points, 3), dtype=np.float64)
        for i in range(num_points):
            line = f.readline().strip()
            coords = line.split()
            points[i] = [float(coords[0]), float(coords[1]), float(coords[2])]
    print(f"Read {num_points} points from {filename}")
    return points


def read_elem_file(filename):
    """Read tetrahedral elements from elem file."""
    with open(filename, 'r') as f:
        num_elements = int(f.readline().strip())
        topology = np.zeros((num_elements, 4), dtype=np.int64)
        cell_tags = np.zeros(num_elements, dtype=np.int32)
        for i in range(num_elements):
            line = f.readline().strip().split()
            # Format: Tt v1 v2 v3 v4 tag
            topology[i] = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
            cell_tags[i] = int(line[5])
    print(f"Read {num_elements} tetrahedral elements from {filename}")
    return topology, cell_tags


def remap_tags(cell_tags):
    """
    Remap subdomain tags to start from 0 (ECS) and be contiguous.
    Returns remapped tags and the mapping dictionary.
    """
    unique_tags = np.unique(cell_tags)
    print(f"Found {len(unique_tags)} unique subdomain tags: {unique_tags[:10]}..."
          if len(unique_tags) > 10 else f"Found {len(unique_tags)} unique subdomain tags: {unique_tags}")

    # Create mapping: original tag -> new tag (starting from 0)
    tag_mapping = {old_tag: new_tag for new_tag, old_tag in enumerate(sorted(unique_tags))}

    # Apply remapping
    remapped_tags = np.array([tag_mapping[t] for t in cell_tags], dtype=np.int32)

    print(f"Remapped tags to range 0-{len(unique_tags)-1}")
    return remapped_tags, tag_mapping


def build_subdomain_adjacency(topology, cell_tags):
    """
    Build adjacency graph between subdomain tags based on shared facets.

    Returns:
        adjacency: dict mapping tag -> set of neighboring tags
    """
    print("Building subdomain adjacency graph...")

    # Dictionary: face tuple -> list of cell tags
    face_to_tags = defaultdict(list)

    # Face indices for a tetrahedron
    face_indices = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]

    for cell_id, tetra in enumerate(topology):
        if cell_id % 100000 == 0:
            print(f"  Processing cell {cell_id}/{len(topology)}...")
        tag = cell_tags[cell_id]
        for i, j, k in face_indices:
            face = tuple(sorted([tetra[i], tetra[j], tetra[k]]))
            face_to_tags[face].append(tag)

    # Build adjacency from shared faces
    adjacency = defaultdict(set)
    for face, tags in face_to_tags.items():
        if len(tags) == 2 and tags[0] != tags[1]:
            adjacency[tags[0]].add(tags[1])
            adjacency[tags[1]].add(tags[0])

    print(f"  Found {len(adjacency)} subdomain tags with neighbors")
    return dict(adjacency)


def graph_coloring(adjacency, tags_to_color):
    """
    Greedy graph coloring algorithm.

    Args:
        adjacency: dict mapping tag -> set of neighboring tags
        tags_to_color: list of tags that need to be colored

    Returns:
        coloring: dict mapping original tag -> color (1, 2, 3, ...)
    """
    coloring = {}

    # Sort tags for deterministic behavior
    for tag in sorted(tags_to_color):
        # Find colors used by neighbors
        neighbor_colors = set()
        if tag in adjacency:
            for neighbor in adjacency[tag]:
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])

        # Assign the smallest available color (starting from 1, 0 is reserved for ECS)
        color = 1
        while color in neighbor_colors:
            color += 1
        coloring[tag] = color

    return coloring


def remap_tags_with_coloring(cell_tags, topology):
    """
    Remap subdomain tags using graph coloring:
    1. Even tags -> 0 (extracellular space)
    2. Odd tags -> minimal colors (1, 2, 3, ...) via graph coloring

    Returns remapped tags and the mapping dictionary.
    """
    unique_tags = np.unique(cell_tags)
    print(f"Found {len(unique_tags)} unique subdomain tags")

    # Separate even and odd tags
    even_tags = [t for t in unique_tags if t % 2 == 0]
    odd_tags = [t for t in unique_tags if t % 2 == 1]

    print(f"  Even tags (ECS): {len(even_tags)} -> will be mapped to 0")
    print(f"  Odd tags (intracellular): {len(odd_tags)} -> will be graph-colored")

    # Build adjacency graph between original tags
    adjacency = build_subdomain_adjacency(topology, cell_tags)

    # Filter adjacency to only include odd tags (intracellular)
    # We only care about adjacency between intracellular cells
    odd_set = set(odd_tags)
    intra_adjacency = {}
    for tag in odd_tags:
        if tag in adjacency:
            # Only keep neighbors that are also odd (intracellular)
            intra_neighbors = adjacency[tag] & odd_set
            if intra_neighbors:
                intra_adjacency[tag] = intra_neighbors

    print(f"  Intracellular adjacency: {len(intra_adjacency)} cells have intracellular neighbors")

    # Apply graph coloring to odd tags
    coloring = graph_coloring(intra_adjacency, odd_tags)

    num_colors = max(coloring.values()) if coloring else 0
    print(f"  Graph coloring used {num_colors} colors for intracellular cells")

    # Create full mapping
    tag_mapping = {}
    for t in even_tags:
        tag_mapping[t] = 0  # ECS
    for t in odd_tags:
        tag_mapping[t] = coloring[t]  # Colored intracellular

    # Apply remapping
    remapped_tags = np.array([tag_mapping[t] for t in cell_tags], dtype=np.int32)

    # Count cells per new tag
    new_unique, counts = np.unique(remapped_tags, return_counts=True)
    print(f"  Final tag distribution:")
    for tag, count in zip(new_unique, counts):
        label = "ECS" if tag == 0 else f"Intra-{tag}"
        print(f"    Tag {tag} ({label}): {count} cells")

    return remapped_tags, tag_mapping


def extract_facets_and_membrane_dict(topology, cell_tags):
    """
    Extract facets from tetrahedra and create membrane connectivity dictionary.

    Returns:
        facet_topology: (num_facets, 3) array of vertex indices for each facet
        facet_tags: (num_facets,) array of facet tags
        membrane_dict: dictionary mapping subdomain tag -> set of membrane tags
    """
    print("Extracting facets from tetrahedra...")

    num_tags = len(np.unique(cell_tags))

    # Dictionary: face tuple -> list of (cell_id, cell_tag)
    face_to_cells = defaultdict(list)

    # Each tetrahedron has 4 faces (triangles)
    # Face indices for a tetrahedron with vertices [0,1,2,3]:
    face_indices = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]

    for cell_id, tetra in enumerate(topology):
        if cell_id % 100000 == 0:
            print(f"  Processing cell {cell_id}/{len(topology)}...")
        for i, j, k in face_indices:
            face = tuple(sorted([tetra[i], tetra[j], tetra[k]]))
            face_to_cells[face].append((cell_id, cell_tags[cell_id]))

    print(f"  Found {len(face_to_cells)} unique facets")

    # Build facet topology and tags
    facet_topology = []
    facet_tags = []
    membrane_dict = defaultdict(set)

    DEFAULT_TAG = -5  # Interior facets within same subdomain

    for face, cells in face_to_cells.items():
        facet_topology.append(face)

        if len(cells) == 1:
            # Boundary facet
            facet_tags.append(DEFAULT_TAG)
        elif len(cells) == 2:
            tag1, tag2 = cells[0][1], cells[1][1]
            if tag1 == tag2:
                # Interior facet within same subdomain
                facet_tags.append(DEFAULT_TAG)
            else:
                # Membrane facet between different subdomains
                # Use encoding: min_tag * (N_TAGS + 1) + max_tag
                membrane_tag = min(tag1, tag2) * (num_tags + 1) + max(tag1, tag2)
                facet_tags.append(membrane_tag)
                membrane_dict[tag1].add(membrane_tag)
                membrane_dict[tag2].add(membrane_tag)
        else:
            # Should not happen in valid mesh
            facet_tags.append(DEFAULT_TAG)

    facet_topology = np.array(facet_topology, dtype=np.int64)
    facet_tags = np.array(facet_tags, dtype=np.int32)

    print(f"  Created membrane dictionary with {len(membrane_dict)} entries")
    num_membrane_facets = np.sum(facet_tags != DEFAULT_TAG)
    print(f"  Found {num_membrane_facets} membrane facets")

    return facet_topology, facet_tags, dict(membrane_dict)


def write_xdmf_h5(points, topology, cell_tags, facet_topology, facet_tags,
                  xdmf_file, h5_file):
    """Write mesh data to XDMF and HDF5 files."""

    print(f"Writing mesh to {xdmf_file} and {h5_file}...")

    # Namespace for xi:include
    xi_ns = "https://www.w3.org/2001/XInclude"
    nsmap = {"xi": xi_ns}

    # Write the HDF5 file
    with h5py.File(h5_file, "w") as h5:
        mesh_group = h5.create_group("Mesh")
        mesh_group.create_dataset("mesh/geometry", data=points)
        mesh_group.create_dataset("mesh/topology", data=topology)
        mesh_group.create_dataset("facet_tags/topology", data=facet_topology)
        mesh_group.create_dataset("facet_tags/Values", data=facet_tags)
        mesh_group.create_dataset("cell_tags/topology", data=topology)
        mesh_group.create_dataset("cell_tags/Values", data=cell_tags)

    # Get just the filename for XDMF references
    h5_basename = os.path.basename(h5_file)

    # Write the XDMF file
    root = etree.Element("Xdmf", Version="3.0", nsmap=nsmap)
    domain = etree.SubElement(root, "Domain")

    # Mesh Grid
    grid_mesh = etree.SubElement(domain, "Grid", Name="mesh", GridType="Uniform")
    topology_mesh = etree.SubElement(grid_mesh, "Topology", TopologyType="Tetrahedron",
                                      NumberOfElements=str(topology.shape[0]))
    etree.SubElement(topology_mesh, "DataItem",
                     Dimensions=f"{topology.shape[0]} 4",
                     NumberType="Int", Format="HDF").text = f"{h5_basename}:/Mesh/mesh/topology"

    geometry_mesh = etree.SubElement(grid_mesh, "Geometry", GeometryType="XYZ")
    etree.SubElement(geometry_mesh, "DataItem",
                     Dimensions=f"{points.shape[0]} 3",
                     Format="HDF").text = f"{h5_basename}:/Mesh/mesh/geometry"

    # Facet Tags Grid
    grid_facet = etree.SubElement(domain, "Grid", Name="facet_tags", GridType="Uniform")
    xi_include_geom = etree.Element(f"{{{xi_ns}}}include",
                                     xpointer="xpointer(/Xdmf/Domain/Grid/Geometry)")
    grid_facet.append(xi_include_geom)
    topology_facet = etree.SubElement(grid_facet, "Topology", TopologyType="Triangle",
                                       NumberOfElements=str(facet_topology.shape[0]))
    etree.SubElement(topology_facet, "DataItem",
                     Dimensions=f"{facet_topology.shape[0]} 3",
                     NumberType="Int", Format="HDF").text = f"{h5_basename}:/Mesh/facet_tags/topology"

    attribute_facet = etree.SubElement(grid_facet, "Attribute", Name="facet_tags",
                                       AttributeType="Scalar", Center="Cell")
    etree.SubElement(attribute_facet, "DataItem",
                     Dimensions=f"{facet_tags.shape[0]}",
                     Format="HDF").text = f"{h5_basename}:/Mesh/facet_tags/Values"

    # Cell Tags Grid
    grid_cell = etree.SubElement(domain, "Grid", Name="cell_tags", GridType="Uniform")
    xi_include_geom = etree.Element(f"{{{xi_ns}}}include",
                                     xpointer="xpointer(/Xdmf/Domain/Grid/Geometry)")
    grid_cell.append(xi_include_geom)
    topology_cell = etree.SubElement(grid_cell, "Topology", TopologyType="Tetrahedron",
                                      NumberOfElements=str(topology.shape[0]))
    etree.SubElement(topology_cell, "DataItem",
                     Dimensions=f"{topology.shape[0]} 4",
                     NumberType="Int", Format="HDF").text = f"{h5_basename}:/Mesh/cell_tags/topology"

    attribute_cell = etree.SubElement(grid_cell, "Attribute", Name="cell_tags",
                                       AttributeType="Scalar", Center="Cell")
    etree.SubElement(attribute_cell, "DataItem",
                     Dimensions=f"{cell_tags.shape[0]}",
                     Format="HDF").text = f"{h5_basename}:/Mesh/cell_tags/Values"

    # Save the XDMF file
    tree = etree.ElementTree(root)
    tree.write(xdmf_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    print(f"  Written {xdmf_file} and {h5_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pts/elem mesh format to XDMF for cardioEMI")
    parser.add_argument("pts_file", help="Input pts file with point coordinates")
    parser.add_argument("elem_file", help="Input elem file with tetrahedral elements")
    parser.add_argument("output_prefix", help="Output prefix for XDMF, H5, and pickle files")
    parser.add_argument("--no-remap", action="store_true",
                        help="Do not remap subdomain tags (keep original values)")
    parser.add_argument("--color-intracellular", action="store_true",
                        help="Map even tags to 0 (ECS), use graph coloring for odd tags")

    args = parser.parse_args()

    # Read input files
    points = read_pts_file(args.pts_file)
    topology, cell_tags = read_elem_file(args.elem_file)

    # Remap tags based on selected mode
    if args.color_intracellular:
        print("\nUsing graph coloring mode:")
        print("  - Even tags -> 0 (extracellular)")
        print("  - Odd tags -> graph-colored (intracellular)\n")
        cell_tags, tag_mapping = remap_tags_with_coloring(cell_tags, topology)
        # Save tag mapping for reference
        mapping_file = args.output_prefix + "_tag_mapping.pickle"
        with open(mapping_file, "wb") as f:
            pickle.dump(tag_mapping, f)
        print(f"Saved tag mapping to {mapping_file}")
    elif not args.no_remap:
        cell_tags, tag_mapping = remap_tags(cell_tags)
        # Save tag mapping for reference
        mapping_file = args.output_prefix + "_tag_mapping.pickle"
        with open(mapping_file, "wb") as f:
            pickle.dump(tag_mapping, f)
        print(f"Saved tag mapping to {mapping_file}")

    # Extract facets and create membrane dictionary
    facet_topology, facet_tags, membrane_dict = extract_facets_and_membrane_dict(
        topology, cell_tags)

    # Write output files
    xdmf_file = args.output_prefix + ".xdmf"
    h5_file = args.output_prefix + ".h5"
    pickle_file = args.output_prefix + ".pickle"

    write_xdmf_h5(points, topology, cell_tags, facet_topology, facet_tags,
                  xdmf_file, h5_file)

    # Save membrane connectivity dictionary
    with open(pickle_file, "wb") as f:
        pickle.dump(membrane_dict, f)
    print(f"Saved membrane connectivity dictionary to {pickle_file}")

    print("\nConversion complete!")
    print(f"  Mesh: {xdmf_file}")
    print(f"  Data: {h5_file}")
    print(f"  Connectivity: {pickle_file}")
    print(f"\nTo use with cardioEMI, update input.yml:")
    print(f"  mesh_file: \"{xdmf_file}\"")
    print(f"  tags_dictionary_file: \"{pickle_file}\"")


if __name__ == "__main__":
    main()
