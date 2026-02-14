#!/usr/bin/env python3
"""
Compute METIS-based component partitions for the pepe36 mesh.

This script:
1. Loads the original mesh connectivity (which interfaces each tag has)
2. Groups consecutive tags into components (tag 2*i, 2*i+1 -> component i)
3. Builds a connectivity graph between components
4. Uses METIS to partition the component graph
5. Saves cell-to-partition mapping for use during simulation

Usage:
    python compute_component_partitions.py <num_partitions> [output_file]

Example:
    python compute_component_partitions.py 4
    python compute_component_partitions.py 8 data/pepe36_partitions_8.pickle
"""

import pickle
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

try:
    import pymetis
    HAS_PYMETIS = True
except ImportError:
    HAS_PYMETIS = False
    print("Warning: pymetis not installed. Install with: pip install pymetis")


def load_tag_interfaces(pickle_file):
    """Load tag interface data from pickle file."""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def build_component_connectivity(tag_interfaces, num_components):
    """Build connectivity graph between components.

    A component consists of two consecutive tags: (2*i, 2*i+1).
    Components are connected if they share an interface.
    """
    # Get interfaces for each component
    component_interfaces = {}
    for c in range(num_components):
        ecs_tag = 2 * c
        cell_tag = 2 * c + 1
        ecs_ifaces = set(int(x) for x in tag_interfaces.get(ecs_tag, set()) if int(x) != ecs_tag)
        cell_ifaces = set(int(x) for x in tag_interfaces.get(cell_tag, set()) if int(x) != cell_tag)
        component_interfaces[c] = ecs_ifaces | cell_ifaces

    # Find which interface IDs are shared between components
    interface_to_components = defaultdict(set)
    for c, ifaces in component_interfaces.items():
        for iface in ifaces:
            interface_to_components[iface].add(c)

    # Build adjacency list
    adjacency = [[] for _ in range(num_components)]
    for iface, components in interface_to_components.items():
        if len(components) == 2:
            c1, c2 = list(components)
            if c2 not in adjacency[c1]:
                adjacency[c1].append(c2)
            if c1 not in adjacency[c2]:
                adjacency[c2].append(c1)

    return adjacency


def partition_components_metis(adjacency, num_partitions):
    """Use METIS to partition the component graph."""
    if not HAS_PYMETIS:
        raise RuntimeError("pymetis is required for METIS partitioning")

    n_cuts, membership = pymetis.part_graph(num_partitions, adjacency=adjacency)
    return membership, n_cuts


def partition_components_simple(adjacency, num_partitions):
    """Simple round-robin partitioning (fallback if METIS unavailable)."""
    num_components = len(adjacency)
    return [i % num_partitions for i in range(num_components)], 0


def create_cell_partition_mapping(tag_interfaces_file, num_partitions, use_metis=True):
    """Create mapping from original mesh tags to partition numbers.

    Returns:
        component_to_partition: dict mapping component_id -> partition_id
        tag_to_partition: dict mapping original_tag -> partition_id
        stats: dict with partitioning statistics
    """
    # Load interface data
    tag_interfaces = load_tag_interfaces(tag_interfaces_file)

    # Determine number of components (consecutive tag pairs)
    num_tags = len(tag_interfaces)
    num_components = num_tags // 2

    print(f"Tags: {num_tags}, Components: {num_components}, Partitions: {num_partitions}")

    # Build connectivity graph
    adjacency = build_component_connectivity(tag_interfaces, num_components)

    # Count edges
    num_edges = sum(len(adj) for adj in adjacency) // 2
    print(f"Component graph: {num_components} nodes, {num_edges} edges")

    # Partition
    if use_metis and HAS_PYMETIS:
        membership, n_cuts = partition_components_metis(adjacency, num_partitions)
        print(f"METIS partitioning: {n_cuts} edge cuts")
    else:
        membership, n_cuts = partition_components_simple(adjacency, num_partitions)
        print("Using simple round-robin partitioning")

    # Create mappings
    component_to_partition = {c: membership[c] for c in range(num_components)}
    tag_to_partition = {}
    for c in range(num_components):
        tag_to_partition[2 * c] = membership[c]      # ECS tag
        tag_to_partition[2 * c + 1] = membership[c]  # Cell tag

    # Compute statistics
    partition_sizes = defaultdict(int)
    for part in membership:
        partition_sizes[part] += 1

    stats = {
        'num_tags': num_tags,
        'num_components': num_components,
        'num_partitions': num_partitions,
        'edge_cuts': n_cuts,
        'components_per_partition': dict(partition_sizes),
        'method': 'metis' if (use_metis and HAS_PYMETIS) else 'round_robin',
        'adjacency': adjacency,  # Save for visualization
    }

    return component_to_partition, tag_to_partition, stats


def main():
    parser = argparse.ArgumentParser(description='Compute component-based mesh partitions')
    parser.add_argument('num_partitions', type=int, help='Number of partitions (MPI ranks)')
    parser.add_argument('--input', '-i', default='data/pepe36.pickle',
                        help='Input pickle file with tag interfaces')
    parser.add_argument('--output', '-o', default=None,
                        help='Output pickle file (default: data/pepe36_component_partitions_N.pickle)')
    parser.add_argument('--no-metis', action='store_true',
                        help='Use simple round-robin instead of METIS')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'data/pepe36_component_partitions_{args.num_partitions}.pickle'

    # Compute partitions
    component_to_partition, tag_to_partition, stats = create_cell_partition_mapping(
        args.input, args.num_partitions, use_metis=not args.no_metis
    )

    # Print summary
    print(f"\nPartition balance (components per partition):")
    for p in range(args.num_partitions):
        count = stats['components_per_partition'].get(p, 0)
        print(f"  Partition {p}: {count} components ({count * 2} tags)")

    # Save
    output_data = {
        'component_to_partition': component_to_partition,
        'tag_to_partition': tag_to_partition,
        'stats': stats,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
