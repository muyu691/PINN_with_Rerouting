from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

try:
    from network_registry import NetworkSpec
except ModuleNotFoundError:
    from .network_registry import NetworkSpec


_META_PATTERN = re.compile(r"<([^>]+)>\s*(.*)")
_ORIGIN_PATTERN = re.compile(r"Origin\s+(\d+)", flags=re.IGNORECASE)
_OD_ENTRY_PATTERN = re.compile(r"(\d+)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


@dataclass(frozen=True)
class TrafficNetworkData:
    network_name: str
    graph: nx.DiGraph
    node_ids: Tuple[int, ...]
    centroid_nodes: Tuple[int, ...]
    metadata: Dict[str, str]
    od_matrix: Optional[np.ndarray]
    node_id_offset: int
    dataset_root: str
    network_file: str
    od_file: str

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def od_dim(self) -> int:
        return 0 if self.od_matrix is None else int(self.od_matrix.shape[0])


def _parse_metadata(lines) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in lines:
        match = _META_PATTERN.match(line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        metadata[key] = match.group(2).strip()
        if key == "end_of_metadata":
            break
    return metadata


def _split_data_section(lines) -> Tuple[Dict[str, str], list[str]]:
    metadata = _parse_metadata(lines)
    data_start = 0
    for idx, line in enumerate(lines):
        if "<END OF METADATA>" in line.upper():
            data_start = idx + 1
            break
    return metadata, lines[data_start:]


def parse_tntp_network(network_file: str, node_id_offset: int = 1) -> tuple[nx.DiGraph, Dict[str, str]]:
    """Parse a TNTP network file into a directed graph."""
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Network file not found: {network_file}")

    with open(network_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    metadata, data_lines = _split_data_section(lines)
    graph = nx.DiGraph()

    num_nodes_meta = int(metadata.get("number_of_nodes", "0") or 0)
    if num_nodes_meta > 0:
        graph.add_nodes_from(range(node_id_offset, node_id_offset + num_nodes_meta))

    for raw_line in data_lines:
        line = raw_line.strip()
        if not line or line.startswith("~") or line.startswith("<"):
            continue
        line = line.replace(";", " ")
        tokens = re.split(r"\s+", line)
        if len(tokens) < 5:
            continue

        init_node = int(float(tokens[0]))
        term_node = int(float(tokens[1]))
        capacity = float(tokens[2])
        length = float(tokens[3])
        free_flow_time = float(tokens[4])

        speed = None
        if len(tokens) > 7:
            try:
                speed = float(tokens[7])
            except ValueError:
                speed = None
        if speed is None or speed <= 0.0:
            speed = (length / free_flow_time) * 60.0 if free_flow_time > 0 else 1.0

        graph.add_edge(
            init_node,
            term_node,
            capacity=capacity,
            length=length,
            free_flow_time=free_flow_time,
            speed=speed,
        )

    return graph, metadata


def parse_tntp_trips(trips_file: str) -> np.ndarray:
    """Parse a TNTP trips file into a dense OD matrix."""
    if not trips_file:
        raise FileNotFoundError("Empty trips file path provided.")
    if not os.path.exists(trips_file):
        raise FileNotFoundError(f"Trips file not found: {trips_file}")

    with open(trips_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    metadata, data_lines = _split_data_section(lines)
    num_zones = int(metadata.get("number_of_zones", "0") or 0)

    entries: Dict[Tuple[int, int], float] = {}
    current_origin: Optional[int] = None
    max_node_id = 0

    for raw_line in data_lines:
        line = raw_line.strip()
        if not line:
            continue

        origin_match = _ORIGIN_PATTERN.match(line)
        if origin_match:
            current_origin = int(origin_match.group(1))
            max_node_id = max(max_node_id, current_origin)
            continue

        if current_origin is None:
            continue

        for destination, demand in _OD_ENTRY_PATTERN.findall(line):
            dest = int(destination)
            entries[(current_origin, dest)] = float(demand)
            max_node_id = max(max_node_id, dest)

    matrix_dim = max(num_zones, max_node_id)
    if matrix_dim <= 0:
        raise ValueError(f"Failed to parse OD entries from trips file: {trips_file}")

    od_matrix = np.zeros((matrix_dim, matrix_dim), dtype=np.float64)
    for (origin, destination), demand in entries.items():
        od_matrix[origin - 1, destination - 1] = demand
    return od_matrix


def _infer_centroid_nodes(
    spec: NetworkSpec,
    node_ids: Tuple[int, ...],
    metadata: Dict[str, str],
    od_matrix: Optional[np.ndarray],
) -> Tuple[int, ...]:
    if spec.centroid_nodes is not None:
        return tuple(spec.centroid_nodes)

    if od_matrix is not None:
        return tuple(node_ids[: int(od_matrix.shape[0])])

    first_thru_node = int(metadata.get("first_thru_node", "0") or 0)
    if first_thru_node > spec.node_id_offset:
        return tuple(node_id for node_id in node_ids if node_id < first_thru_node)

    number_of_zones = int(metadata.get("number_of_zones", "0") or 0)
    if number_of_zones > 0:
        return tuple(node_ids[: min(number_of_zones, len(node_ids))])

    return tuple()


def load_network_data(spec: NetworkSpec) -> TrafficNetworkData:
    """Load network and optional OD metadata from a resolved network spec."""
    parser = spec.parser.lower()
    if parser != "tntp":
        raise ValueError(f"Unsupported parser='{spec.parser}'. Only 'tntp' is implemented.")

    graph, metadata = parse_tntp_network(spec.network_file, node_id_offset=spec.node_id_offset)
    node_ids = tuple(sorted(graph.nodes()))

    od_matrix = None
    if spec.od_file:
        if os.path.exists(spec.od_file):
            od_matrix = parse_tntp_trips(spec.od_file)
        elif spec.demand_source == "trips":
            raise FileNotFoundError(f"Trips file not found: {spec.od_file}")

    centroid_nodes = _infer_centroid_nodes(spec, node_ids, metadata, od_matrix)

    return TrafficNetworkData(
        network_name=spec.network_name,
        graph=graph,
        node_ids=node_ids,
        centroid_nodes=centroid_nodes,
        metadata=metadata,
        od_matrix=od_matrix,
        node_id_offset=spec.node_id_offset,
        dataset_root=spec.dataset_root,
        network_file=spec.network_file,
        od_file=spec.od_file,
    )
