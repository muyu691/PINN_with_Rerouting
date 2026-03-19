from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import os


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class MutationPolicy:
    """Network-size aware mutation policy.

    The default ratios are calibrated so that Sioux Falls reproduces the
    previous absolute ranges:
      - top_k_range:    5-10 over 24 nodes
      - delete_range:   5-10 over 76 edges
      - add edges/node: 1-3
    """

    top_k_ratio_range: Tuple[float, float] = (5 / 24, 10 / 24)
    edges_per_node_range: Tuple[int, int] = (1, 3)
    delete_ratio_range: Tuple[float, float] = (5 / 76, 10 / 76)
    cap_scale_range: Tuple[float, float] = (0.3, 2.0)
    spd_scale_range: Tuple[float, float] = (0.3, 2.0)


@dataclass(frozen=True)
class NetworkSpec:
    """Resolved network specification used by the data pipeline."""

    network_name: str
    dataset_root: str
    network_file: str
    od_file: str = ""
    parser: str = "tntp"
    node_id_offset: int = 1
    demand_source: str = "lhs"
    centroid_nodes: Optional[Tuple[int, ...]] = None
    mutation_policy: MutationPolicy = MutationPolicy()


_BUILTIN_SPECS = {
    "siouxfalls": {
        "network_name": "SiouxFalls",
        "dataset_root": "../sioux_data",
        "network_file": "SiouxFalls_net.tntp",
        "od_file": "",
        "centroid_nodes": tuple(range(1, 12)),
    },
    "ema": {
        "network_name": "EMA",
        "dataset_root": "../ema_data",
        "network_file": "EMA_net.tntp",
        "od_file": "EMA_trips.tntp",
        "centroid_nodes": None,
    },
    "anaheim": {
        "network_name": "Anaheim",
        "dataset_root": "../anaheim_data",
        "network_file": "Anaheim_net.tntp",
        "od_file": "Anaheim_trips.tntp",
        "centroid_nodes": None,
    },
}


def _normalize_name(network_name: str) -> str:
    return (network_name or "SiouxFalls").strip().lower()


def parse_centroid_nodes(raw_value: Optional[Iterable[int] | str]) -> Optional[Tuple[int, ...]]:
    """Parse explicit centroid node ids from CLI/config values."""
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if not cleaned:
            return None
        return tuple(int(x.strip()) for x in cleaned.split(",") if x.strip())
    return tuple(int(x) for x in raw_value)


def _resolve_path(dataset_root: str, file_path: str) -> str:
    if not file_path:
        return ""
    if os.path.isabs(file_path):
        return file_path
    if dataset_root:
        return os.path.join(dataset_root, file_path)
    return file_path


def _resolve_dataset_root(dataset_root: str, relative_to_module: bool = False) -> str:
    if not dataset_root:
        return ""
    if os.path.isabs(dataset_root):
        return dataset_root
    base_dir = _MODULE_DIR if relative_to_module else os.getcwd()
    return os.path.abspath(os.path.join(base_dir, dataset_root))


def resolve_network_spec(
    network_name: str = "SiouxFalls",
    dataset_root: str = "",
    network_file: str = "",
    od_file: str = "",
    parser: str = "tntp",
    node_id_offset: int = 1,
    demand_source: str = "lhs",
    centroid_nodes: Optional[Iterable[int] | str] = None,
) -> NetworkSpec:
    """Resolve a network spec from built-in defaults plus user overrides."""
    key = _normalize_name(network_name)
    if key not in _BUILTIN_SPECS:
        raise ValueError(
            f"Unsupported network_name='{network_name}'. "
            f"Supported values: {sorted(_BUILTIN_SPECS)}"
        )

    base = _BUILTIN_SPECS[key]
    resolved_root = _resolve_dataset_root(
        dataset_root or base["dataset_root"],
        relative_to_module=not bool(dataset_root),
    )
    resolved_network_file = _resolve_path(
        resolved_root,
        network_file or base["network_file"],
    )
    resolved_od_file = _resolve_path(
        resolved_root,
        od_file or base["od_file"],
    )

    explicit_centroids = parse_centroid_nodes(centroid_nodes)
    if explicit_centroids is None:
        explicit_centroids = base["centroid_nodes"]

    return NetworkSpec(
        network_name=base["network_name"],
        dataset_root=resolved_root,
        network_file=resolved_network_file,
        od_file=resolved_od_file,
        parser=parser,
        node_id_offset=node_id_offset,
        demand_source=demand_source,
        centroid_nodes=explicit_centroids,
        mutation_policy=MutationPolicy(),
    )
