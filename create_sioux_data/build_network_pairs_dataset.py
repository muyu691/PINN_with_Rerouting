"""
Build a processed PyG dataset from solved traffic network pairs.

Input:
  network_pairs_dataset.pkl from solve_network_pairs.py

Output:
  train_dataset.pt / val_dataset.pt / test_dataset.pt
  scalers/attr_scaler.pkl / scalers/flow_scaler.pkl
  dataset_meta.json
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm


DEFAULT_SIOUX_NODE_IDS = tuple(range(1, 25))
DEFAULT_SIOUX_CENTROIDS = tuple(range(1, 12))


def extract_edge_attrs(G, edge_list: list) -> np.ndarray:
    """Extract [capacity, speed, length] in the canonical edge order."""
    caps = np.array([G[u][v]['capacity'] for u, v in edge_list], dtype=np.float64)
    speeds = np.array([G[u][v]['speed'] for u, v in edge_list], dtype=np.float64)
    lengths = np.array([G[u][v]['length'] for u, v in edge_list], dtype=np.float64)
    return np.column_stack([caps, speeds, lengths])


def infer_node_ids(pair: dict) -> tuple[int, ...]:
    """Infer canonical node ordering for one pair."""
    if pair.get('node_ids'):
        return tuple(int(node_id) for node_id in pair['node_ids'])
    if pair.get('G') is not None:
        return tuple(sorted(pair['G'].nodes()))
    if pair.get('G_prime') is not None:
        return tuple(sorted(pair['G_prime'].nodes()))
    raise ValueError("Cannot infer node ids from pair metadata.")


def infer_centroid_nodes(pair: dict, node_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Infer centroid nodes from metadata with a Sioux fallback for old pickles."""
    if pair.get('centroid_nodes'):
        return tuple(int(node_id) for node_id in pair['centroid_nodes'])

    od_matrix = pair.get('od_matrix')
    if isinstance(od_matrix, np.ndarray) and od_matrix.ndim == 2:
        return tuple(node_ids[: int(od_matrix.shape[0])])

    if node_ids == DEFAULT_SIOUX_NODE_IDS:
        return DEFAULT_SIOUX_CENTROIDS

    return tuple()


def infer_network_name(pair: dict, node_ids: tuple[int, ...]) -> str:
    """Infer network name while keeping backward compatibility."""
    if pair.get('network_name'):
        return str(pair['network_name'])
    if node_ids == DEFAULT_SIOUX_NODE_IDS:
        return 'SiouxFalls'
    return 'Unknown'


def edge_list_to_index(edge_list: list, node_id_to_index: dict[int, int]) -> np.ndarray:
    """Convert raw node ids to contiguous 0-indexed edge_index."""
    if len(edge_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(
        [[node_id_to_index[u], node_id_to_index[v]] for u, v in edge_list],
        dtype=np.int64,
    )
    return arr.T


def fit_scalers(pairs: list, train_idx: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
    """Fit scalers on the training split only."""
    if len(train_idx) == 0:
        raise ValueError("Training split is empty, cannot fit StandardScaler.")

    print("  Collecting training features for fitting StandardScaler...")
    all_attrs = []
    all_flows = []

    for idx in train_idx:
        pair = pairs[int(idx)]
        all_attrs.append(extract_edge_attrs(pair['G'], pair['edge_list_old']))
        all_attrs.append(extract_edge_attrs(pair['G_prime'], pair['edge_list_new']))
        all_flows.append(pair['flows_old'].reshape(-1, 1))
        all_flows.append(pair['flows_new'].reshape(-1, 1))

    all_attrs_np = np.vstack(all_attrs)
    all_flows_np = np.vstack(all_flows)

    print(f"  Training edge attribute matrix shape: {all_attrs_np.shape}")
    print(f"  Training flow matrix shape:          {all_flows_np.shape}")

    attr_scaler = StandardScaler()
    attr_scaler.fit(all_attrs_np)

    flow_scaler = StandardScaler()
    flow_scaler.fit(all_flows_np)

    print(f"  attr_scaler mean: {attr_scaler.mean_}")
    print(f"  attr_scaler std:  {attr_scaler.scale_}")
    print(f"  flow_scaler mean: {flow_scaler.mean_[0]:.2f}")
    print(f"  flow_scaler std:  {flow_scaler.scale_[0]:.2f}")

    return attr_scaler, flow_scaler


def save_scalers(
    attr_scaler: StandardScaler,
    flow_scaler: StandardScaler,
    output_dir: str,
) -> None:
    """Persist fitted scalers."""
    scalers_dir = os.path.join(output_dir, 'scalers')
    os.makedirs(scalers_dir, exist_ok=True)

    with open(os.path.join(scalers_dir, 'attr_scaler.pkl'), 'wb') as f:
        pickle.dump(attr_scaler, f)
    with open(os.path.join(scalers_dir, 'flow_scaler.pkl'), 'wb') as f:
        pickle.dump(flow_scaler, f)

    print(f"  Scaler saved to: {scalers_dir}/")


def build_single_data_object(
    pair: dict,
    attr_scaler: StandardScaler,
    flow_scaler: StandardScaler,
) -> Data:
    """Convert one solved pair into a dynamic-size PyG Data object."""
    node_ids = infer_node_ids(pair)
    num_nodes = len(node_ids)
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    centroid_nodes = infer_centroid_nodes(pair, node_ids)
    centroid_set = set(centroid_nodes)
    network_name = infer_network_name(pair, node_ids)
    od_dim = (
        int(pair['od_matrix'].shape[0])
        if isinstance(pair.get('od_matrix'), np.ndarray) and pair['od_matrix'].ndim == 2
        else len(centroid_nodes)
    )

    raw_attr_old = extract_edge_attrs(pair['G'], pair['edge_list_old'])
    raw_attr_new = extract_edge_attrs(pair['G_prime'], pair['edge_list_new'])
    norm_attr_old = attr_scaler.transform(raw_attr_old).astype(np.float32)
    norm_attr_new = attr_scaler.transform(raw_attr_new).astype(np.float32)

    norm_flow_old = flow_scaler.transform(pair['flows_old'].reshape(-1, 1)).astype(np.float32)
    norm_flow_new = flow_scaler.transform(pair['flows_new'].reshape(-1, 1)).astype(np.float32)

    edge_index_old = edge_list_to_index(pair['edge_list_old'], node_id_to_index)
    edge_index_new = edge_list_to_index(pair['edge_list_new'], node_id_to_index)

    x = torch.ones((num_nodes, 1), dtype=torch.float32)
    non_centroid_mask = torch.tensor(
        [node_id not in centroid_set for node_id in node_ids],
        dtype=torch.bool,
    )

    src_old = np.array([node_id_to_index[u] for u, _ in pair['edge_list_old']], dtype=np.int64)
    dst_old = np.array([node_id_to_index[v] for _, v in pair['edge_list_old']], dtype=np.int64)
    flows_old_real = pair['flows_old'].reshape(-1).astype(np.float64)
    net_demand_np = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(net_demand_np, dst_old, flows_old_real)
    np.add.at(net_demand_np, src_old, -flows_old_real)
    net_demand = torch.from_numpy(net_demand_np.astype(np.float32))

    old_edge_set = set(tuple(edge) for edge in pair['edge_list_old'])
    new_edge_mask = torch.tensor(
        [tuple(edge) not in old_edge_set for edge in pair['edge_list_new']],
        dtype=torch.bool,
    )

    return Data(
        x=x,
        edge_index_old=torch.from_numpy(edge_index_old).long(),
        edge_attr_old=torch.from_numpy(norm_attr_old),
        flow_old=torch.from_numpy(norm_flow_old),
        edge_index_new=torch.from_numpy(edge_index_new).long(),
        edge_attr_new=torch.from_numpy(norm_attr_new),
        y=torch.from_numpy(norm_flow_new),
        non_centroid_mask=non_centroid_mask,
        net_demand=net_demand,
        new_edge_mask=new_edge_mask,
        node_ids=torch.tensor(node_ids, dtype=torch.long),
        num_nodes=num_nodes,
        num_edges_old=len(pair['edge_list_old']),
        num_edges_new=len(pair['edge_list_new']),
        centroid_count=len(centroid_nodes),
        od_dim=od_dim,
        mutation_type=pair['mutation_type'],
        network_name=network_name,
    )


def build_full_dataset(
    pairs: list,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    attr_scaler: StandardScaler,
    flow_scaler: StandardScaler,
) -> tuple[list, list, list]:
    """Build Data lists for train/val/test splits."""

    def _build_split(idx_arr: np.ndarray, split_name: str) -> list:
        dataset = []
        for idx in tqdm(idx_arr, desc=f"  Building {split_name} set"):
            dataset.append(build_single_data_object(pairs[int(idx)], attr_scaler, flow_scaler))
        return dataset

    print(f"\n  Building training set ({len(train_idx)} samples)...")
    train_dataset = _build_split(train_idx, 'Train')

    print(f"\n  Building validation set ({len(val_idx)} samples)...")
    val_dataset = _build_split(val_idx, 'Val')

    print(f"\n  Building test set ({len(test_idx)} samples)...")
    test_dataset = _build_split(test_idx, 'Test')

    return train_dataset, val_dataset, test_dataset


def split_indices(
    num_samples: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate train/val/test split indices with small-sample protection."""
    if num_samples <= 0:
        raise ValueError("No valid samples found in the input pickle.")

    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(num_samples)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    if n_train == 0:
        n_train = 1
    if val_ratio > 0 and num_samples - n_train >= 2 and n_val == 0:
        n_val = 1
    if n_train + n_val > num_samples:
        n_val = max(0, num_samples - n_train)

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def print_dataset_stats(train_dataset: list, val_dataset: list, test_dataset: list) -> None:
    """Print summary statistics for each split."""

    def _stats_one_split(dataset: list, name: str) -> None:
        if len(dataset) == 0:
            print(f"\n  [{name}] 0 samples")
            print("    Split is empty, skip statistics")
            return

        e_old = [int(d.num_edges_old) for d in dataset]
        e_new = [int(d.num_edges_new) for d in dataset]
        y_all = torch.cat([d.y for d in dataset]).numpy()
        mutation_dist = Counter(d.mutation_type for d in dataset)

        print(f"\n  [{name}] {len(dataset)} samples")
        print(f"    G  edges: fixed {e_old[0]}")
        print(f"    G' edges: min={min(e_new)}, max={max(e_new)}, mean={np.mean(e_new):.1f}")
        print(
            f"    y (normalized flows_new): min={y_all.min():.3f}, "
            f"max={y_all.max():.3f}, mean={y_all.mean():.3f}, std={y_all.std():.3f}"
        )
        print(f"    Mutation type dist: {dict(mutation_dist)}")

    print(f"\n{'=' * 60}")
    print("Dataset Statistics Summary")
    print(f"{'=' * 60}")
    _stats_one_split(train_dataset, 'Train')
    _stats_one_split(val_dataset, 'Val')
    _stats_one_split(test_dataset, 'Test')


def validate_single_data_object(data: Data) -> None:
    """Validate dynamic-size fields and value ranges for one Data object."""
    num_nodes = int(data.num_nodes)
    e_old = int(data.num_edges_old)
    e_new = int(data.num_edges_new)

    assert data.x.shape == (num_nodes, 1), f"x shape error: {data.x.shape}"
    assert data.edge_index_old.shape == (2, e_old), "edge_index_old shape error"
    assert data.edge_index_new.shape == (2, e_new), "edge_index_new shape error"
    assert data.edge_attr_old.shape == (e_old, 3), "edge_attr_old shape error"
    assert data.flow_old.shape == (e_old, 1), "flow_old shape error"
    assert data.edge_attr_new.shape == (e_new, 3), "edge_attr_new shape error"
    assert data.y.shape == (e_new, 1), "y shape error"
    assert data.non_centroid_mask.shape == (num_nodes,), "non_centroid_mask shape error"
    assert data.net_demand.shape == (num_nodes,), f"net_demand shape error: {data.net_demand.shape}"
    assert data.new_edge_mask.shape == (e_new,), "new_edge_mask shape error"

    for name, ei in [('edge_index_old', data.edge_index_old), ('edge_index_new', data.edge_index_new)]:
        if ei.numel() == 0:
            continue
        assert ei.min() >= 0, f"{name} has negative node id"
        assert ei.max() < num_nodes, f"{name} node id out of range (max={ei.max()})"

    for name, tensor in [
        ('edge_attr_old', data.edge_attr_old),
        ('flow_old', data.flow_old),
        ('edge_attr_new', data.edge_attr_new),
        ('y', data.y),
        ('net_demand', data.net_demand),
    ]:
        assert not torch.isnan(tensor).any(), f"{name} contains NaN"
        assert not torch.isinf(tensor).any(), f"{name} contains Inf"

    assert torch.all(data.x == 1.0), "x is not all-ones placeholder."
    assert data.non_centroid_mask.dtype == torch.bool, "non_centroid_mask must be bool"
    assert data.new_edge_mask.dtype == torch.bool, "new_edge_mask must be bool"
    assert data.net_demand.dtype == torch.float32, f"net_demand dtype should be float32, got {data.net_demand.dtype}"

    if data.non_centroid_mask.any():
        non_centroid_residual = data.net_demand[data.non_centroid_mask].abs().max().item()
        assert non_centroid_residual < 1.0, (
            f"net_demand on non-centroid nodes exceeds tolerance: {non_centroid_residual:.4f} veh/hr"
        )


def save_dataset_metadata(
    output_dir: str,
    example_data: Data,
    train_size: int,
    val_size: int,
    test_size: int,
) -> None:
    """Persist metadata used by loaders and downstream configs."""
    node_ids = example_data.node_ids.tolist() if hasattr(example_data, 'node_ids') else []
    centroid_nodes = [
        node_id
        for node_id, is_non_centroid in zip(node_ids, example_data.non_centroid_mask.tolist())
        if not is_non_centroid
    ]
    metadata = {
        'network_name': str(getattr(example_data, 'network_name', 'Unknown')),
        'num_nodes': int(example_data.num_nodes),
        'num_edges_old': int(example_data.num_edges_old),
        'num_edges_new': int(example_data.num_edges_new),
        'od_dim': int(getattr(example_data, 'od_dim', 0)),
        'centroid_count': int(getattr(example_data, 'centroid_count', 0)),
        'node_ids': node_ids,
        'centroid_nodes': centroid_nodes,
        'splits': {
            'train': int(train_size),
            'val': int(val_size),
            'test': int(test_size),
        },
        'files': {
            'train': 'train_dataset.pt',
            'val': 'val_dataset.pt',
            'test': 'test_dataset.pt',
            'attr_scaler': 'scalers/attr_scaler.pkl',
            'flow_scaler': 'scalers/flow_scaler.pkl',
        },
    }

    meta_path = os.path.join(output_dir, 'dataset_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Dataset metadata saved: {meta_path}")


def _print_data_summary(data: Data, split_name: str) -> None:
    """Print a concise sample summary."""
    print(f"\n  [{split_name}] First sample field summary:")
    print(f"    x                  : {tuple(data.x.shape)}")
    print(f"    edge_index_old     : {tuple(data.edge_index_old.shape)}")
    print(f"    edge_attr_old      : {tuple(data.edge_attr_old.shape)}")
    print(f"    flow_old           : {tuple(data.flow_old.shape)}")
    print(f"    edge_index_new     : {tuple(data.edge_index_new.shape)}")
    print(f"    edge_attr_new      : {tuple(data.edge_attr_new.shape)}")
    print(f"    y                  : {tuple(data.y.shape)}")
    print(
        f"    non_centroid_mask  : {tuple(data.non_centroid_mask.shape)} "
        f"(True count: {int(data.non_centroid_mask.sum())})"
    )
    nd = data.net_demand
    centroid_mask = ~data.non_centroid_mask
    centroid_values = nd[centroid_mask]
    non_centroid_values = nd[data.non_centroid_mask]
    print(f"    net_demand         : {tuple(nd.shape)}  (from flow divergence, NO OD)")
    if centroid_values.numel() > 0:
        print(
            f"      centroid nodes   : count={centroid_values.numel()}, "
            f"range=[{centroid_values.min():.1f}, {centroid_values.max():.1f}]"
        )
    else:
        print("      centroid nodes   : count=0")
    if non_centroid_values.numel() > 0:
        print(
            f"      non-centroids    : count={non_centroid_values.numel()}, "
            f"max_abs={non_centroid_values.abs().max():.4f}"
        )
    else:
        print("      non-centroids    : count=0")
    print(f"    network_name       : {getattr(data, 'network_name', 'Unknown')}")
    print(f"    num_nodes          : {int(data.num_nodes)}")
    print(f"    od_dim             : {int(getattr(data, 'od_dim', 0))}")
    print(f"    mutation_type      : {data.mutation_type}")


def _save_split(dataset: list, path: str, name: str) -> None:
    """Save one split to a .pt file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(dataset, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  {name:6s}: {len(dataset):5d} samples -> {path} ({size_mb:.1f} MB)")


def run(args) -> None:
    """Execute the full dataset build workflow."""
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Step 1 - Load (G, G') network pairs")
    print(f"{'=' * 60}")
    print(f"  File path: {args.input_pkl}")

    with open(args.input_pkl, 'rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        pairs = payload['pairs']
        print(f"  Number of dropped samples (solve failed): {len(payload.get('failed_indices', []))}")
    else:
        pairs = payload

    num_samples = len(pairs)
    print(f"  Number of valid network pairs: {num_samples}")

    first = pairs[0]
    print(f"\n  First sample fields:")
    print(f"    G  edges      : {len(first['edge_list_old'])}")
    print(f"    G' edges      : {len(first['edge_list_new'])}")
    print(f"    flows_old     : {first['flows_old'].shape}  (real space, veh/hr)")
    print(f"    flows_new     : {first['flows_new'].shape}  (real space, veh/hr)")
    print(f"    mutation_type : {first['mutation_type']}")
    print("    NOTE: net_demand will be derived from flows_old divergence (NO OD matrix)")

    print(f"\n{'=' * 60}")
    print("Step 2 - Split Train / Val / Test indices")
    print(f"{'=' * 60}")
    train_idx, val_idx, test_idx = split_indices(
        num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  Train : {len(train_idx)} samples ({len(train_idx) / num_samples * 100:.1f}%)")
    print(f"  Val   : {len(val_idx)} samples ({len(val_idx) / num_samples * 100:.1f}%)")
    print(f"  Test  : {len(test_idx)} samples ({len(test_idx) / num_samples * 100:.1f}%)")

    print(f"\n{'=' * 60}")
    print("Step 3 - Fit StandardScaler (training set only)")
    print(f"{'=' * 60}")
    attr_scaler, flow_scaler = fit_scalers(pairs, train_idx)
    save_scalers(attr_scaler, flow_scaler, args.output_dir)

    print(f"\n{'=' * 60}")
    print("Step 4 - Build PyG Data objects")
    print(f"{'=' * 60}")
    train_dataset, val_dataset, test_dataset = build_full_dataset(
        pairs, train_idx, val_idx, test_idx, attr_scaler, flow_scaler
    )
    save_dataset_metadata(
        output_dir=args.output_dir,
        example_data=train_dataset[0],
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )

    print(f"\n{'=' * 60}")
    print("Step 5 - Validate sample correctness")
    print(f"{'=' * 60}")
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        if len(dataset) == 0:
            print(f"  [{name}] Split is empty, skip sample validation")
            continue
        validate_single_data_object(dataset[0])
        print(f"  [{name}] First sample passed validation")
        _print_data_summary(dataset[0], name)

    print_dataset_stats(train_dataset, val_dataset, test_dataset)

    print(f"\n{'=' * 60}")
    print("Step 7 - Save dataset")
    print(f"{'=' * 60}")
    _save_split(train_dataset, os.path.join(args.output_dir, 'train_dataset.pt'), 'Train')
    _save_split(val_dataset, os.path.join(args.output_dir, 'val_dataset.pt'), 'Val')
    _save_split(test_dataset, os.path.join(args.output_dir, 'test_dataset.pt'), 'Test')

    print(f"\n  All datasets saved to: {args.output_dir}/")
    print(f"  Scaler saved to:       {args.output_dir}/scalers/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build PyG dataset for solved traffic network pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input_pkl',
        type=str,
        default='processed_data/pairs/network_pairs_dataset.pkl',
        help='Path to the pickle output by solve_network_pairs.py',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='processed_data/pyg_dataset',
        help='Output directory for .pt files, scaler files, and dataset metadata',
    )
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Training set ratio')
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (test = 1 - train - val)',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    return parser.parse_args()


def main():
    print("\n" + "=" * 60)
    print("  Build PyG Network Pair Dataset")
    print("=" * 60)
    print(f"  Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()
    print(f"\n  Run arguments:")
    for key, value in vars(args).items():
        print(f"    {key:20s}: {value}")

    try:
        run(args)
    except KeyboardInterrupt:
        print("\n\n  User interrupted, exiting.")
        sys.exit(1)
    except Exception as exc:
        import traceback

        print(f"\n\n  [Error] Dataset construction failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
