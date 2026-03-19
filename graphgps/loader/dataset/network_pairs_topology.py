"""In-memory loader for processed traffic network-pairs datasets."""

import logging
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset


class NetworkPairsTopologyDataset(InMemoryDataset):
    """Load one split of the processed network-pairs dataset."""

    _SPLIT_FILES = {
        'train': 'train_dataset.pt',
        'val': 'val_dataset.pt',
        'test': 'test_dataset.pt',
    }

    def _download(self):
        pass

    def _process(self):
        pass

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        assert split in self._SPLIT_FILES, (
            f"split must be one of {tuple(self._SPLIT_FILES)}, got '{split}'"
        )
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

        pt_path = osp.join(self.processed_dir, self._SPLIT_FILES[split])
        if not osp.exists(pt_path):
            raise FileNotFoundError(
                f"Missing processed split file: {pt_path}\n"
                f"Run the data pipeline first:\n"
                f"  1. python solve_network_pairs.py --network_name <SiouxFalls|EMA|Anaheim>\n"
                f"  2. python build_network_pairs_dataset.py --output_dir {self.root}\n"
            )

        data_list = torch.load(pt_path, weights_only=False)
        logging.info(
            f"[NetworkPairsTopologyDataset] Loaded {split} split: "
            f"{len(data_list)} graphs from {pt_path}"
        )
        self.data, self.slices = self.collate(data_list)

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return list(self._SPLIT_FILES.values())

    def download(self):
        raise FileNotFoundError(
            "NetworkPairs datasets are not downloaded automatically.\n"
            "Generate them locally first:\n"
            "  1. python solve_network_pairs.py --network_name <SiouxFalls|EMA|Anaheim>\n"
            f"  2. python build_network_pairs_dataset.py --output_dir {self.root}\n"
        )

    def process(self):
        pass

    def __repr__(self) -> str:
        return (
            f"NetworkPairsTopologyDataset("
            f"split={self.split}, "
            f"num_graphs={len(self)})"
        )
