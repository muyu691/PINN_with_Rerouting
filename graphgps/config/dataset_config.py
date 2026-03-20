from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options."""

    # Default GraphGym dataset knobs.
    cfg.dataset.node_encoder_num_types = 0
    cfg.dataset.edge_encoder_num_types = 0
    cfg.dataset.slic_compactness = 10
    cfg.dataset.infer_link_label = "None"

    # Raw / processed dataset selectors for multi-network traffic datasets.
    cfg.dataset.dataset_name = 'traffic-network-pairs'
    cfg.dataset.network_name = 'SiouxFalls'
    cfg.dataset.dataset_root = ''
    cfg.dataset.network_file = ''
    cfg.dataset.od_file = ''
    cfg.dataset.parser = 'tntp'
    cfg.dataset.processed_root = ''
    cfg.dataset.node_id_offset = 1
    cfg.dataset.centroid_nodes = ''
    cfg.dataset.demand_source = 'lhs'

    # Metadata loaded from dataset_meta.json at train time.
    cfg.dataset.num_nodes = 0
    cfg.dataset.num_edges_old = 0
    cfg.dataset.num_edges_new = 0
    cfg.dataset.od_dim = 0
    cfg.dataset.centroid_count = 0

    # Flow de-normalization parameters injected by the dataset loader.
    cfg.dataset.flow_mean = 0.0
    cfg.dataset.flow_std = 1.0
    cfg.dataset.free_flow_time_ref = 0.0

    # Edge-attribute scaler statistics injected by the dataset loader. These
    # allow recovering real [capacity, speed, length] values when older
    # processed datasets do not yet store raw edge attributes explicitly.
    cfg.dataset.edge_attr_mean = []
    cfg.dataset.edge_attr_std = []

    # Feature / routing flags kept for backward compatibility.
    cfg.dataset.use_virtual_links = True
    cfg.dataset.mask_capacity = False
    cfg.dataset.mask_fft = False
