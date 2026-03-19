try:
    from network_parser import load_network_data
    from network_registry import resolve_network_spec
except ModuleNotFoundError:
    from .network_parser import load_network_data
    from .network_registry import resolve_network_spec


def load_sioux_falls_network(net_file_path, od_file_path=""):
    """Backward-compatible Sioux Falls loader.

    The implementation now delegates to the generic TNTP parser while keeping
    the legacy function name so existing scripts continue to work.
    """
    spec = resolve_network_spec(
        network_name="SiouxFalls",
        network_file=net_file_path,
        od_file=od_file_path,
    )
    network_data = load_network_data(spec)
    centroids = list(network_data.centroid_nodes)

    print("Network statistics:")
    print(f"  Number of nodes: {network_data.graph.number_of_nodes()}")
    print(f"  Number of edges: {network_data.graph.number_of_edges()}")
    print(f"  Centroids: {centroids}")

    return network_data.graph, centroids
