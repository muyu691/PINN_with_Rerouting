"""
Pseudo-Spatiotemporal Physics-Informed Diffusion Model (ST-PINN)
================================================================

Core innovation: Simulate traffic flow redistribution as a physics-driven
diffusion process over K pseudo-time steps, predicting new equilibrium flows
after network topology reconfiguration.  Absolutely NO OD matrix is used.

Architecture (self-driven physical simulator unrolled over K steps):

  ┌─────────────────────────────────────────────────────────────────┐
  │ Phase 2: Initial State Projection (k=0)                         │
  │   _project_initial_flow  → f_scaled_0 : [E_new, 1]             │
  │   _compute_initial_pressure → rho_v_0 : [N, 1]  (real space)   │
  ├─────────────────────────────────────────────────────────────────┤
  │ EdgeAlignmentModule → aligned_features : [E_new, 8]             │
  │   (static edge context for h_e^(0) initialization)              │
  ├─────────────────────────────────────────────────────────────────┤
  │ Phase 3: Pseudo-Time Diffusion Loop  (k = 1 … K)               │
  │   DiffusionCell (Neural Darcy's Law):                           │
  │     Dual-stream: GatedGCN (local) + Self-Attention (global)     │
  │     → Δf_scaled^(k)                                             │
  │   Flow update:  f^(k) = f^(k-1) + Δf^(k)                       │
  │   LWR update:   ρ^(k) = ρ^(k-1) + scatter(Δf_real, in - out)   │
  ├─────────────────────────────────────────────────────────────────┤
  │ Output: f_scaled^(K)  (terminal equilibrium flows, normalized)  │
  │         ρ_v^(K)       (terminal pressure, should → 0)           │
  └─────────────────────────────────────────────────────────────────┘

Legacy modules (OldGraphEncoder, NewGraphReasoner) are retained for reference
but are no longer used in the forward pass.

Batch processing note:
  PyG's Batch.from_data_list() adds node offsets via __inc__, so edge_index_old
  and edge_index_new contain globally unique node IDs after batching.
  Hash keys (src × total_nodes + dst) remain unique across graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.utils import match_edge_indices


# ============================================================
# Lightweight GNN Batch container
# ============================================================

class _GNNBatch:
    """
    Minimalist data container—intended solely to drive GatedGCNLayer.forward().

    GatedGCNLayer.forward(batch) only accesses these three fields:
      batch.x          : [N, H]  node features
      batch.edge_attr  : [E, H]  edge features
      batch.edge_index : [2, E]  connectivity

    When GatedGCNLayer is initialized with equivstable_pe=False,
    code path `batch.pe_EquivStableLapPE` will never be executed
    (Python short-circuit evaluation), so this container does not need it.
    """

    __slots__ = ('x', 'edge_attr', 'edge_index')

    def __init__(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        self.x = x
        self.edge_attr = edge_attr
        self.edge_index = edge_index


# ============================================================
# Module 1: OldGraphEncoder (historical hub memory encoder)
# ============================================================

class OldGraphEncoder(nn.Module):
    """
    Encodes historical traffic congestion patterns on the old graph G.

    Design notes:
      - Nodes are initialized as "blank slate" ones(N, H), starting directly in hidden space,
        avoiding a meaningless 1 → H linear mapping.
      - Edge inputs = cat([edge_attr_old(3), flow_old(1)]) = [E_old, 4],
        then projected into hidden space and fed to the GatedGCN stack.
      - The gated aggregation of GatedGCN allows the model to distinguish high/low flow links,
        effectively learning "which nodes are congestion hubs."

    Tensor shape conventions (H = hidden_dim):
      Input:
        edge_index_old  : [2, E_old]
        edge_attr_old   : [E_old, 3]  - [capacity, speed, length] (normalized)
        flow_old        : [E_old, 1]  - historical equilibrium flows (normalized)
        num_nodes       : int         - batch total node count

      Output:
        h_nodes_old     : [num_nodes, H]  - node historical memory embedding
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Edge feature projection: [capacity, speed, length, flow_old] → hidden space
        # Shape change: [E_old, 4] → [E_old, H]
        # GatedGCNLayer requires node and edge features to have the same dimension (H)
        self.edge_proj = nn.Linear(4, hidden_dim)

        # Stacked GatedGCN layers: multi-round message passing on old graph topology
        # Each layer: in_dim = out_dim = H, keep dimensions invariant
        self.gnn_layers = nn.ModuleList([
            GatedGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        edge_index_old: torch.Tensor,   # [2, E_old]
        edge_attr_old: torch.Tensor,    # [E_old, 3]
        flow_old: torch.Tensor,         # [E_old, 1]
        num_nodes: int,
    ) -> torch.Tensor:                  # → [num_nodes, H]

        device = edge_attr_old.device
        dtype  = edge_attr_old.dtype

        # -- Blank node initialization -------------------------------------
        # Directly initialize in hidden space H, avoiding meaningless 1→H layer.
        # All nodes start in the same "no prior" state; history is injected solely
        # via message passing from edge flows.
        # Shape: [num_nodes, H]
        x = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        # -- Edge feature projection ---------------------------------------
        # cat([edge_attr_old(3), flow_old(1)]) → [E_old, 4]
        # then project to hidden space      → [E_old, H]
        e_raw = torch.cat([edge_attr_old, flow_old], dim=-1)  # [E_old, 4]
        e = self.edge_proj(e_raw)                              # [E_old, H]

        # -- Stacked message passing (old graph topology) -----------------
        for layer in self.gnn_layers:
            mini_batch = _GNNBatch(x, e, edge_index_old)
            mini_batch = layer(mini_batch)
            x = mini_batch.x           # [num_nodes, H]
            e = mini_batch.edge_attr   # [E_old, H]

        # Return node embedding as "historical congestion memory", discard edge embeddings
        return x   # h_nodes_old : [num_nodes, H]


# ============================================================
# Module 2: EdgeAlignmentModule (heterogeneous edge feature alignment)
# ============================================================

class EdgeAlignmentModule(nn.Module):
    """
    Vectorized heterogeneous edge feature alignment module (no learnable params).

    Core algorithm: O(E_old + E_new) vectorized matching based on node-pair hash.

    Alignment rules (strictly follow .cursorrules Phase 2 spec):
    ┌──────────────────────────────────────────────────────────────┐
    │ Retained edge (present in both G and G'):                   │
    │   [edge_attr_old(3), flow_old(1), edge_attr_new(3), 0(1)]   │
    │   → 8-dim total, is_new_edge = 0                            │
    ├──────────────────────────────────────────────────────────────┤
    │ Added/new edge (only in G'):                                │
    │   [zeros(3),         zero(1),    edge_attr_new(3), 1(1)]    │
    │   → 8-dim total, is_new_edge = 1                            │
    └──────────────────────────────────────────────────────────────┘

    Hash coding scheme:
      key(src, dst) = src × total_nodes + dst
      Since src, dst ∈ [0, total_nodes), key is unique for each node-pair.

    Batch correctness:
      PyG applies node offset to edge indices (__inc__) such that node IDs in different graphs
      do not overlap. Thus, using total_nodes (batch-global node count) as hash base, keys are unique.

    Tensor shape conventions (H = hidden_dim):
      Input:
        edge_index_old  : [2, E_old]
        edge_attr_old   : [E_old, 3]
        flow_old        : [E_old, 1]
        edge_index_new  : [2, E_new]
        edge_attr_new   : [E_new, 3]
        total_nodes     : int

      Output:
        aligned_features : [E_new, 8]
    """

    def forward(
        self,
        edge_index_old: torch.Tensor,   # [2, E_old]
        edge_attr_old: torch.Tensor,    # [E_old, 3]
        flow_old: torch.Tensor,         # [E_old, 1]
        edge_index_new: torch.Tensor,   # [2, E_new]
        edge_attr_new: torch.Tensor,    # [E_new, 3]
        total_nodes: int,
    ) -> torch.Tensor:                  # → [E_new, 8]

        device = edge_attr_new.device
        dtype  = edge_attr_new.dtype
        E_old  = edge_index_old.shape[1]
        E_new  = edge_index_new.shape[1]

        # -- Step 1: Encode edge node-pairs as unique integer keys ----------
        #
        # Formula: key(src, dst) = src × total_nodes + dst
        #
        # Rationale: Since 0 ≤ src, dst < total_nodes,
        #   any distinct (src, dst) produces distinct key—equiv. to
        #   flattening 2D index to 1D (row-major order).
        #
        # Time complexity: O(E_old + E_new), no Python for-loops.
        match_idx = match_edge_indices(
            edge_index_old=edge_index_old,
            edge_index_new=edge_index_new,
            total_nodes=total_nodes,
        )

        # -- Step 4: Construct aligned features from old graph side (4 dims)
        #
        # Retained edge: use real old features + old flows
        # Added edge: fill zeros (representing "did not exist before")
        #
        # old_feats[j] = cat([edge_attr_old[j], flow_old[j]]) shape [4]
        old_feats = torch.cat([edge_attr_old, flow_old], dim=-1)  # [E_old, 4]

        # Initialize zero matrix (default: all-added edges are zero from old side)
        aligned_old = torch.zeros(E_new, 4, dtype=dtype, device=device)

        # Boolean mask: mark which new edges exist in old graph (retained)
        retained_mask = match_idx >= 0  # [E_new], bool

        # Only fill for retained edges, avoid -1 index out of bound
        if retained_mask.any():
            # aligned_old[retained_mask] ← old_feats[match_idx[retained_mask]]
            # pure tensor indexing, no Python loops
            aligned_old[retained_mask] = old_feats[match_idx[retained_mask]]

        # -- Step 5: Build is_new_edge indicator ---------------------------
        #
        # is_new_edge = 1 marks new edge (not exist in old graph), model can distinguish
        # is_new_edge = 0 marks retained edge (exist in old graph)
        # Shape: [E_new, 1], float dtype for use in linear layers
        is_new_edge = (~retained_mask).to(dtype).unsqueeze(1)  # [E_new, 1]

        # -- Step 6: Concatenate final aligned feature ---------------------
        #
        # Retained edge: [edge_attr_old(3), flow_old(1), edge_attr_new(3), 0(1)] = 8
        # New edge:      [zeros(3),        zero(1),      edge_attr_new(3), 1(1)] = 8
        # Shape strictly [E_new, 8]
        aligned_features = torch.cat(
            [aligned_old, edge_attr_new, is_new_edge], dim=-1
        )  # [E_new, 8]

        return aligned_features


# ============================================================
# Implicit Demand Virtual Routing Layer (Implicit Virtual Edge Layer)
# ============================================================

class ImplicitVirtualRoutingLayer(nn.Module):
    r"""
    Implicit demand virtual routing layer -- uses global self-attention to break the locality limitation of GNN receptive field.

    Physical meaning:
      When a traffic network experiences disconnects or dramatic changes in attributes, flows get redistributed globally.
      Traditional GNN aggregates only 1-hop neighbors per layer, thus for flow transfer between distant OD pairs,
      many layers must be stacked to propagate the signal, but this can cause over-smoothing.

      This layer computes self-attention weights among all nodes within the same graph,
      equivalent to dynamically establishing "implicit virtual edges" (Implicit Virtual Links):
        - High attention score $\alpha_{ij}$ → strong association between node $i$ and $j$
          (potential OD relationship, alternative path relationship, flow conservation constraint relationship)
        - These virtual edges allow global flow redistribution signals to be transmitted in a single layer

      Mathematical expression:
        $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

      Architecture (Post-LN Transformer Block):
        $$\mathbf{x}_{\text{mid}} = \text{LN}\!\left(\mathbf{x} + \text{MHA}(\mathbf{x})\right)$$
        $$\mathbf{x}_{\text{out}} = \text{LN}\!\left(\mathbf{x}_{\text{mid}} + \text{FFN}(\mathbf{x}_{\text{mid}})\right)$$

    Batch safety (strictly prevents cross-graph contamination):
      PyG's to_dense_batch transforms flat [N, H] → [B, Max_N, H] dense tensors,
      together with key_padding_mask=~mask passed to MultiheadAttention, ensures:
        1. Padding positions (dummy nodes) do not participate in attention computation
        2. Nodes from different graphs will never communicate with each other
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-head self-attention: establish global implicit virtual edges within the same graph
        # batch_first=True → input/output format [B, Seq, H], consistent with to_dense_batch output
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-Attention LayerNorm (normalize after residual fusion)
        self.norm1 = nn.LayerNorm(hidden_dim)
        # Post-FFN LayerNorm
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed Forward Network (FFN): two Linear + ReLU layers, standard 4x expansion ratio
        # Function: applies nonlinear transform to globally aggregated information from attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,      # [N, hidden_dim]  flat node features
        batch: torch.Tensor,   # [N]             node belong-to index
    ) -> torch.Tensor:         # → [N, hidden_dim]
        """
        Forward pass: global self-attention → residual + LayerNorm → FFN → residual + LayerNorm.

        Args:
            x     : Flat node features. Nodes of all graphs in batch concatenated together. Shape [N, H]
            batch : Node assignment indices; the i-th node belongs to batch[i]-th graph. Shape [N]

        Returns:
            x_flat : Node features with global perspective (contains implicit virtual edge information), shape [N, H]
        """
        # -- Step A: Dense transformation --------------------------------
        # Transform flat [N, H] into 3D dense tensor [B, Max_N, H]
        # mask: [B, Max_N], True = real node, False = padding (filled with 0)
        # Physical meaning: aligns node lists for each graph in mini-batch to same-length sequences
        #                   for Self-Attention matrix ops
        x_dense, mask = to_dense_batch(x, batch)  # [B, Max_N, H], [B, Max_N]

        # -- Step B: Global self-attention (establish implicit virtual edges) -----------
        # key_padding_mask=~mask: Positions with value True are ignored by attention
        #   → Padding nodes neither generate nor receive attention
        #   → Nodes in different graphs never communicate (because they're on different batch dimension)
        # Physical meaning: each node calculates attention with all other nodes in the same graph,
        #                   building feature similarity-based "implicit virtual edges",
        #                   so that potential OD and alternative path relations are captured instantly
        attn_out, _ = self.attn(
            query=x_dense,
            key=x_dense,
            value=x_dense,
            key_padding_mask=~mask,
        )  # [B, Max_N, H]

        # -- Step C: Residual connection + LayerNorm (Post-Attention) -----
        # $\mathbf{x}_{\text{mid}} = \text{LN}(\mathbf{x} + \text{MHA}(\mathbf{x}))$
        # Residual connection retains original local features, with attention output adding global info
        x_dense = self.norm1(x_dense + attn_out)  # [B, Max_N, H]

        # -- Step D: FFN + residual + LayerNorm ---------------------------
        # $\mathbf{x}_{\text{out}} = \text{LN}(\mathbf{x}_{\text{mid}} + \text{FFN}(\mathbf{x}_{\text{mid}}))$
        ffn_out = self.ffn(x_dense)                # [B, Max_N, H]
        x_dense = self.norm2(x_dense + ffn_out)    # [B, Max_N, H]

        # -- Step E: Restore as flat format -------------------------------
        # Use mask boolean indexing to select real nodes, filter out padding
        # The number of True in mask matches N (number of all real nodes), dimension matches strictly
        x_flat = x_dense[mask]  # [N, H]

        return x_flat


# ============================================================
# Module 3: NewGraphReasoner (reasoner for new topology)
# ============================================================

class NewGraphReasoner(nn.Module):
    """
    Fuses historical memory and reasons future equilibrium flows over new graph G'.

    Node Fusion principle (critical constraint):
      x_new_init = ones(N, H)           ← blank node for new graph
      x_fused    = cat([x_new_init, h_nodes_old])  → [N, 2H]
      x          = node_fusion(x_fused)            → [N, H]

      ⚠️  Strictly forbid residual connections after node_fusion!
          Residual x = x_fused_proj + h_nodes_old makes a "memory shortcut",
          letting the model copy old flows and bypass learning new topology.
          Force nonlinear compression (Linear→ReLU→Dropout) to ensure the model
          rediscover flow patterns by new graph message passing.

    Edge Decoder design:
      Input = cat([x_src(H), x_dst(H), aligned_features(8)]) → [2H + 8]
      Output = flow_pred [E_new, 1]
      Last layer is a pure linear, no activation (Unbounded Output).
      Target has been StandardScaler-normalized (mean 0, std 1, can be negative),
      any bounded activation (like ReLU/Sigmoid) introduces distribution bias.

    Tensor shape conventions (H = hidden_dim):
      Input:
        edge_index_new   : [2, E_new]
        aligned_features : [E_new, 8]
        h_nodes_old      : [num_nodes, H]
        num_nodes        : int

      Output:
        flow_pred : [E_new, 1]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        residual: bool,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # -- Node Fusion layer --------------------------------------------
        # Input: cat([x_new_init(H), h_nodes_old(H)]) → [N, 2H]
        # Output: [N, H]
        #
        # Principle: Nonlinear compression forces a dynamic balance between
        # the "blank" (for new topology adaptation) and "historical memory" (old flow pattern).
        # Strictly no residual — avoids degenerate strategies that just outputs old flows.
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -- Implicit Demand Virtual Routing Layer (Implicit Virtual Routing) -------------
        # After node_fusion output and before GatedGCN local message passing,
        # inject full-graph perspective to each node via global self-attention,
        # overcoming GNN shortsightedness, allowing far-end flow redistribution signals
        # to propagate instantly.
        # Controlled by cfg.dataset.use_virtual_links (default True).
        self.use_virtual_links = cfg.dataset.use_virtual_links
        if self.use_virtual_links:
            self.virtual_routing = ImplicitVirtualRoutingLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Project the aligned edge features: [E_new, 8] → [E_new, H]
        # GatedGCNLayer requires same dimension for node and edge features
        self.edge_proj = nn.Linear(8, hidden_dim)

        # Stacked GatedGCN layers: multi-round message passing on new graph topology
        self.gnn_layers = nn.ModuleList([
            GatedGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

        # -- Edge-level flow decoder --------------------------------------
        # Input: cat([x_src(H), x_dst(H), aligned_features(8)]) → [2H + 8]
        #
        # Principle: Explicitly fuses "source node state + target node state + edge features";
        # better captures directional flow laws than decoding from edge only.
        # aligned_features retains original 8 dims (including is_new_edge bit);
        # allows decoder to distinguish new/retained edges and predict them differently.
        #
        # Last layer: nn.Linear(H, 1), no activation (Unbounded Output).
        decoder_in_dim = hidden_dim * 2 + 8
        self.edge_decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # ⚠️  Intentionally omit activation here!
            # StandardScaler-processed targets span (−∞, +∞),
            # any bounded activation (ReLU/Tanh/Sigmoid) will bias predictions.
        )

    def forward(
        self,
        edge_index_new: torch.Tensor,    # [2, E_new]
        aligned_features: torch.Tensor,  # [E_new, 8]
        h_nodes_old: torch.Tensor,       # [num_nodes, H]
        num_nodes: int,
        batch_vec: torch.Tensor,         # [num_nodes] node assignment index (must be provided by PyG Batch)
    ) -> torch.Tensor:                   # → [E_new, 1]

        device = h_nodes_old.device
        dtype  = h_nodes_old.dtype

        # -- Blank node initialization (new graph) ------------------------
        # x_new_init represents "no prior knowledge" for new graph.
        # After fusing with h_nodes_old, model must (re)assign flows by new topology,
        # not just inherit old states.
        # Shape: [num_nodes, H]
        x_new_init = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        # -- Historical memory fusion (Node Fusion) ------------------------
        # Concatenate blank node with historical embedding: [N, H] || [N, H] → [N, 2H]
        # Then nonlinearly compress: [N, 2H] → [N, H]
        #
        # ⚠️  No residual connection! x must be entirely determined by node_fusion nonlinear transform,
        #     h_nodes_old can't shortcut to downstream network.
        x_cat = torch.cat([x_new_init, h_nodes_old], dim=-1)  # [N, 2H]
        x = self.node_fusion(x_cat)                            # [N, H]

        # -- Implicit Demand Virtual Routing Layer (Implicit Virtual Routing) -------------------
        if self.use_virtual_links:
            assert batch_vec is not None, (
                "batch_vec is None! ImplicitVirtualRoutingLayer requires PyG's batch.batch vector."
            )
            assert batch_vec.shape[0] == num_nodes, (
                f"batch_vec length ({batch_vec.shape[0]}) does not match num_nodes ({num_nodes})!"
            )
            x = self.virtual_routing(x, batch_vec)  # [N, H], global implicit OD information fused

        # -- Aligned edge feature projection -------------------------------
        # Project 8-dim aligned feature into hidden space for GatedGCNLayer
        # Shape: [E_new, 8] → [E_new, H]
        e = self.edge_proj(aligned_features)  # [E_new, H]

        # -- Stacked message passing over new graph topology ---------------
        # This is where model "understands" new topology:
        #   - Added edges (is_new_edge=1) can propagate their physical attrs to neighbors
        #   - Deleted edges disappear, flow redistributes over retained paths
        for layer in self.gnn_layers:
            mini_batch = _GNNBatch(x, e, edge_index_new)
            mini_batch = layer(mini_batch)
            x = mini_batch.x           # [num_nodes, H]
            e = mini_batch.edge_attr   # [E_new, H]

        # -- Edge-level flow decoding --------------------------------------
        # For each new edge, extract src/dst node embeddings,
        # concat with raw 8-dim aligned features (not projected), feed to decoder.
        #
        # Keep original aligned_features (not projected/e),
        #   1. aligned_features contains is_new_edge indicator (explicitly shows new/retained)
        #   2. Raw physical attrs (capacity, speed, length) have flow-related dimension,
        #      directly feed un-distorted.
        src_idx, dst_idx = edge_index_new[0], edge_index_new[1]
        edge_repr = torch.cat(
            [x[src_idx], x[dst_idx], aligned_features], dim=-1
        )  # [E_new, 2H + 8]

        flow_pred = self.edge_decoder(edge_repr)  # [E_new, 1]

        return flow_pred


# ============================================================
# Phase 3: DiffusionCell (Neural Darcy's Law — single step)
# ============================================================

class DiffusionCell(nn.Module):
    r"""
    One pseudo-time step of Neural Darcy's Law.

    Implements the dual-stream architecture specified in .cursorrules Phase 3:

      $\mathbf{h}_e^{(k)} = \text{GNN\_with\_Global\_Attention}
          \left(\mathbf{Attr}_e,\; \mathbf{h}_e^{(k-1)},\;
                \rho_u^{(k-1)},\; \rho_v^{(k-1)}\right)$

      $\Delta f_{e\_scaled}^{(k)} = \text{MLP\_Readout}(\mathbf{h}_e^{(k)})$

    At each step k the cell:
      1. Injects current physics state (ρ_src, ρ_dst, f_scaled) into edge hidden state
      2. Injects current pressure ρ_v into node hidden state
      3. Local stream:  GatedGCN message passing on edge_index_new
      4. Global stream: Self-attention over all nodes (implicit virtual routing)
      5. Decodes Δf_scaled (scalar flow change) from updated edge features

    Tensor conventions (H = hidden_dim):
      Inputs:
        h_v            : [N, H]       node hidden state
        h_e            : [E_new, H]   edge hidden state
        rho_v          : [N, 1]       node pressure (real space, veh/hr)
        f_scaled_k     : [E_new, 1]   current scaled flow
        edge_index_new : [2, E_new]   new graph connectivity
        batch_vec      : [N]          node batch assignment

      Outputs:
        h_v_new        : [N, H]       updated node hidden state
        h_e_new        : [E_new, H]   updated edge hidden state
        delta_f_scaled : [E_new, 1]   predicted flow delta (unbounded)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()

        # Edge physics injection: fuse latent state with physical observables
        # [h_e(H), ρ_src(1), ρ_dst(1), f_scaled(1)] → [H]
        self.edge_inject = nn.Linear(hidden_dim + 3, hidden_dim)

        # Node physics injection: fuse latent state with current pressure
        # [h_v(H), ρ_v(1)] → [H]
        self.node_inject = nn.Linear(hidden_dim + 1, hidden_dim)

        # Local stream: GatedGCN for topology-aware message passing
        self.local_gnn = GatedGCNLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            residual=residual,
        )

        # Global stream: self-attention for long-range flow redistribution
        self.global_attn = ImplicitVirtualRoutingLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Δf readout: edge hidden → scalar delta flow
        # Last layer has NO activation (delta can be positive or negative)
        self.delta_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_v: torch.Tensor,             # [N, H]
        h_e: torch.Tensor,             # [E_new, H]
        rho_v: torch.Tensor,           # [N, 1]
        f_scaled_k: torch.Tensor,      # [E_new, 1]
        edge_index_new: torch.Tensor,  # [2, E_new]
        batch_vec: torch.Tensor,       # [N]
        apply_global_attn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        src, dst = edge_index_new[0], edge_index_new[1]

        # ── Step 1: Inject physics into edge representations ──────────────
        # The GNN "observes" the pressure gradient (ρ_src - ρ_dst) and current
        # flow through each edge, enabling physically-informed flow prediction.
        rho_src = rho_v[src]                                              # [E_new, 1]
        rho_dst = rho_v[dst]                                              # [E_new, 1]
        e_input = torch.cat([h_e, rho_src, rho_dst, f_scaled_k], dim=-1) # [E_new, H+3]
        h_e_injected = self.edge_inject(e_input)                          # [E_new, H]

        # ── Step 2: Inject physics into node representations ──────────────
        v_input = torch.cat([h_v, rho_v], dim=-1)                        # [N, H+1]
        h_v_injected = self.node_inject(v_input)                          # [N, H]

        # ── Step 3: Local stream — GatedGCN message passing ──────────────
        mini = _GNNBatch(h_v_injected, h_e_injected, edge_index_new)
        mini = self.local_gnn(mini)
        h_v_new = mini.x                                                  # [N, H]
        h_e_new = mini.edge_attr                                          # [E_new, H]

        # ── Step 4: Global stream — Self-attention (virtual routing) ─────
        if apply_global_attn:
            h_v_new = self.global_attn(h_v_new, batch_vec)                # [N, H]

        # ── Step 5: Readout Δf_scaled ─────────────────────────────────────
        delta_f_scaled = self.delta_readout(h_e_new)                      # [E_new, 1]

        return h_v_new, h_e_new, delta_f_scaled


# ============================================================
# Main model: NetworkPairsTopologyModel
# ============================================================

@register_network('topology_gnn')
class NetworkPairsTopologyModel(nn.Module):
    r"""
    ST-PINN: Pseudo-Spatiotemporal Physics-Informed Diffusion Model
    for traffic network topology reconfiguration.

    Self-driven physical simulator that unrolls K pseudo-time diffusion
    steps to predict new equilibrium flows without any OD matrix.

    Submodules:
      self.aligner          : EdgeAlignmentModule   (static edge context, no params)
      self.edge_init_proj   : Linear(8 → H)         (project aligned features → h_e^(0))
      self.diffusion_cell   : DiffusionCell          (shared-weight recurrent cell)

    Registered buffers:
      self.flow_mean : [1]  flow scaler μ (veh/hr)
      self.flow_std  : [1]  flow scaler σ (veh/hr)

    Expected PyG Batch fields:
      batch.edge_index_old : [2, E_old]    old graph connectivity (node offset done)
      batch.edge_attr_old  : [E_old, 3]    old graph physical attrs (normalized)
      batch.flow_old       : [E_old, 1]    old graph flows (normalized)
      batch.edge_index_new : [2, E_new]    new graph connectivity (node offset done)
      batch.edge_attr_new  : [E_new, 3]    new graph physical attrs (normalized)
      batch.y              : [E_new, 1]    ground truth flows (normalized)
      batch.net_demand     : [N]           real-space D_v (veh/hr)
      batch.batch          : [N]           node-to-graph assignment

    Outputs attached to batch (for Phase 4 loss):
      batch.rho_v_final   : [N, 1]          terminal pressure ρ^(K) (real space)
      batch.rho_v_history : list of [N, 1]  pressure trajectory ρ^(0)…ρ^(K)

    Args:
        dim_in  : placeholder (GraphGym API), not used
        dim_out : placeholder (GraphGym API), output dim always 1
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()

        # ── Phase 2: Register flow scaler statistics as non-learnable buffers ──
        # These are loaded from flow_scaler.pkl by preformat_NetworkPairs() in master_loader.py
        # and written to cfg.dataset before the model is constructed.
        # register_buffer ensures:
        #   1. Auto-device sync (moves to GPU with model.to(device))
        #   2. Saved/loaded with state_dict (checkpoint persistence)
        #   3. NOT treated as learnable parameters (no gradients)
        self.register_buffer(
            'flow_mean', torch.tensor([cfg.dataset.flow_mean], dtype=torch.float32)
        )
        self.register_buffer(
            'flow_std', torch.tensor([cfg.dataset.flow_std], dtype=torch.float32)
        )

        # Read hyperparams from cfg.topology_gnn
        hidden_dim = cfg.topology_gnn.hidden_dim
        dropout    = cfg.topology_gnn.dropout
        residual   = cfg.topology_gnn.residual
        num_heads  = cfg.topology_gnn.num_heads
        self.K           = cfg.topology_gnn.num_diffusion_steps
        self.attention_every_k_steps = int(
            getattr(cfg.topology_gnn, 'attention_every_k_steps', 1)
        )
        self.enable_node_potential_head = bool(
            getattr(cfg.model, 'enable_node_potential_head', False)
            or float(getattr(cfg.model, 'lambda_rc', 0.0)) > 0.0
        )
        self.hidden_dim  = hidden_dim

        if self.attention_every_k_steps < 1:
            raise ValueError("topology_gnn.attention_every_k_steps must be >= 1")

        # ── Static edge context encoder (for h_e^(0) initialization) ──────
        # EdgeAlignmentModule produces 8-dim aligned features per new edge:
        #   [edge_attr_old(3), flow_old(1), edge_attr_new(3), is_new_edge(1)]
        # These encode the static "context" of each edge (its history & identity).
        self.aligner = EdgeAlignmentModule()
        self.edge_init_proj = nn.Linear(8, hidden_dim)

        # ── Pseudo-time recurrent diffusion cell (shared weights ∀ k) ─────
        # A single cell is unrolled K times in forward().
        # Weight sharing is physically motivated: the same Darcy's law applies
        # at every pseudo-time step; only the state (ρ, f) evolves.
        self.diffusion_cell = DiffusionCell(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            residual=residual,
        )

        # Lightweight node-potential head for the OD-free reduced-cost
        # surrogate. The loss itself is applied outside forward().
        if self.enable_node_potential_head:
            self.node_potential_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.node_potential_head = None

        def replace_bn_with_ln(module):
            for name, child in module.named_children():
                if 'BatchNorm' in child.__class__.__name__:
                    setattr(module, name, nn.LayerNorm(child.num_features))
                else:
                    replace_bn_with_ln(child)

        replace_bn_with_ln(self)

    # ================================================================
    # Phase 2: Initial State Projection (k=0)
    # ================================================================

    def _project_initial_flow(self, batch) -> torch.Tensor:
        r"""
        Project old equilibrium flows onto the new graph topology at pseudo-time k=0.

        Uses hash-key encoding (identical scheme to EdgeAlignmentModule) for
        fully vectorized, batch-safe edge matching:

          key(src, dst) = src × total_nodes + dst

        Mapping rules:
          - Retained edge (exists in both G and G'):
              f_scaled_0[e] = flow_old[matching_old_edge]  (already normalized)
          - New edge (only in G'):
              f_scaled_0[e] = (0.0 - μ) / σ  (representing exactly 0.0 real flow)

        Args:
            batch: PyG Batch object

        Returns:
            f_scaled_0: [E_new_total, 1]  initial edge flows in normalized space
        """
        device = batch.flow_old.device
        dtype = batch.flow_old.dtype
        total_nodes: int = batch.num_nodes

        E_old = batch.edge_index_old.shape[1]
        E_new = batch.edge_index_new.shape[1]

        # ── Hash-key encoding ──────────────────────────────────────────────
        match_idx = match_edge_indices(
            edge_index_old=batch.edge_index_old,
            edge_index_new=batch.edge_index_new,
            total_nodes=total_nodes,
        )
        retained_mask = match_idx >= 0            # [E_new], bool

        # ── Initialize: new edges → scaled representation of 0.0 real flow ─
        scaled_zero = (-self.flow_mean / self.flow_std).item()  # scalar
        f_scaled_0 = torch.full((E_new, 1), scaled_zero, device=device, dtype=dtype)

        # ── Copy retained edges' historical flows ──────────────────────────
        if retained_mask.any():
            f_scaled_0[retained_mask] = batch.flow_old[match_idx[retained_mask]]

        return f_scaled_0

    def _compute_initial_pressure(
        self,
        f_scaled_0: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        r"""
        Compute initial node pressure ρ_v^(0) in real physical space (veh/hr).

        This captures the flow imbalance ("pressure shockwave") caused by the
        sudden topology change from G to G':

          ρ_v^(0) = [ Σ_{e∈In(v)} f_{e,real}^{(0)} - Σ_{e∈Out(v)} f_{e,real}^{(0)} ] - D_v

        where D_v = batch.net_demand (pre-computed in Phase 1 from flows_old divergence).

        Physical meaning:
          - ρ > 0  →  traffic accumulation (jam) at node v
          - ρ < 0  →  demand void at node v
          - ρ = 0  →  node is in equilibrium

        Uses native PyTorch scatter_add_ for O(E) vectorized aggregation.

        Args:
            f_scaled_0: [E_new_total, 1]  initial flows in normalized space
            batch:      PyG Batch object

        Returns:
            rho_v_0: [N_total, 1]  initial pressure in real space (veh/hr)
        """
        device = f_scaled_0.device
        dtype = f_scaled_0.dtype
        total_nodes: int = batch.num_nodes

        # ── Step 1: Convert to real physical space ─────────────────────────
        f_real_0 = f_scaled_0 * self.flow_std + self.flow_mean   # [E_new, 1]
        f_real_0_flat = f_real_0.squeeze(-1)                      # [E_new]

        # ── Step 2: Scatter-add inflow / outflow per node ──────────────────
        src = batch.edge_index_new[0]  # [E_new]
        dst = batch.edge_index_new[1]  # [E_new]

        inflow = torch.zeros(total_nodes, device=device, dtype=dtype)
        outflow = torch.zeros(total_nodes, device=device, dtype=dtype)
        inflow.scatter_add_(0, dst, f_real_0_flat)
        outflow.scatter_add_(0, src, f_real_0_flat)

        # ── Step 3: Pressure = (inflow - outflow) - D_v ───────────────────
        divergence = inflow - outflow                                    # [N_total]
        net_demand = batch.net_demand.to(device=device, dtype=dtype)     # [N_total]
        rho_v_0 = (divergence - net_demand).unsqueeze(-1)                # [N_total, 1]

        return rho_v_0

    def _attach_real_new_edge_attrs(self, batch) -> None:
        """Expose real new-graph edge attributes for future physics losses.

        New processed datasets store raw edge attributes directly. For older
        datasets, fall back to attr_scaler statistics loaded into cfg by the
        dataset loader so reduced-cost experiments can still recover
        capacity/speed/length without changing the main model inputs.
        """
        if hasattr(batch, 'edge_attr_new_real'):
            edge_attr_new_real = batch.edge_attr_new_real.to(
                device=batch.edge_attr_new.device,
                dtype=batch.edge_attr_new.dtype,
            )
        else:
            attr_mean = getattr(cfg.dataset, 'edge_attr_mean', [])
            attr_std = getattr(cfg.dataset, 'edge_attr_std', [])
            edge_dim = batch.edge_attr_new.shape[1]
            if len(attr_mean) != edge_dim or len(attr_std) != edge_dim:
                raise RuntimeError(
                    "Reduced-cost surrogate requires real new-edge attributes. "
                    "Expected either batch.edge_attr_new_real with columns "
                    "[capacity, speed, length], or cfg.dataset.edge_attr_mean/std "
                    "loaded from attr_scaler.pkl so the model can recover those "
                    "real values from batch.edge_attr_new."
                )
            mean = batch.edge_attr_new.new_tensor(attr_mean).view(1, -1)
            std = batch.edge_attr_new.new_tensor(attr_std).view(1, -1)
            edge_attr_new_real = batch.edge_attr_new * std + mean

        if not torch.isfinite(edge_attr_new_real).all():
            bad = int((~torch.isfinite(edge_attr_new_real)).sum().item())
            raise RuntimeError(
                f"Recovered real new-edge attributes contain {bad} non-finite values. "
                f"Reduced-cost surrogate needs finite capacity / speed / length."
            )

        capacity_new_real = edge_attr_new_real[:, 0:1]
        speed_raw_new_real = edge_attr_new_real[:, 1:2]
        length_new_real = edge_attr_new_real[:, 2:3]
        if torch.any(capacity_new_real <= 0):
            count = int((capacity_new_real <= 0).sum().item())
            raise RuntimeError(
                f"Reduced-cost surrogate requires strictly positive capacity on "
                f"new edges, but found {count} invalid entries."
            )
        if torch.any(speed_raw_new_real <= 0):
            count = int((speed_raw_new_real <= 0).sum().item())
            raise RuntimeError(
                f"Reduced-cost surrogate requires strictly positive speed on "
                f"new edges, but found {count} invalid entries."
            )
        if torch.any(length_new_real <= 0):
            count = int((length_new_real <= 0).sum().item())
            raise RuntimeError(
                f"Reduced-cost surrogate requires strictly positive length on "
                f"new edges, but found {count} invalid entries."
            )

        speed_new_real = torch.clamp(speed_raw_new_real, min=1e-6)
        free_flow_time_new_real = length_new_real / speed_new_real * 60.0
        if not torch.isfinite(free_flow_time_new_real).all() or torch.any(free_flow_time_new_real <= 0):
            raise RuntimeError(
                "Recovered free-flow time contains invalid values. Please check "
                "the real new-edge capacity / speed / length fields."
            )

        batch.edge_attr_new_real = edge_attr_new_real
        batch.capacity_new_real = capacity_new_real
        batch.speed_new_real = speed_new_real
        batch.length_new_real = length_new_real
        batch.free_flow_time_new_real = free_flow_time_new_real

    def _attach_reduced_cost_state(
        self,
        batch,
        f_scaled_final: torch.Tensor,
        h_v_final: torch.Tensor,
    ) -> None:
        """Attach final-step real-space tensors for the OD-free RC surrogate.

        This is not an exact UE / Beckmann path gap because no OD or path set
        is available at train time. Instead, it builds the final-step KKT
        ingredients of an OD-free convex transshipment surrogate using only:
          - node net demand D,
          - separable edge cost c(f),
          - node potential phi.
        The goal is to provide a transparent "cost-driven + conservation"
        physics prior without introducing OD/path supervision.
        """
        rc_bpr_alpha = float(getattr(cfg.model, 'rc_bpr_alpha', 0.15))
        rc_bpr_beta = float(getattr(cfg.model, 'rc_bpr_beta', 4.0))
        flow_softplus_scale = max(
            float(getattr(cfg.model, 'flow_softplus_scale', 5.0)),
            1e-6,
        )

        f_real_final = f_scaled_final * self.flow_std + self.flow_mean
        # Keep both quantities in real flow units:
        #   - f_cost_pos: smooth positive proxy for cost evaluation
        #   - f_active  : activity weight for complementarity only
        f_cost_pos_final = flow_softplus_scale * F.softplus(
            f_real_final / flow_softplus_scale
        )
        f_active_final = torch.relu(f_real_final)
        if not torch.isfinite(f_real_final).all():
            raise RuntimeError("Final real-space flow contains NaN/Inf values.")
        if not torch.isfinite(f_cost_pos_final).all():
            raise RuntimeError("Final smooth positive flow for the reduced-cost cost branch contains NaN/Inf values.")
        if not torch.isfinite(f_active_final).all():
            raise RuntimeError("Final active-flow weight for the reduced-cost branch contains NaN/Inf values.")

        batch.f_real_final = f_real_final
        batch.f_cost_pos_final = f_cost_pos_final
        batch.f_active_final = f_active_final

        if self.node_potential_head is None:
            batch.phi_v_final = None
            batch.edge_cost_final = None
            batch.reduced_cost_final = None
            return

        phi_v_final = self.node_potential_head(h_v_final)
        if not torch.isfinite(phi_v_final).all():
            raise RuntimeError("Node-potential head produced NaN/Inf values in phi_v_final.")

        capacity_new_real = torch.clamp(
            batch.capacity_new_real.to(device=f_cost_pos_final.device, dtype=f_cost_pos_final.dtype),
            min=1e-6,
        )
        free_flow_time_new_real = torch.clamp(
            batch.free_flow_time_new_real.to(
                device=f_cost_pos_final.device,
                dtype=f_cost_pos_final.dtype,
            ),
            min=1e-6,
        )
        flow_capacity_ratio = torch.clamp(f_cost_pos_final / capacity_new_real, min=0.0, max=1.0e6)
        edge_cost_final = free_flow_time_new_real * (
            1.0 + rc_bpr_alpha * torch.pow(flow_capacity_ratio, rc_bpr_beta)
        )
        if not torch.isfinite(edge_cost_final).all():
            raise RuntimeError("BPR edge cost produced NaN/Inf values in edge_cost_final.")

        src = batch.edge_index_new[0]
        dst = batch.edge_index_new[1]
        reduced_cost_final = edge_cost_final + phi_v_final[dst] - phi_v_final[src]
        if not torch.isfinite(reduced_cost_final).all():
            raise RuntimeError("Reduced-cost surrogate produced NaN/Inf values in reduced_cost_final.")

        batch.phi_v_final = phi_v_final
        batch.edge_cost_final = edge_cost_final
        batch.reduced_cost_final = reduced_cost_final

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(self, batch):
        r"""
        Self-driven pseudo-spatiotemporal diffusion forward pass.

        Pipeline:
          Phase 2 — Initial State Projection (k=0):
            f_scaled_0  : [E_new, 1]   old flows projected onto new topology
            rho_v_0     : [N, 1]       initial pressure shockwave (real space)

          Phase 3 — Pseudo-Time Diffusion Loop (k=1..K):
            for each step k:
              DiffusionCell  → Δf_scaled^(k)              [E_new, 1]
              Flow update    → f_scaled^(k)               [E_new, 1]
              LWR update     → ρ_v^(k)                    [N, 1]

          Output: f_scaled^(K) as final predicted flows (normalized space)

        Physics state is attached to ``batch`` for Phase 4 loss computation:
          batch.rho_v_final   : [N, 1]          terminal pressure (should → 0)
          batch.rho_v_history : list of [N, 1]  pressure at every step

        Args:
            batch : PyG Batch object

        Returns:
            pred : [E_new_total, 1]  predicted equilibrium flows (normalized)
            true : [E_new_total, 1]  ground truth flows (normalized)
        """
        total_nodes: int = batch.num_nodes
        device = batch.edge_attr_new.device
        dtype = batch.edge_attr_new.dtype
        if self.enable_node_potential_head:
            self._attach_real_new_edge_attrs(batch)

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: Initial State Projection (k=0)
        # ══════════════════════════════════════════════════════════════════
        f_scaled_0 = self._project_initial_flow(batch)                # [E_new, 1]
        rho_v_0 = self._compute_initial_pressure(f_scaled_0, batch)   # [N, 1]

        # ══════════════════════════════════════════════════════════════════
        # Phase 3: Initialize Latent Hidden States
        # ══════════════════════════════════════════════════════════════════

        # Edge hidden state h_e^(0): encode static edge context
        # aligned_features captures [old_attrs, old_flow, new_attrs, is_new_edge]
        aligned_features = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=total_nodes,
        )                                                              # [E_new, 8]
        h_e = self.edge_init_proj(aligned_features)                    # [E_new, H]

        # Node hidden state h_v^(0): blank slate (ones)
        h_v = torch.ones(total_nodes, self.hidden_dim,
                         device=device, dtype=dtype)                   # [N, H]

        # ══════════════════════════════════════════════════════════════════
        # Phase 3: Pseudo-Time Diffusion Loop (k = 1 … K)
        # ══════════════════════════════════════════════════════════════════
        f_scaled_k = f_scaled_0       # [E_new, 1]
        rho_v_k    = rho_v_0          # [N, 1]
        rho_v_history = [rho_v_0]     # track pressure at every step

        src = batch.edge_index_new[0]  # [E_new]  (cache for LWR updates)
        dst = batch.edge_index_new[1]  # [E_new]

        for _k in range(self.K):
            rho_v_scaled = rho_v_k / self.flow_std
            apply_global_attn = ((_k + 1) % self.attention_every_k_steps == 0)

            # ── Neural Darcy's Law: observe (ρ, f) → predict Δf ──────────
            h_v, h_e, delta_f_scaled = self.diffusion_cell(
                h_v=h_v,
                h_e=h_e,
                rho_v=rho_v_scaled,
                f_scaled_k=f_scaled_k,
                edge_index_new=batch.edge_index_new,
                batch_vec=batch.batch,
                apply_global_attn=apply_global_attn,
            )
            # h_v         : [N, H]       updated node hidden
            # h_e         : [E_new, H]   updated edge hidden
            # delta_f_scaled : [E_new, 1]   predicted flow change

            # ── Update edge flow (normalized space) ──────────────────────
            f_scaled_k = f_scaled_k + delta_f_scaled                   # [E_new, 1]

            # ── Convert Δf to real space ─────────────────────────────────
            # CRITICAL: multiply by σ ONLY. Do NOT add μ!
            # Δf_real = Δf_scaled × σ  (the mean cancels in the delta)
            delta_f_real_flat = (delta_f_scaled * self.flow_std).squeeze(-1)  # [E_new]

            # ── Corrected Discrete LWR: update node pressure ─────────────
            # ρ_v^(k) = ρ_v^(k-1) + (Σ Δf_in_real - Σ Δf_out_real)
            delta_in = torch.zeros(total_nodes, device=device, dtype=dtype)
            delta_out = torch.zeros(total_nodes, device=device, dtype=dtype)
            delta_in.scatter_add_(0, dst, delta_f_real_flat)
            delta_out.scatter_add_(0, src, delta_f_real_flat)

            rho_v_k = rho_v_k + (delta_in - delta_out).unsqueeze(-1)  # [N, 1]
            rho_v_history.append(rho_v_k)

        # ══════════════════════════════════════════════════════════════════
        # Attach physics state to batch for Phase 4 loss computation
        # ══════════════════════════════════════════════════════════════════
        batch.rho_v_final = rho_v_k         # [N, 1]  terminal pressure
        batch.rho_v_history = rho_v_history  # list of K+1 tensors [N, 1]

        # Expose final diffusion states and reduced-cost tensors while keeping
        # the main model output unchanged.
        batch.h_v_final = h_v
        batch.h_e_final = h_e
        self._attach_reduced_cost_state(
            batch=batch,
            f_scaled_final=f_scaled_k,
            h_v_final=h_v,
        )

        return f_scaled_k, batch.y
