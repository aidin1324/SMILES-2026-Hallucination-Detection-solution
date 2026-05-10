"""
aggregation.py — Compact response-heavy hidden-state aggregation.

The final tokens contain the assistant answer, so we summarize the final layer
with the last token plus mean/std pooling over the last 32 real tokens.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _selected_layer_indices(n_layers: int) -> list[int]:
    """Select a diverse set of layers: early, middle, late, very late."""
    candidates = [
        0,                        # embedding layer
        n_layers // 4,            # early
        n_layers // 2,            # middle
        (3 * n_layers) // 4,      # late-middle
        n_layers - 3,             # late
        n_layers - 1,             # final
    ]
    return sorted(set(max(0, min(int(i), n_layers - 1)) for i in candidates))


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-layer aggregation: 6 layers × 3 stats (last token + mean/std of last 64)."""
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    last_pos = int(real_positions[-1].item())
    end_pos = last_pos + 1
    n_layers = hidden_states.shape[0]

    layer_indices = _selected_layer_indices(n_layers)

    features: list[torch.Tensor] = []
    for layer_idx in layer_indices:
        layer = hidden_states[layer_idx]
        window = layer[max(0, end_pos - 64) : end_pos]

        features.extend([
            layer[last_pos],
            window.mean(dim=0),
            window.std(dim=0, unbiased=False),
        ])

    return torch.cat(features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Enhanced geometric features: layer norms, inter-layer cosine, ratios."""
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    first_pos = int(real_positions[0].item())
    last_pos = int(real_positions[-1].item())
    end_pos = last_pos + 1
    seq_len = max(end_pos - first_pos, 1)

    layer_indices = _selected_layer_indices(hidden_states.shape[0])

    # Last-token vectors across selected layers
    last_vectors = torch.stack([hidden_states[i, last_pos] for i in layer_indices])

    # Mean-pooled vectors (last 96 tokens) across selected layers
    window_start = max(first_pos, end_pos - 96)
    pooled_vectors = torch.stack([
        hidden_states[i, window_start:end_pos].mean(dim=0) for i in layer_indices
    ])

    scalars: list[torch.Tensor] = [
        hidden_states.new_tensor(float(seq_len)),
        hidden_states.new_tensor(float(seq_len)).log1p(),
    ]

    # Layer-wise L2 norms (last token)
    last_norms = last_vectors.norm(dim=1)  # (n_selected_layers,)
    scalars.extend(last_norms.unbind())

    # Layer-wise L2 norms (pooled)
    pooled_norms = pooled_vectors.norm(dim=1)
    scalars.extend(pooled_norms.unbind())

    # Norm ratios between adjacent layers
    if len(layer_indices) > 1:
        norm_ratios = last_norms[1:] / (last_norms[:-1] + 1e-8)
        scalars.extend(norm_ratios.unbind())

    # Cosine similarities: adjacent layers + vs final layer
    if len(layer_indices) > 1:
        adjacent_cos = F.cosine_similarity(last_vectors[:-1], last_vectors[1:], dim=1)
        final_cos = F.cosine_similarity(last_vectors[:-1], last_vectors[-1:], dim=1)
        scalars.extend(adjacent_cos.unbind())
        scalars.extend(final_cos.unbind())

    # Variance of last-token vectors across layers (captures representation stability)
    last_var = last_vectors.var(dim=0).mean()  # scalar
    scalars.append(last_var)

    # Variance of pooled vectors across layers
    pooled_var = pooled_vectors.var(dim=0).mean()
    scalars.append(pooled_var)

    return torch.stack(scalars).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    # Always include geometric features — hallucination signals live in
    # cross-layer dynamics. Config flag is ignored to keep solution.py untouched.
    geo_features = extract_geometric_features(hidden_states, attention_mask)
    return torch.cat([agg_features, geo_features], dim=0)
