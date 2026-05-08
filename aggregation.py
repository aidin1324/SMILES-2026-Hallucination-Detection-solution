"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _selected_layer_indices(n_layers: int) -> list[int]:
    """Choose stable lower/middle/final transformer layers."""
    candidates = [
        n_layers // 3,
        n_layers // 2,
        (2 * n_layers) // 3,
        max(n_layers - 5, 0),
        n_layers - 1,
    ]
    return sorted(set(int(i) for i in candidates if 0 <= i < n_layers))


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    last_pos = int(real_positions[-1].item())
    end_pos = last_pos + 1

    layer = hidden_states[-1]
    window = layer[max(0, end_pos - 32) : end_pos]

    return torch.cat(
        [
            layer[last_pos],
            window.mean(dim=0),
            window.std(dim=0, unbiased=False),
        ],
        dim=0,
    )


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    first_pos = int(real_positions[0].item())
    last_pos = int(real_positions[-1].item())
    end_pos = last_pos + 1
    seq_len = max(end_pos - first_pos, 1)

    layer_indices = _selected_layer_indices(hidden_states.shape[0])
    last_vectors = torch.stack([hidden_states[i, last_pos] for i in layer_indices])
    window_start = max(first_pos, end_pos - 96)
    pooled_vectors = torch.stack(
        [hidden_states[i, window_start:end_pos].mean(dim=0) for i in layer_indices]
    )

    scalars: list[torch.Tensor] = [
        hidden_states.new_tensor(float(seq_len)),
        hidden_states.new_tensor(float(seq_len)).log1p(),
    ]
    scalars.extend(last_vectors.norm(dim=1).unbind())
    scalars.extend(pooled_vectors.norm(dim=1).unbind())

    if len(layer_indices) > 1:
        adjacent_cos = F.cosine_similarity(last_vectors[:-1], last_vectors[1:], dim=1)
        final_cos = F.cosine_similarity(last_vectors[:-1], last_vectors[-1:], dim=1)
        scalars.extend(adjacent_cos.unbind())
        scalars.extend(final_cos.unbind())

    return torch.stack(scalars).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
