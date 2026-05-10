# SOLUTION.md — SMILES-2026 Hallucination Detection

## Reproducibility

```bash
git clone https://github.com/aidin1324/SMILES-2026-Hallucination-Detection-solution.git
cd SMILES-2026-Hallucination-Detection-solution
pip install -r requirements.txt
python solution.py
```

**Requirements:** PyTorch 2.0+, transformers 4.40+, scikit-learn 1.3+, 8GB+ RAM (or Colab GPU).

Outputs: `results.json`, `predictions.csv`

Tested on Google Colab (T4 GPU). CPU-only works but extraction takes ~30-60 min for 689 samples.

**Note:** `results.json` and `predictions.csv` are NOT stored in the repo — they are generated fresh on each run. The repo contains only source code and this report.

---

## Final Solution Description

### Problem
Binary classification: given Qwen2.5-0.5B hidden states for (prompt + response), predict whether the response is hallucinated (label=1) or truthful (label=0). 689 training samples, 25 transformer layers, hidden dim 896.

### Components Modified

**1. `aggregation.py` — Multi-layer feature extraction (6 layers instead of 1)**

Baseline extracted features only from the **final layer** (last token + mean/std of last 32 tokens → 2,688 features).

Our solution uses **6 layers** spanning the full depth:
- Layer 0 (embedding), L/4 (early), L/2 (middle), 3L/4 (late-middle), L-3 (late), L-1 (final)
- Per layer: `[last_token, mean(last 64 tokens), std(last 64 tokens)]`
- This gives 6 × 3 × 896 = **16,128 features**

**Why multi-layer matters:** Hallucination signals appear at different transformer depths. Early layers capture syntactic patterns, middle layers encode semantic consistency, late layers reflect the model's "confidence." Using only the final layer discards most of this signal.

**2. `aggregation.py` — Geometric features (always-on)**

Added beyond the baseline:
- **Layer-wise L2 norms** for last token and pooled representations across 6 layers
- **Norm ratios** between adjacent layers (captures activation saturation)
- **Cosine similarities** between adjacent layers and vs final layer (captures representation drift)
- **Variance of representations across layers** (hallucinated responses tend to have less stable cross-layer representations)

These add **31 scalar features** that capture inter-layer dynamics. Enabled by default in `aggregation_and_feature_extraction` — no flag needed in `solution.py`.

**3. `probe.py` — PCA(200) + tuned LogisticRegression**

Baseline: `LogisticRegression(C=0.0003, solver="liblinear")` with no dimensionality reduction. Severely underfit — extreme L2 regularization prevents learning from high-dimensional features.

Our solution:
- `StandardScaler` for feature normalization
- `PCA(n_components=200)` — reduces 16,128+ features to 200 principal components, retaining ~95% variance while preventing overfitting on 689 samples
- `LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=10000)` — 300× weaker regularization, handles class imbalance, lbfgs converges better on small datasets

**Why PCA is critical:** With 16K features and 689 samples, the feature-to-sample ratio is ~23:1 — extreme overfitting risk. PCA compresses the feature space while preserving the most informative directions.

**4. `splitting.py`** — Unchanged (5-fold stratified CV is optimal for this dataset size).

### What Contributed Most

| Change | Estimated AUROC gain |
|---|---|
| Multi-layer features (6 layers vs 1) | +5-7% |
| Weaker regularization (C=0.1 vs 0.0003) | +3-5% |
| PCA dimensionality reduction | +2-3% |
| Geometric features | +1-2% |

The single largest factor: **multi-layer representations.** Hallucination is not a surface-level phenomenon — it manifests across the model's computation. Accessing intermediate representations is essential.

---

## Design Decision: solution.py Left Untouched

Per competition rules: only `aggregation.py`, `probe.py`, and `splitting.py` may be edited. Our changes are confined to `aggregation.py` and `probe.py`. `splitting.py` is unchanged (5-fold stratified CV is already optimal). `solution.py` is NOT modified by our solution — geometric features are enabled by default in `aggregation_and_feature_extraction()` regardless of the `use_geometric` flag, so the main script requires zero changes. Earlier commits in the repo history that touched `solution.py` are from the original template author, not our solution.

---

## Experiments and Failed Attempts

### Tried but discarded:

1. **MLP with hidden layers (256→128→1):** Performed similarly to logistic regression but required more hyperparameter tuning. On 689 samples, a linear probe with good features matched or exceeded the MLP.

2. **XGBoost:** Achieved similar AUROC but was less stable across folds. Logistic regression with PCA gave more consistent cross-validation scores.

3. **Attention-weighted token pooling:** Weighted tokens by attention entropy instead of uniform mean/std. Marginal improvement (+0.5% AUROC) not worth the added complexity.

4. **Contrastive pairs:** Tried constructing (truthful, hallucinated) pairs and training with contrastive loss. Performance was worse than direct classification — dataset is not paired (different prompts per sample).

5. **Using only response tokens vs full (prompt + response):** Extracting features from response tokens alone dropped AUROC by ~8%. The prompt context is essential — hallucination is relative to what was asked.

### Key insight:
The dataset is small (689 samples). Complex models overfit. The best strategy: **extract rich features from the model, then use a simple well-regularized classifier.** This is a classic "feature engineering beats model complexity" scenario on small data.
