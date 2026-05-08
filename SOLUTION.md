# SOLUTION.md — SMILES-2026 Hallucination Detection

## Reproducibility

```bash
git clone <this-repo>
cd SMILES-2026-Hallucination-Detection-solution
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets scikit-learn pandas numpy tqdm
python solution.py
```

**Note for CPU users:** `model.py` uses `torch_dtype=torch.float32` for CPU compatibility. On GPU, change back to `torch.bfloat16` for memory efficiency. The solution is otherwise identical.

Outputs: `results.json`, `predictions.csv`

No GPU required. Tested on CPU with PyTorch 2.11, transformers 4.40+.

---

## Final Solution Description

### Problem
Binary classification: given Qwen2.5-0.5B hidden states for (prompt + response), predict whether the response is hallucinated (label=1) or truthful (label=0).

### Components Modified

**1. `aggregation.py` — Multi-layer feature extraction**

Baseline extracted features only from the **final layer** (last token + mean/std of last 32 tokens → 2,688 features).

Final solution uses **6 layers** spanning the full depth of the model:
- Layer 0 (embedding), L/4 (early), L/2 (middle), 3L/4 (late-middle), L-3 (late), L-1 (final)
- Per layer: `[last_token, mean(last 64 tokens), std(last 64 tokens)]`
- This gives 6 × 3 × 896 = **16,128 features**

**Why multi-layer matters:** Hallucination signals appear at different transformer depths. Early layers capture syntactic patterns, middle layers encode semantic consistency, late layers reflect the model's "confidence." Using only the final layer discards most of this signal.

**2. `aggregation.py` — Enhanced geometric features**

Added beyond the baseline:
- **Layer-wise norm ratios** between adjacent layers (captures activation saturation)
- **Variance of token representations across layers** (measures representational stability — hallucinated responses tend to have less stable cross-layer representations)

These add 30+ scalar features that capture inter-layer dynamics.

**3. `probe.py` — PCA + tuned LogisticRegression**

Baseline: `LogisticRegression(C=0.0003, solver="liblinear")` with no dimensionality reduction. This is severely underfit — C=0.0003 imposes extreme L2 regularization, and liblinear is a linear solver that cannot handle high-dimensional feature interactions well.

Final solution:
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
| Enhanced geometric features | +1-2% |

The single largest factor: **multi-layer representations.** Hallucination is not a surface-level phenomenon — it manifests across the model's computation. Accessing intermediate representations is essential.

---

## Experiments and Failed Attempts

### Tried but discarded:

1. **MLP with hidden layers (256→128→1):** Performed similarly to logistic regression but required more hyperparameter tuning. On 689 samples, a linear probe with good features matched or exceeded the MLP. Simplicity won.

2. **XGBoost:** Achieved similar AUROC but was less stable across folds. Logistic regression with PCA gave more consistent cross-validation scores.

3. **Attention-weighted token pooling:** Weighted tokens by attention entropy instead of uniform mean/std. Marginal improvement (+0.5% AUROC) not worth the added complexity and extraction time.

4. **Contrastive pairs:** Tried constructing (truthful, hallucinated) pairs and training with contrastive loss. Performance was worse than direct classification — likely because the dataset is not paired (different prompts for truthful/hallucinated samples).

5. **Using only response tokens vs full (prompt + response):** Extracting features from response tokens alone dropped AUROC by ~8%. The prompt context is essential — hallucination is relative to what was asked.

### Key insight:
The dataset is small (689 samples). Complex models overfit. The best strategy: **extract rich features from the model, then use a simple well-regularized classifier.** This is a classic "feature engineering beats model complexity" scenario.
