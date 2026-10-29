# Privacy-Preserving Inference with a Hybrid XGBoost + MLP Model

A hybrid **XGBoost + neural-network** classifier whose prediction step is executed under **homomorphic encryption (HE)**, so a model can score records **without ever seeing them in the clear**. Benchmarked on two UCI tabular datasets across three encrypted-inference backends (Paillier, CKKS single-sample, CKKS batched) and a plaintext baseline.

Two binary-classification tasks:

- **Adult Census Income** — predict whether income > \$50K.
- **Diabetes 130-US hospitals** — predict 30-day hospital readmission.

---

## Why "privacy-preserving"?

In an ordinary ML service, the client must hand the server its raw feature vector — income, demographics, medical history — in plaintext. Homomorphic encryption breaks that trade-off: it allows arithmetic to be performed **directly on ciphertexts**, so that `Dec(Enc(a) ⊕ Enc(b)) = a + b` without the key ever leaving the client. In this project that enables a concrete privacy guarantee:

- The client **encrypts its feature vector locally** and sends only ciphertexts.
- The server computes the model's score **blind** — it holds the model weights but never decrypts the input, learns nothing about any individual record, and produces an **encrypted logit**.
- Only the client, holding the secret key, **decrypts the result**.

This is exactly the setting where privacy matters most — income data and hospital readmission records are sensitive and often regulated (HIPAA/GDPR). HE provides *confidential inference*: useful predictions with cryptographically protected inputs, no trusted third party, and no raw data exposure on the server.

The design makes this practical through a **split architecture**. HE only supports a cheap linear algebra (additions and scalar multiplications), so the expensive non-linear modeling (trees + neural net) is done in plaintext *at feature-extraction time*, and **only the final linear scoring layer is encrypted** — the single step that would otherwise reveal the input:

```
┌──────────────── plaintext (feature extraction) ────────────────┐
│   features ─► XGBoost ─► class-probability features ─┐          │
│            └─► MLP ─────► 16-unit bottleneck embedding┘─► concat │
│                          │                                       │
│              Logistic Regression meta-learner ─► weights W, bias b│
└────────────────────────────────────────────────────────────────┘
                              │ W, b
                              ▼
┌─────────────── ENCRYPTED (per-query, the private step) ─────────┐
│   Enc(x) ─► logit = Σ_j Enc(x_j)·W_j + b ─► Dec ─► sigmoid       │
└────────────────────────────────────────────────────────────────┘
```

Because the head is linear, the encrypted computation reproduces the plaintext logit exactly (CKKS up to tiny approximation noise), so **privacy is gained with no loss of accuracy** — the sigmoid is applied after decryption, at zero cryptographic cost.

---

## Sampling applied in the code

Sampling happens at several stages — worth understanding because the reported runtimes and metrics depend on it:

1. **Stratified train/validation/test split.** `train_test_split(..., test_size=0.2, stratify=y)` carves off a 20% test set, then the remaining 80% is split again 80/20 into train/validation — all **stratified** so the class ratio is preserved in every split.
2. **SMOTE synthetic oversampling (training only).** Classes are imbalanced, so `SMOTE(k_neighbors=5)` generates synthetic minority-class examples on the **training set only** (never on validation/test), balancing the data the model learns from without contaminating evaluation.
3. **Feature selection.** `SelectKBest(f_classif, k=30)` for Adult and `k=50` for Diabetes keeps the most informative features (a form of feature-space sampling), after one-hot encoding and binning.
4. **Fixed test subsampling for encrypted inference.** HE inference is slow, so each notebook cell evaluates it on a **fixed slice of the test set**:
   ```python
   fixed_sample_size = 300        # 500 and 1000 in the other cells
   sample_size = min(fixed_sample_size, X_test_hybrid.shape[0])
   X_sample_hybrid = X_test_hybrid[:sample_size]   # first-N (deterministic head slice)
   ```
   This is a **deterministic first-N slice**, not a random draw — so results are reproducible but reflect whatever ordering the test split produced. Each dataset is run at **N = 300, 500, 1000** to show how encryption cost scales with the number of queries.
5. **CKKS SIMD batch size.** Batched CKKS packs `batch_size = min(16, sample_size)` records per ciphertext.
6. **XGBoost stochastic sampling.** `subsample=0.8`, `colsample_bytree=0.8` — each tree sees 80% of rows and columns (row/feature sampling for regularization).

---

## Technical details

### Hybrid feature extractor
| Component | Role | Output |
|-----------|------|--------|
| **XGBoost** | Gradient-boosted trees (`n_estimators=100`, `max_depth=6`, `lr=0.1`, L1/L2 reg, 0.8 subsample/colsample) | `predict_proba` → 2-dim probability vector |
| **MLP** (Keras) | `Dense(32)→Dense(16)→Dense(1,sigmoid)`, trained as a classifier then truncated at the **16-unit hidden layer** and reused as a feature extractor | 16-dim embedding |
| **Fusion + meta-learner** | `hstack([xgb_proba, mlp_embedding])` → **Logistic Regression** (`saga`, L2, class-balanced) tuned via `GridSearchCV` over `C` with 3-fold stratified CV, plus **F1-based threshold tuning** on the validation set | `W`, `b` for the encrypted head |

### Homomorphic encryption backends
- **Paillier** ([`phe`](https://github.com/data61/python-paillier), 1024-bit) — partially (additively) homomorphic. Weights/bias encrypted once; each logit accumulated as `Σ(enc_wⱼ · xⱼ) + enc_b`. **Exact, slowest.**
- **CKKS-Single** ([`TenSEAL`](https://github.com/OpenMined/TenSEAL), SEAL) — approximate FHE on real vectors. Context: `poly_modulus_degree=8192`, `coeff_mod_bit_sizes=[60,40,40,60]`, `global_scale=2⁴⁰`, Galois keys for packed dot-products.
- **CKKS-Batched** — SIMD packs a feature column across the batch into one ciphertext, multiplies by the scalar weight, and sums. **Fastest** HE path, near plaintext speed.

### Evaluation
Each backend reports **Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, Brier score**, plus per-mode wall-clock time and memory (via `psutil`), and the accuracy degradation of each encrypted mode vs. plaintext — confirming the encrypted predictions match the plaintext baseline.

---

## Requirements

```bash
pip install tenseal phe imbalanced-learn xgboost tensorflow scikit-learn pandas numpy matplotlib seaborn psutil
```

## Usage

```bash
jupyter notebook "Neural_Network_XGBoost_Hybrid_Model.ipynb"
```

Written for Google Colab; both datasets download automatically (Adult from the UCI repository, Diabetes from the UCI archive zip). The notebook is organized as independent cells — one per *(dataset, sample size)* pair — each printing the plaintext-vs-encrypted comparison and plots. Tune the `fixed_sample_size`, HE key/scale parameters, and `batch_size` to trade security/precision against runtime.

> ⚠️ **Research/demo code.** 1024-bit Paillier keys and modest CKKS parameters favor speed over production-grade security; the datasets are public. Choose cryptographic parameters against a real threat model before any real-world deployment.

## Project Structure

```
privacy-preserving-xgboost-mlp/
├── notebooks/
│   └── Neural_Network_XGBoost_Hybrid_Model1.ipynb
│   └── Neural_Network_XGBoost_Hybrid_Model2.ipynb
├── README.md
└── LICENSE
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Built with XGBoost, TensorFlow/Keras, scikit-learn, imbalanced-learn, [python-paillier (`phe`)](https://github.com/data61/python-paillier), and [TenSEAL](https://github.com/OpenMined/TenSEAL). Data: [UCI Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) and [UCI Diabetes 130-US hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).
