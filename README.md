# CDAF

**Context-Conditioned Diffusion and Availability-Aware Evidence Fusion for Drug–Target Interaction Prediction**

CDAF refines drug and protein representations using disease-conditioned latent diffusion and combines three complementary evidence channels: bilinear feature compatibility, training-graph connectivity, and shared-disease paths. Channel availability is explicitly masked before pair-specific fusion.

Project repository: <https://github.com/ZLSRW/CDAF>

![CDAF architecture](docs/assets/framework.png)

## Contents

| Location | Contents |
| --- | --- |
| `cdaf/` | Model, data loader, grouped splits, training, metrics, and inference |
| `configs/` | Fixed manuscript configurations for B, C, and F |
| `datasets/` | The 45 original CSV files, including precomputed features and identifier mappings |
| `reference/` | Exact partitions, graph contexts, archived validation predictions, metrics, and training histories for all 15 dataset–fold combinations |
| `tests/` | Behavioral tests for split isolation, missing evidence, diffusion, and checkpoint loading |
| `scripts/summarize_reference.py` | Recompute the reported CDAF metrics from archived predictions |
| `docs/PROTOCOL.md` | Data, evaluation, and training definitions |
| `docs/VALIDATION.md` | Checks performed on this release |

The package includes the fixed CDAF implementation and its reference predictions. Pretrained checkpoints, adapted baseline implementations, and molecular-docking software are not included. Training creates checkpoints locally. The reference results below are **validation results**; the reserved test partitions are separate.

## Installation

Run all commands from the repository root. The release was validated with Python 3.12 and CPU PyTorch 2.8.0. Python 3.10 or newer is required by the pinned numerical dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. For GPU training, install the PyTorch 2.8.0 build compatible with your CUDA environment before installing `requirements.txt`. The model reads the bundled Mol2Vec and ESM features directly; no language-model or molecular-embedding download is required.

## Verify the supplied data and reference results

```bash
python main.py --mode verify-data
python scripts/summarize_reference.py
python -m unittest discover -s tests -v
```

The first command checks all 45 CSV hashes and reconstructs all 15 fold definitions, including the exact row order, labels, positive-source indices, and graph contexts. The second recomputes the metrics from saved predictions; it does not rerun model inference.

| Dataset | AUC | AUPR | Accuracy | F1 |
| --- | --- | --- | --- | --- |
| B | 0.8981 ± 0.0179 | 0.8988 ± 0.0212 | 0.8253 ± 0.0178 | 0.8346 ± 0.0138 |
| C | 0.9117 ± 0.0100 | 0.9170 ± 0.0152 | 0.8416 ± 0.0176 | 0.8407 ± 0.0129 |
| F | 0.9209 ± 0.0123 | 0.9280 ± 0.0076 | 0.8570 ± 0.0123 | 0.8535 ± 0.0101 |

Values are the mean and sample standard deviation over five validation folds. Repeated records are aggregated by drug–protein pair before calculating these metrics. AUPR is `sklearn.metrics.average_precision_score`. Checkpoint selection and classification-threshold selection use validation data.

## Train

Run the fixed five-fold configurations for all datasets:

```bash
python main.py --mode train --dataset all --device auto --run-name cdaf_main
```

Run one dataset and fold:

```bash
python main.py --mode train --dataset B-dataset --folds 0 --device auto --run-name cdaf_B_fold0
```

The run directory contains its configuration, environment, exact partitions, training histories, selected thresholds, predictions, and checkpoints. For the second command, the final checkpoint is:

```text
outputs/cdaf_B_fold0/B-dataset/fold_0/final/best.pt
```

Each fold performs supervised warmup, restarts from the same random seed while loading the selected warmup checkpoint, pretrains diffusion, and jointly fine-tunes the trainable modules. The restart is part of the archived experimental protocol. Base feature encoders are frozen after warmup; disease-context encoders remain trainable in the full configuration.

For a quick execution check, run one batch and one epoch per objective:

```bash
python main.py --dataset B-dataset --folds 0 --smoke --device cpu --run-name cdaf_smoke
```

Smoke outputs are explicitly marked in `environment.json` and the metrics. They do not reproduce the manuscript's trained performance. A named run never overwrites an existing directory; omit `--run-name` to generate a timestamped name.

The public CLI accepts controlled overrides for epoch budgets, hidden dimension, fine-tuning learning rate, and the two auxiliary-loss weights. `python main.py --help` lists them. Defaults come from the dataset JSON files; modified settings describe a new experiment.

## Evaluate trained checkpoints

After the single-fold training command above:

```bash
python main.py --mode evaluate --dataset B-dataset --folds 0 --device auto \
  --weights-root outputs/cdaf_B_fold0 --run-name cdaf_B_fold0_validation
```

Evaluation loads the selected weights and stored threshold. The default evaluates validation pairs and reports the maximum score difference from the archived reference predictions. It does not adjust weights or select a new threshold.

To evaluate the reserved test partition with the same checkpoint:

```bash
python main.py --mode evaluate --dataset B-dataset --folds 0 --device auto \
  --weights-root outputs/cdaf_B_fold0 --eval-split test --run-name cdaf_B_fold0_test
```

Test evaluation writes a separate result. It is not the source of the manuscript's validation table.

## Predict candidate pairs

The input CSV must contain zero-based integer columns `drug,protein` in the original feature/node order. `examples/pairs_B.csv` supplies five valid example pairs for dataset B.

```bash
python main.py --mode predict --device auto \
  --checkpoint outputs/cdaf_B_fold0/B-dataset/fold_0/final/best.pt \
  --pairs-file examples/pairs_B.csv --prediction-output outputs/predictions_B.csv
```

The output includes entity identifiers, the DTI score, the stored-threshold prediction, each channel's score, availability and fusion weight, and the most highly weighted disease path when one exists. `top_disease_index=-1` denotes an unavailable path. Scores are ranking outputs; they have not been clinically calibrated.

## Reproducibility settings

| Setting | Value |
| --- | --- |
| Data and training seed | 42 |
| Latent dimension | 96 |
| Diffusion noise levels / denoiser width / residual blocks | 100 / 256 / 4 |
| DDIM start / transitions / stochasticity | 30 / 10 / η = 0 |
| Bilinear heads / rank | 4 / 32 |
| Graph layers | 2 |
| Path attention heads / maximum path instances | 4 / 32 |
| Warmup / diffusion / joint epoch budgets | 20 / 30 / 200 |
| Learning rates, in the same order | 10⁻³ / 3 × 10⁻⁴ / 10⁻⁴ |
| Supervised batch / diffusion batch | 128 / 256 |
| Weight decay / gradient norm limit | 10⁻⁴ / 5 |
| Early-stopping patience | 10 |
| B loss weights `(lambda_diffusion, lambda_aux)` | (0.2, 0.2) |
| C loss weights `(lambda_diffusion, lambda_aux)` | (0.4, 1.0) |
| F loss weights `(lambda_diffusion, lambda_aux)` | (0.2, 0.6) |

Inference noise uses `42 + 10000 + fold`. Floating-point execution can vary across platforms and devices. `reference/` preserves the original validation predictions and thresholds so that the reported aggregate metrics can be audited independently of retraining.

## Data provenance and use

The source benchmark studies cited in the manuscript are Zhang et al. (2018), DOI [10.1186/s12859-018-2220-4](https://doi.org/10.1186/s12859-018-2220-4); Luo et al. (2016), DOI [10.1093/bioinformatics/btw228](https://doi.org/10.1093/bioinformatics/btw228); and Gottlieb et al. (2011), DOI [10.1038/msb.2011.26](https://doi.org/10.1038/msb.2011.26). The bundled model-ready files, identifier ordering, and checksum manifest define the exact inputs used here. See [the protocol](docs/PROTOCOL.md) and [rights notice](RIGHTS_NOTICE.md).

## Citation and contact

The repository's citation metadata is in `CITATION.cff`. Correspondence: Guodong Li, <liguodong@ccnu.edu.cn>.
