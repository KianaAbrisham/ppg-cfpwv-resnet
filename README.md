# PPG Spectrogram Regression with ResNet-18

[![Checks](https://github.com/KianaAbrisham/ppg-cfpwv-resnet/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/KianaAbrisham/ppg-cfpwv-resnet/actions/workflows/checks.yml)

Estimate carotid–femoral pulse wave velocity (cf-PWV, m/s) from photoplethysmography (PPG) spectrograms with a PyTorch ResNet-18 regressor. The project provides waveform validation, spectrogram preprocessing, training with a separate validation set, held-out evaluation, and saved-model inference.

**Author:** [Kiana Pilevar Abrisham](https://github.com/KianaAbrisham)  
**Related publication:** [Deep Learning-Based Estimation of Arterial Stiffness from PPG Spectrograms: A Novel Approach for Non-Invasive Cardiovascular Diagnostics](https://doi.org/10.1109/EMBC53108.2024.10782553) (2024)

Refactored from the author's research notebook, with strict subject-ID matching, training-only target scaling, repeatable splits, and automated tests. The workflow has completed software checks on artificial data. Full reproduction of the paper's numerical results has not been established; see the [validation record](docs/VALIDATION.md).

## Model and preprocessing

| Stage | Implementation |
| --- | --- |
| Signal input | One fixed-length waveform per subject from one artery |
| Spectrogram | SciPy power spectral density, converted to log power |
| Model input | Per-spectrogram min–max scaling, square resizing, three repeated channels, and fixed ImageNet channel normalization |
| Regressor | Torchvision ResNet-18 with its final layer replaced by a scalar linear output |
| Training | All model parameters optimized; validation loss selects the checkpoint |
| Output | Predicted cf-PWV restored to m/s |

Research mode uses 224 × 224 inputs and supports ImageNet initialization or random initialization. The supplied runner executes on CPU. Exact preprocessing choices are documented in the [research notes](docs/RESEARCH_NOTES.md).

## Quick start

Use **Python 3.12** in this repository's folder. Create an isolated environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\activate` in Windows Command Prompt. Linux CPU is the validated environment.

Install the pinned CPU dependencies, run the tests, and train the demo:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --demo --output runs/demo
```

The demo generates **72 artificial waveforms**, uses **512 samples per waveform** and **64 × 64 spectrogram inputs**, and trains for **one epoch on a single train/validation/test split**. It uses random initialization and does not download ImageNet weights. These fixtures are independent of PWDB; their scores measure neither research nor clinical performance. Use a new output folder for each training run.

Reload the saved checkpoint and write predictions:

```bash
python predict.py --model-dir runs/demo/resnet18/fold_1 --signals runs/demo/demo_input/signals.csv --output runs/demo/new_predictions.csv
```

The prediction CSV contains `subject_id`, `predicted_cfpwv_m_s`, and `model_status`. This command demonstrates checkpoint reuse on demo inputs. The directory name `fold_1` follows the shared output format; this project uses one holdout split, not cross-validation.

The [executed quickstart notebook](notebooks/01_quickstart.ipynb) shows training, inference, and an evaluation plot. See the [validation record](docs/VALIDATION.md) for completed checks and GitHub Actions coverage.

## Use research data

The related study uses [PWDB](https://zenodo.org/records/3275625), an in-silico dataset of virtual adults. Original research CSV exports and paper-trained checkpoints are not included. `prepare_data.py` converts existing wide CSV exports; it does not download or process the raw PWDB release automatically.

| File | Required columns | Meaning |
| --- | --- | --- |
| `signals.csv` | `subject_id`, `s0000`, `s0001`, … | One waveform per subject from one artery; sample columns in time order |
| `targets.csv` | `subject_id`, `cfpwv_m_s` | Finite, positive cf-PWV values in m/s |

Targets are matched by subject ID. IDs must be unique, and both files must contain exactly the same subjects. The loader rejects empty or constant waveforms, internal missing samples, infinities, and metadata mixed into sample columns. Shorter waveforms, including trailing empty sample cells, are zero-padded; longer waveforms are rejected without silent cropping.

Convert existing wide exports using the actual source column names:

```bash
python prepare_data.py --signals original_signals.csv --targets original_targets.csv --signal-id-column subject_id --target-id-column subject_id --target-column cfpwv_m_s --task regression --output data/converted
```

Quote column names containing spaces. Use `--drop-signal-columns` to remove exported indexes or metadata explicitly. If both files lack IDs, use `--assume-row-aligned` only after independently verifying their subject order.

Train with ImageNet initialization:

```bash
python train.py --signals data/converted/signals.csv --targets data/converted/targets.csv --site digital --models resnet18 --weights imagenet --output runs/digital-01
```

`--weights imagenet` loads Torchvision's `IMAGENET1K_V1` backbone weights, downloading them if they are not cached. The new regression layer is randomly initialized, and the full network is trained. Use `--weights none` for random initialization of the entire model. The recorded demo and CI checks use `none`; full research training and the ImageNet-initialized path have not been validated in this repository's recorded checks.

Set `--site` from verified dataset identity: `digital`, `radial`, or `brachial`. Use separate runs for different arteries; multiple records from one subject must not be treated as independent subjects.

Research defaults are `--length 2000`, `--fs 500`, `--nperseg 76`, `--window hamming`, a maximum of 200 epochs, early-stopping patience 20, batch size 32, and seed 42. Select length and sampling rate from the acquisition/export specification; the pipeline does not resample signals. Run `python train.py --help` for all options. The shared `--folds` argument does not change this runner's single-holdout protocol.

## Evaluation and saved outputs

The runner reserves 20% of subjects for testing, then 20% of the remaining subjects for validation: approximately **64% training / 16% validation / 20% testing**, subject to rounding. Regression target statistics are fitted on training subjects only. Input min–max scaling is performed separately for each spectrogram, followed by fixed ImageNet channel normalization. No dataset-wide input statistics are fitted.

Early stopping uses validation loss, and test predictions use the best validation state. Evaluation reports **MAE and RMSE (m/s), R², and MAPE (%)**, alongside a baseline that predicts the training subjects' mean target.

Each run saves:

- Configuration, dependency versions, input hashes, and subject IDs for each partition.
- Preprocessing metadata, the best model's `model.pt` state dictionary, loss history, metrics, and baseline results.
- Held-out predictions, an evaluation plot, and a summary for the single holdout split.

The shared summary format uses `mean` and `std` fields. Here, `mean` is the single-split result and `std: 0` is a placeholder; it does not estimate variability or uncertainty. Repeated model or hyperparameter selection requires a separate final test set or nested cross-validation.

Timing reports the median and 95th percentile of 20 synchronous, batch-one CPU forward passes after three warmups; input preprocessing is excluded. Checkpoint size measures the serialized model state dictionary, including model buffers and excluding optimizer state.

## Repository guide

| Location | Contents |
| --- | --- |
| [`ppg/`](ppg/) | Data validation, preprocessing, model, training, and inference |
| [`train.py`](train.py), [`predict.py`](predict.py), [`prepare_data.py`](prepare_data.py) | Command-line entry points |
| [`tests/`](tests/) | Input, split-helper, preprocessing, and checkpoint regression tests |
| [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Executed artificial-data walkthrough |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Completed checks, evidence, and coverage limits |
| [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) | Provenance and implementation choices |
| [`.github/workflows/checks.yml`](.github/workflows/checks.yml) | Automated CPU tests and demo |

Citation metadata are available in [`CITATION.cff`](CITATION.cff). Performance on simulated profiles alone does not establish performance on patient or wearable recordings.
