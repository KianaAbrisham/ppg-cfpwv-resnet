# Validation record

This record distinguishes completed software checks from full research evaluation. Documentation was updated on **2026-09-27**. The recorded training, notebook execution, test suite, and GitHub Actions run below were completed on **2026-09-24**.

## Local validation

**Environment:** Linux CPU, Python 3.12.14, PyTorch 2.7.1+cpu, Torchvision 0.22.1+cpu. The recorded environment and run arguments are in [`smoke_results/run.json`](smoke_results/run.json).

| Check | Recorded result |
| --- | --- |
| Automated test suite | 10 test cases passed |
| Subject-ID matching and invalid-input rejection | Passed |
| Shared split-helper regression test | Disjoint, complete, repeatable cross-validation partitions |
| Scaling and spectrogram-window regression tests | Passed |
| ResNet-18 checkpoint save/reload | Passed |
| Actual training | One epoch with a single train/validation/test split |
| Reloaded trained predictions compared with saved held-out predictions | Passed |
| CSV converter round trip | Preserved IDs, waveforms, and targets |
| Quickstart notebook | Five code cells executed, with saved outputs and no exceptions |
| Notebook schema and Python syntax | Passed |
| Original research notebook hash | Unchanged |
| Dependency consistency | `pip check` passed |

The shared unit test for split construction exercises cross-validation. The ResNet training runner uses the helper's separate holdout branch. During the documentation review on **2026-09-27**, that holdout branch was checked directly with 72 rows and seed 42: **45 training, 12 validation, and 15 test rows**, with complete coverage, no overlaps, and repeatable assignments. This additional check did not retrain the model.

Evidence:

- [`test_output.txt`](test_output.txt): actual automated test output, ending with `OK`.
- [`integration_checks.json`](integration_checks.json): trained-checkpoint and converter checks.
- [`smoke_results/`](smoke_results/): recorded environment, configuration, and small-run metrics.
- [`../notebooks/01_quickstart.ipynb`](../notebooks/01_quickstart.ipynb): executed training, inference, and plot.
- [`provenance.json`](provenance.json): source notebook hash and reproduction status.

Recorded training used **72 artificial software-fixture waveforms**, 512 samples per waveform, 64 × 64 model inputs, random initialization, and one epoch. These checks exercise data flow, optimization, evaluation, and checkpoint reuse. They do not establish convergence, physiological simulation quality, published accuracy, or clinical performance.

The recorded `run.json` contains `arguments.folds: 2` from the shared demo configuration. ResNet's holdout branch ignores that setting: [`smoke_results/summary.json`](smoke_results/summary.json) correctly reports one evaluated split. Its zero standard deviations are placeholders for a single result.

Paths in the recorded configuration identify the original validation runtime. Large checkpoints are excluded from the repository and can be regenerated with the demo command.

## GitHub Actions

The [Checks run on 2026-09-24](https://github.com/KianaAbrisham/ppg-cfpwv-resnet/actions/runs/36049289738) completed successfully for commit [`aebceb4`](https://github.com/KianaAbrisham/ppg-cfpwv-resnet/commit/aebceb49d7fea3e44b45bebd4ab14bd07047062d).

The [workflow](../.github/workflows/checks.yml) installs the pinned CPU dependencies on Ubuntu with Python 3.12, then runs:

```bash
python -m unittest discover -s tests -v
python train.py --demo --output runs/ci-demo
```

The workflow checks the test suite and one-epoch ResNet demo with random initialization. This record refers to that specific successful run; the repository badge reports the current workflow status.

## Coverage limits

Original research CSV exports, paper-trained checkpoints, and a complete final experiment record were unavailable. Full PWDB experiments, ImageNet-initialized training, and reproduction of the publication's numerical results have not been verified. Windows and macOS execution have not been validated. The supplied runner executes on CPU and does not implement GPU device selection.

Before reporting research performance, follow the [research notes](RESEARCH_NOTES.md) and retain the full experiment's configuration, input hashes, subject splits, preprocessing, checkpoint, and held-out predictions.

## Public-source conversion verified — 2026-09-28

The official PWDB v0.2 waveform archive, haemodynamic targets and provided fiducials were downloaded and verified against publisher checksums. All 4,374 subject IDs were aligned explicitly, all waveform values passed a CSV round-trip check, and the target units and sampling rate were checked against the source documentation. Four additional tests verify shuffled-ID alignment and rejection of duplicate IDs, missing subjects and corrupt cached downloads. See [public data setup](PUBLIC_DATA.md) and the [conversion manifest](public_data/conversion_manifest.json). These checks validate data preparation; they do not establish numerical reproduction of a paper.
