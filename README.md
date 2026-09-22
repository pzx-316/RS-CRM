# RS-CRM: Response-Stratified Categorical Rewiring

Python simulation and empirical-analysis code accompanying the manuscript
**Response-Stratified Categorical Rewiring for Longitudinal Omics Data with Application to Angina Response**.

RS-CRM studies temporal composition, intermediate-state predictive value,
between-group dynamic compatibility, and gene/submodule attribution in longitudinal
omics modules. This repository contains the two author-provided research scripts,
with portable input/output options and installation instructions.

## Contents

| File | Purpose |
| --- | --- |
| `rscrm_simulation_code.py` | Six simulation settings and driver-recovery experiments |
| `rscrm_analysis.py` | Analysis of a prepared 39-gene, 37-patient, three-time-point workbook |
| `requirements.txt` | Exact package versions used for the local validation |
| `VALIDATION.md` | Checks performed and their limitations |
| `source_manifest.json` | SHA-256 fingerprints of the supplied and packaged scripts |

## Installation

The scripts were validated with Python 3.9.13 on Windows. Use Python 3.9 for
the recorded environment; the pinned package versions are not intended for all
newer Python versions.

```sh
python -m venv .venv
```

Activate the environment using `.venv\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv/bin/activate` on macOS/Linux, then install:

```sh
python -m pip install -r requirements.txt
```

## Simulations

No external data are needed. From the repository folder:

```sh
python rscrm_simulation_code.py
```

This preserves the supplied script's execution defaults: 80 replications per
main setting (480 rows across six settings), and 20 driver-recovery replications.
The random seeds remain 20260428 and 20260429, respectively.

For a quick execution check:

```sh
python rscrm_simulation_code.py --replications 2 --driver-replications 2 --output-dir results/quickcheck
```

The draft manuscript specifies 500 main replications per setting and 200
driver-recovery replications. Those counts can be requested explicitly:

```sh
python rscrm_simulation_code.py --replications 500 --driver-replications 200 --output-dir results/manuscript_counts
```

Selecting these counts does not by itself verify agreement with the manuscript's
reported numbers. The original default counts differ from the draft; see
`VALIDATION.md` for the scope of validation.

Four CSV files are saved in the chosen folder (default `results/simulation`):

- `rscrm_simulation_replicates.csv`: replicate-level results for six settings.
- `rscrm_simulation_summary.csv`: mean and standard deviation by setting.
- `rscrm_driver_replicates.csv`: driver-recovery results.
- `rscrm_driver_summary.csv`: driver-recovery means and standard deviations.

## Empirical analysis

```sh
python rscrm_analysis.py --input "path/to/prepared_workbook.xlsx" --output-dir results/analysis
```

The workbook must already contain the selected module expression values; this
script does not process raw sequencing reads or construct the initial gene module.
The clinical workbook is not included in this repository.

Required layout:

- Worksheets named `0天37人数据`, `14天37人数据`, and `30天37人数据`.
- Each sheet contains 39 gene rows and 37 sample columns, in addition to its
  header row and first column of gene names.
- Row 1, columns B onward: text sample IDs, such as `P001-0`, `P001-14`,
  and `P001-30`. Removing the time suffix must identify the same participant.
- Column A, rows 2 onward: gene names in the same order on all three sheets.
- Other cells: nonnegative numeric expression values, without missing values.
- On **every sheet**, the first 7 sample columns belong to ME, the next 6 to E,
  the next 6 to I, and the last 18 to N. The script assigns groups from column
  position; cell colors are not used. The participant identities and group
  assignments must agree across sheets.
- Additional sheets are ignored.

The workflow performs log1p transformation and pooled standardization/PCA,
fits affine temporal maps, estimates bootstrap intervals, compares prediction
models, and exports structural, submodule, and gene-attribution tables and figures.
The supplied settings are retained: ridge penalty 1.0, 300 structural bootstrap
replicates, and 1,000 resamples for prediction-gain intervals.

Outputs are written to `results/analysis` by default. Existing files of the same
names in a selected output folder are replaced; use a new folder to retain a run.

## Data access

The accompanying draft identifies China National GeneBank DataBase (CNGBdb),
CNSA project **CNP0000461**, as the source of the transcriptomic study data.
This accession is recorded from the manuscript and was not independently
validated during repository preparation. Access to source data does not
necessarily provide the prepared 39-gene workbook, subject matching, and response
group assignments required by this script. Obtain the prepared inputs and their
provenance from the study authors under the applicable access conditions.

## Implementation scope

The original computational functions and empirical workflow have been preserved.
Preparation changes are limited to command-line input/output options, a guarded
entry point, positive replication-count checks, and saving simulation summaries.
Original source fingerprints are recorded in `source_manifest.json`.

The simulation and empirical scripts use different regularization settings
(0.05 versus 1.0), different preprocessing (standardization versus log1p followed
by standardization), and different gene perturbations (masking in a fixed PCA
space versus deleting a gene and refitting PCA). Prediction-model LOOCV uses a
latent space estimated from all observations, rather than refitting PCA inside
each held-out fold. These behaviors are inherited from the supplied code; they
should be considered when interpreting results and aligning the manuscript.

## Citation and reuse

Refer to this repository and the accompanying manuscript when describing use
of these scripts. No publication DOI or Zenodo archive DOI has been assigned
as part of this upload. No software license has been selected by the authors
for this deposit; this repository does not grant additional reuse permissions.

The intended journal, *Statistics in Medicine*, also requests code/simulations
as **Data Files** in its submission system. A repository link does not replace
that submission step. See the [journal's author guidelines](https://onlinelibrary.wiley.com/page/journal/10970258/homepage/forauthors.html).
