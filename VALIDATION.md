# Validation record

Validation date: 2026-09-22.

## Completed checks

- Both scripts parse successfully and display their command-line help.
- All computational function definitions are identical to the supplied source
  when compared as Python abstract syntax trees (ignoring source locations).
- The empirical workflow statements are also identical after wrapping them
  in the command-line entry point.
- The simulation script completed with its original execution defaults:
  80 replications for each of six settings and 20 driver-recovery replications.
  The output contained 480 main rows and 20 driver rows; all numerical values
  were finite. All four documented CSV files were generated.
- The empirical script completed on the author's locally available prepared
  workbook with 39 genes, 37 participants, and three longitudinal time points.
  It produced 12 nonempty CSV tables, 13 readable PNG figures, and a submodule
  membership text file. All numerical table values were finite. The full
  gene-score table contains 39 rows, and the pairwise table contains six pairs.
- A targeted scan found no embedded credentials or original absolute local
  input/output paths in the packaged files.

The clinical workbook and its generated outputs are not distributed here.

## Environment

Windows; Python 3.9.13. Package versions are listed in `requirements.txt`.
Validation used the installed environment; a fresh installation on a different
operating system has not been tested.

## Limits and manuscript alignment

- The supplied simulation entry point runs 80/20 replications, whereas the
  draft specifies 500/200. The 500/200 run has not been validated in this record.
- This is an execution and packaging check, not a complete scientific audit
  or verification of every number and figure reported in the draft.
- The simulation code masks each gene in a fixed PCA reference space for
  attribution; the empirical code deletes a gene and refits PCA. Their
  preprocessing and ridge penalties also differ. These original behaviors
  have not been changed during packaging.
- The latent preprocessing uses all observations before prediction-model
  leave-one-out fitting. This is not fully nested preprocessing validation.
- Successful execution with the author's local workbook does not establish
  that a third-party reader can reconstruct it directly from raw deposited data.
- No independent verification of the manuscript's data accession or its
  access conditions was performed.
