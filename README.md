# DCD SM replication package for readers

This folder contains the code, settings and saved results for **Simulation
evidence** and **Empirical evidence** in both the JFEC manuscript (`main.pdf`)
and the arXiv manuscript (`main_arxiv.pdf`). References use section titles
because section numbering differs between the two formats.
The manuscript versions and their SHA256 hashes, distributed-file hashes and
original snapshot hashes are recorded in `manifest.json`.
This folder can be copied to a separate repository without
the parent working repository or its Git history.

This is the public package for readers. It includes the replication code,
settings, saved aggregate results and data-collection instructions. Individual
market-return observations and prepared estimation inputs are not distributed
in this folder. The collection commands below obtain data from the original
providers and create a local `code/data/` workspace, excluded from Git.

## Correspondence with the paper

The following table links section titles and analyses to their result folders.
All paths in this table are relative to `code/paper/results/`.
The `sec4_*` and `sec5_*` path prefixes retain the manuscript source-file
identifiers. Use the section titles below to locate the analyses in either
manuscript version.

Experiment names, file names and directory names must identify the paper's
sections, models (M1–M5), DGP cases (0–4), targets (Raw/PC1 residual) or
dependence specifications. Execution dates must not be used as identifiers.
Sample dates, dated external sources, audit timestamps and numerical random
seeds retain their usual meaning.

| Paper content | Result folder | Contents |
| --- | --- | --- |
| Simulation evidence | `sec4_simulation_m1_m5/` | M1–M5 estimates summarized across Cases 0–4, independent draws and a 10% common factor, with N = 20, 50, 100 |
| Simulation evidence / Generating dynamics and sequential estimation | `sec4_simulation_dgp/` | Reference DGP specification used to check the simulation design and calibrated coefficients |
| Empirical evidence | `sec5_empirical_raw_pc1/` | The 19 equity panels, each with matched Raw and training-fixed PC1 residual inputs |
| Empirical evidence / Communication Services PC1 residual estimates | `sec5_empirical_raw_pc1/communication_services_pc1_m3_m5_reoptimization/` | Additional M3 optimizer search and the dependent M5 fit, with audit records |

## Reproduce the published tables

Python 3.10 or later is required. Table generation uses only Python's standard
library; neither market data nor LaTeX is needed.

```console
python reproduce.py verify
python reproduce.py tables
```

The second command regenerates all nine numerical tables (seven main-paper and
two supplementary tables), checks the numerical assertions in the original
builder, and checks the generated tables against the SHA256 hashes recorded in
`manifest.json`. Table hashes use LF line endings on every operating system.
The source hashes and validation counts, including 380 valid main-sample Wald
tests, must also match the manifest. Output is written to `code/paper/tables/`.
The generated TeX files stay local and are not distributed with this package.
This command reproduces tables from saved simulation and empirical results.
The numerical commands below reestimate the models separately and write their
new results to `code/work/`.

## Simulation evidence

Install the numerical dependencies in a separate Python environment:

```console
python -m venv .venv
```

Activate `.venv` (`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS), then install the dependencies:

```console
python -m pip install -r requirements.txt
```

Use the activated environment for the subsequent numerical commands. The
dependency file lists minimum versions; full model reestimation has not been
validated in a fresh environment for this initial package.
The exact original Python and numerical package versions are recorded in
`code/paper/results/sec4_simulation_m1_m5/design.json`.

```console
python reproduce.py simulation --smoke
python reproduce.py simulation --workers 3
```

The full command runs 100 replications for each of five DGPs, two dependence
settings and N = 20, 50, 100: 3,000 settings with M1 through M5 fitted in each.
Training, OOS and burn-in lengths are 1,886, 642 and 500; the seed is 20261006.
The fitted M1 determines the Fisher geometry in each setting. The frozen
calibration estimates are included, so this experiment needs no market data.
The smoke command uses one small iid static design and cannot reproduce the
paper's reported numbers. Outputs and resumable checkpoints are kept under
`code/work/`; full recomputation is a substantial computational task.
The command succeeds only when every setting has all five fitted models,
all ten numerically valid Wald tests and finite OOS scores for every model.
Otherwise it returns a nonzero exit code. `run_status.json` distinguishes
`processing_complete` from `validation_passed` and lists failed settings.
Failure records and checkpoints are retained for diagnosis.

## Empirical evidence

The paper uses 19 cross sections, each with matched raw and training-fixed PC1
residual inputs, for 38 fitted panels. PCA means and loadings use dates through
29 December 2023. Parameter Wald tests use lag zero; forecast differences use
HAC lag 22. Communication Services PC1 residual M3 and its dependent M5 include
an additional optimizer search, documented in the reoptimization folder above.

Follow the collection steps below to download prices, construct the 19 return
panels and prepare the 38 estimation inputs. Then run
`python reproduce.py empirical --workers 3`. By default it reads the newly
prepared manifest at `code/data/reader/input_manifest.json` and writes
fits and summaries to `code/work/empirical/`. With a custom collection
workspace, pass its manifest using `--manifest`.

Reference input hashes are recorded in
`code/paper/results/sec5_empirical_raw_pc1/input_manifest.json`.
A fresh download can differ from the paper's data vintage. The empirical
command estimates models on the collected inputs and then runs the additional
Communication Services PC1 residual M3 search and dependent M5 fit when that
panel is selected. It uses the reader's prepared inputs and initial estimates.
If the initial starts do not converge, it retries using the original
experiment's additional starts, with the same objective, bounds and acceptance
rules. These retries are recorded in each panel's `optimization_retry.json`.
It checks likelihood derivatives, Hessian curvature, unchanged M1/M2/M4 fits
and OOS scores before applying the correction to the reader's results. Initial
results, search records and the audit are retained under
`code/work/empirical/reoptimization/`. Final aggregate CSVs are written to
`code/work/empirical/tables/`. Missing fits, invalid Wald tests or missing OOS
scores cause a nonzero exit code.

If the initial estimation is already complete, resume the reoptimization,
audit and aggregation with:

```console
python reproduce.py empirical --finalize-only
```

Use the same `--manifest`, `--output` and `--datasets` arguments as in the
initial run if those were customized. Frozen manuscript estimates and
aggregate results remain under `code/paper/results/`. The `tables` command
uses those frozen results. It does not substitute a new data vintage into the
published tables.

The prepared NPZ format contains finite flattened `values`, integer daily
`offsets` with length `len(dates)+1`, and ascending `dates`. Daily observations
occupy `values[offsets[t]:offsets[t+1]]`. Training ends on 29 December 2023.
The manifest contains sample sizes, date ranges and input hashes for each panel.

## Collect data from the original providers

Obtain data directly from the sources below. Install the numerical
dependencies above and Git, and run the commands from this package's directory.

| Input | Original source | Version used in the paper |
| --- | --- | --- |
| Daily constituent prices | [Yahoo Finance](https://finance.yahoo.com/), using its chart response; [historical-download help](https://help.yahoo.com/kb/finance/download-historical-data-yahoo-finance-sln2311.html) | Daily prices queried from 20 June 2016 through 29 July 2026, including earlier prices needed to calculate the first return |
| S&P 500 membership history | [fja05680/sp500](https://github.com/fja05680/sp500), file `S&P 500 Historical Components & Changes (Updated).csv`; [download CSV](https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv) | Most recent snapshot on or before each return date; retained source ends on 30 June 2026, carried forward to 24 July 2026 |
| Nikkei 225 membership | Official [component CSV](https://indexes.nikkei.co.jp/nkave/archives/file/nikkei_stock_average_weight_en.csv) and [component-change ledger](https://indexes.nikkei.co.jp/nkave/archives/file/history_of_nikkei_stock_average_component_changes_en.pdf) | Component set dated 31 August 2026, reversed and replayed using the event ledger in the collection source code |
| US sector classifications | Git history of [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies), with the audited rules in `code/analysis/build_sp500_n225_sector_daily_2016_2026.py` | Git revision `493df85c63794d76e8e01a3d0d2ac3c36c55fb59`; individual snapshot dates and revisions are in `source_records/sp500_classification_snapshot_commits.csv` |
| Japanese sector classifications | The official Nikkei component CSV above, plus explicit classifications for 43 former constituents in the collection source code | Six Nikkei sectors; current classifications and documented historical assignments |

`data_sources.json` contains the download URLs, original source hashes, date
limits and Git revision in machine-readable form:

```console
python reproduce.py collect sources
```

### Download prices and reconstruct membership

```console
python reproduce.py collect sp500 --workers 3
python reproduce.py collect nikkei --workers 3
```

These commands use the original price-download and return-construction
functions, with outputs isolated under `code/data/reader/`. They download each
historical constituent's daily price history, preserve fetch failures in status
files, and retain returns only while the security belongs to the index. The
S&P 500 command collects only the US equity panel, without downloading the
unrelated ADR samples in the original research project. Japanese stock codes
use Yahoo's `.T` suffix; US class-share tickers use Yahoo's hyphen convention.

The provider links are updated over time. In particular, a newer Nikkei
component CSV cannot simply be combined with the frozen event ledger. The
collector checks the component date and stops if it is not 31 August 2026.
Use the matching historical source files when available:

```console
python reproduce.py collect sp500 --membership-csv path/to/historical_membership.csv
python reproduce.py collect nikkei --components-csv path/to/components_20260831.csv --history-pdf path/to/component_changes.pdf
```

If a different source is already cached, use a fresh workspace by appending
`--workspace code/data/reader_vintage` to each collection stage. When a retained
historical document is no longer available from its provider, obtain that
version from the corresponding author. The links above point to the original
providers.

For returns, the downloader selects adjusted closes when the response contains
that series, otherwise reported closes, and computes `log(P_t/P_previous)`.
Both prices must be finite and positive, the gap must be at most seven calendar
days, and the current observation must have positive volume. Missing histories
and delisted securities without prices stay missing. No balanced panel,
winsorization or return imputation is introduced. The Nikkei builder excludes
dates with fewer than 150 observed constituents. The paper then restricts the
analysis to 5 July 2016 through 24 July 2026.

### Construct the 17 sector panels

Clone the classification repository with its complete history and check out
the recorded revision. Do not use a shallow clone or the latest membership
table as a substitute for historical classifications.

```console
git clone https://github.com/datasets/s-and-p-500-companies.git code/data/reader/data/_external/s-and-p-500-companies
git -C code/data/reader/data/_external/s-and-p-500-companies checkout --detach 493df85c63794d76e8e01a3d0d2ac3c36c55fb59
python reproduce.py collect sectors
```

The sector builder uses dated US classification snapshots, the documented
sparse-history backcasts, the 2016/2018/2023 GICS transition rules, ticker
aliases and explicit historical overrides. These rules form an audited public
reconstruction of classifications. For Japan it uses the Nikkei 36-industry to
six-sector mapping and the former-constituent assignments in the same source
file. It writes the source revisions, classification spells and overrides
beside the sector returns. The density models use the `log_return` column;
the auxiliary standardized-return columns are not model inputs. US Real Estate
begins on 1 September 2016.

### Prepare the matched raw and PC1 residual inputs

```console
python reproduce.py collect pca
python reproduce.py empirical --workers 3
```

PCA uses each panel's training returns through 29 December 2023. For PCA only,
missing training returns are replaced by that constituent's observed training
mean. Constituents without training observations or positive training variation
are excluded. There is no volatility standardization. The leading covariance
eigenvector has unit norm and a nonnegative loading sum; means and loadings
remain fixed during OOS evaluation. On each date the factor is the
least-squares projection onto the loadings for observed eligible constituents;
residuals subtract this projection and the fixed training means. A date needs
at least two eligible observations and a squared loading norm of at least
`1e-12`. Raw and residual inputs contain the same constituent-dates. Missing
values imputed for PCA never enter the density model.

The output is `code/data/reader/input_manifest.json` plus 38 NPZ files under
`code/data/reader/prepared/`. The manifest records new input hashes and whether
each matches the paper's retained input. The empirical command above uses
these new hashes and writes to `code/work/empirical/`; it does not substitute
saved manuscript estimates. The training cutoff remains 29 December 2023;
OOS starts on 2 January 2024 in the US and 4 January 2024 in Japan, with 642
and 624 paper OOS dates respectively.

### Check against the paper

Preserve source versions and inspect fetch-status logs before interpreting
differences. Revised adjusted prices, unavailable delisted histories, newer
membership snapshots and classification changes can alter a fresh sample.
The source catalog records the reference vintage; the reader collector records
the actual source hashes. The 17 sector return panels were rebuilt from the
retained broad-market panels with exact numerical matches. PCA preparation was
checked using all 19 retained
source return panels: all 38 inputs have identical dates and daily offsets,
36 have identical file hashes, and the largest return-value discrepancy in
the other two residual panels was `3.34e-16` (floating-point precision).
The full provider download and subsequent model reestimation have not been
rerun for this package. Reproducing the original published tables from the
included saved results remains available with `python reproduce.py tables`.

## Contents and provenance

- `reproduce.py`: reader commands for verification, table generation, data
  collection and numerical runs.
- `code/`: frozen numerical sources and their transitive local dependencies.
  Original directory names are retained to preserve imports and provenance.
  Some imported modules have older experiment names; the simulation entry point
  is `run_symmetric_t_full_stage_m1.py`, and the empirical entry point is
  `run_case1_training_fixed_pca_wald.py`.
- `code/paper/results/`: saved table inputs, validation records, empirical
  estimates and the documented reoptimization, organized by the paper
  correspondence above. The DGP reference is used to check the simulation
  specification; the M1–M5 simulation results are in `sec4_simulation_m1_m5/`.
- `manifest.json`: the JFEC and arXiv PDF hashes in `manuscripts`, frozen file
  hashes, correspondence by section title in `paper_sections`, and
  `table_checks` containing source hashes, generated-table hashes and validation
  counts.
- `data_sources.json` and `source_records/`: collection URLs, original data
  vintage, source hashes and classification revision records.

Numerical source files are copied rather than linked to files outside this
folder. Changes to the working repository do not automatically update this
snapshot. `.gitattributes` preserves original file bytes because source
fingerprints depend on them. Original runs used Windows; fingerprints that
encode operating-system path separators can differ on another OS even when
the model code and data are unchanged. Floating-point libraries and optimizer
versions can also affect reestimated values.


Distributed paths and path literals use this naming convention, including
the numerical-source snapshots and calibration inputs. Historical source
hashes in saved run records identify the original run sources. The current
file hashes in `manifest.json` verify the distributed files. Renaming does
not change the observations, fitted numerical results or reference tables.
