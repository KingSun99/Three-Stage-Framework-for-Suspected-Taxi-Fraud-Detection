# Data Availability

The raw July and August 2026 High Volume For-Hire Vehicle trip records are publicly released by the NYC Taxi and Limousine Commission on its [Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Download the monthly Parquet files and place them under `data/raw/` using the filenames documented in `README.md`.

This repository includes the frozen derived cohorts and feature tables needed for the listed classifier evaluations, along with the fitted joint CTGAN and PCTGAN checkpoints for seeds 42–46. It does not include the full monthly raw files or all generated synthetic rows. The derived cohorts preserve the study's frozen IDs, feature construction, candidate labels, and evaluation populations. These labels represent rule- or cluster-derived order-pattern candidates and should not be interpreted as confirmed fraud.

Before public release, verify the TLC dataset's current terms and the redistribution rights for derived trip-level records. Do not include any manually annotated review data unless the reviewers' contribution and public-release permissions have been confirmed.
