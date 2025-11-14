# AdRank-ML: Search Ads CTR Prediction & Ranking Simulation

This repository contains an end-to-end simulation of a search advertising ranking system.

It covers:

- **Prediction models** for:
  - Click-through rate (CTR)
  - Conversion probability (CVR)
  - Quality Score (QS proxy)
- **Auction simulation** using a Generalized Second Price (GSP)-style mechanism.
- **Bid optimization** using Bayesian optimization.
- **Data pipelines** using SQL + PySpark over synthetic **100M impression logs**.
- **Experiment design** with incrementality analysis, CUPED variance reduction, and A/B testing.



## Project Goals

1. Build predictive models that estimate:
   - `P(click | query, ad, user, context)`
   - `P(conv | click, query, ad, user, context)`
   - A proxy `QualityScore` combining relevance + predicted CTR.

2. Simulate a **GSP-style auction**:
   - Rank by `bid × QualityScore` (or `bid × pCTR`).
   - Charge second-price-like CPC.

3. Optimize bids with **Bayesian optimization** to maximize advertiser ROI or platform revenue.

4. Design experiments to measure:
   - Lift in CTR / CVR
   - Revenue/ROI impact
   - Variance reduction using **CUPED**
   - Incrementality beyond baseline models.

---

## Repo Structure

```text
src/
  config.py        # Global config (paths, seeds, hyperparams)
  data_loader.py   # Load raw/processed data
  features.py      # PySpark feature engineering pipelines
  models.py        # CTR/CVR/QS models (logistic regression, GBDT)
  auction.py       # GSP-style auction + ranking position bias
  optimization.py  # Bayesian optimization of bids
  experiments.py   # A/B tests, CUPED, incrementality calculations
  utils.py         # Common helpers (logging, metrics, seeds)
sql/
  create_impressions_table.sql
  feature_aggregation.sql
notebooks/
  01_eda_feature_eng.ipynb
  02_modeling_ctr_cvr.ipynb
  03_gsp_auction_simulation.ipynb
  04_bid_optimization_experiments.ipynb
