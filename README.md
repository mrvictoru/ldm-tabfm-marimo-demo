# IBM LDM-style quote workflow with Google TabFM

This repository is an independent technical recreation of the insurance workflow
described in IBM's article [Meet large database models
(LDMs)](https://www.ibm.com/think/news/meet-large-database-models-ldms), implemented
with Google's [TabFM zero-shot foundation model for tabular
data](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).

The notebook lets a user:

**Demo 1 — quote acceptance**

1. enter an insurance quote,
2. estimate its acceptance probability,
3. compare TabFM with no-skill and conventional baselines,
4. retrieve similar historical quotes,
5. compare deductible, discount, and named next-best offers,
6. and recalculate candidate outcomes.

**Demo 2 — expected loss cost**

1. reuse the same 100 context rows and the same candidate quote,
2. load TabFM as a regressor on `expected_loss_cost`,
3. compare MAE / RMSE with a mean baseline and untuned linear regression,
4. and view acceptance versus expected cost on the same term grid and offers.

## Important scope

This project does **not** claim that IBM SQL Data Insights uses TabFM or the same
architecture. It recreates the public interaction pattern with an independently
available tabular foundation model.

The bundled quote history is synthetic because IBM's reported insurance data is
proprietary. The notebook therefore demonstrates software behavior and evaluation
discipline, not production insurance performance or business ROI.

TabFM's pretrained weights are non-commercial and must not be used in commercial
or production settings.

## Run in Molab (Chromebook-friendly)

Open `ldm_tabfm_marimo_demo.py` in
[Molab](https://molab.marimo.io/). Molab runs the notebook in a hosted Python
environment, so a Chromebook does not need Python installed locally. The inline
PEP 723 dependency block tells the hosted runtime which packages to install.

Use a hosted GPU runtime if one is available. On the first run, allow time for the
dependencies and **two** TabFM PyTorch checkpoints (classification, then
regression) to download. Before a live presentation, run every cell once,
including Demo 2, and keep the Molab session open.

## Optional local run

If Python is available on another machine:

```powershell
pip install -r requirements.txt
marimo edit ldm_tabfm_marimo_demo.py
```

## Evidence gates

The notebook does not call TabFM useful merely because it emits a probability or
a dollar amount. Each target has its own gate.

**Classification (Demo 1)** reports:

- ROC AUC and average precision on a held-out set,
- a bootstrap confidence interval for TabFM ROC AUC,
- a no-skill class-prior baseline,
- an untuned logistic regression trained on the same 100 context rows,
- average-precision lift over the holdout prevalence.

**Regression (Demo 2)** reports:

- MAE and RMSE on the same held-out rows,
- a no-skill mean baseline,
- an untuned linear regression trained on the same 100 context rows,
- MAE lift versus the mean baseline.

If a gate does not pass, the notebook instructs the presenter not to claim useful
predictive performance for that task on that run. Passing Demo 1 does not
validate Demo 2.

## Using real data

Replace `make_quote_history()` with a governance-approved historical quote table
that includes **both** labels (`accepted` and a numeric loss or cost target)
and the same feature schema. Keep the train/context/holdout separation and rerun
**both** evidence gates. Do not present synthetic results as evidence of real
customer behavior.
