# IBM LDM-style quote workflow with Google TabFM

This repository is an independent technical recreation of the insurance workflow
described in IBM's article [Meet large database models
(LDMs)](https://www.ibm.com/think/news/meet-large-database-models-ldms), implemented
with Google's [TabFM zero-shot foundation model for tabular
data](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).

The demo lets a user:

1. enter an insurance quote,
2. estimate its acceptance probability,
3. compare TabFM with no-skill and conventional baselines,
4. retrieve similar historical quotes,
5. change the deductible or discount,
6. and recalculate candidate outcomes.

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
dependencies and TabFM PyTorch checkpoint to download. Before a live presentation,
run every cell once and keep the Molab session open.

## Optional local run

If Python is available on another machine:

```powershell
pip install -r requirements.txt
marimo edit ldm_tabfm_marimo_demo.py
```

## Evidence gate

The notebook does not call TabFM useful merely because it emits a probability. It
reports:

- ROC AUC and average precision on a held-out set,
- a bootstrap confidence interval for TabFM ROC AUC,
- a no-skill class-prior baseline,
- an untuned logistic regression trained on the same 100 context rows,
- average-precision lift over the holdout prevalence.

If the evidence gate does not pass, the notebook explicitly instructs the
presenter not to claim useful predictive performance for that run.

## Using real data

Replace `make_quote_history()` with a governance-approved historical quote table
using the same feature and target schema. Keep the train/context/holdout separation
and rerun the evidence gate. Do not present synthetic results as evidence of real
customer behavior.
