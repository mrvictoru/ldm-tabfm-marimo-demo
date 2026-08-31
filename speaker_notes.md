# Speaker notes: Foundation models for tabular data

These notes assume the audience has not read the IBM LDM article and has no
prior knowledge of TabFM. The presentation first explains the general idea of a
tabular foundation model, then uses the insurance notebook as a concrete demo.

The notes are organized in presentation order:

1. walk through the seven introductory slides,
2. then walk through the notebook by section.

IBM's article is background inspiration for the decision-support interaction,
not a prerequisite for understanding the presentation. It does not disclose
that IBM uses TabFM or establish IBM's underlying model architecture.

---

# Part I: Slide-by-slide presentation

## Slide 1: Foundation models for tables

### What the slide shows

The title expands the audience's idea of a foundation model beyond language and
images. It introduces three concepts:

- structured tabular data,
- a pretrained model,
- and prediction from labeled context examples.

### Say

> "When people hear foundation model, they usually think about text generation or
> image generation. But much of an organization's operational data is stored in
> tables: customers, transactions, insurance quotes, claims, machines, and
> products."

> "A tabular foundation model is pretrained to recognize patterns across tabular
> tasks. We can then show it labeled rows from a new table and ask it to predict
> the outcome for unseen rows."

> "Today I will explain how that works, how it differs from training a conventional
> model for one dataset, what it may be useful for, and what still has to be
> validated."

### Transition

> "First, we need to be clear about what makes a table different from text."

---

## Slide 2: A table is not a block of text

### What the slide shows

The slide defines the basic supervised tabular prediction problem:

- one row represents one case,
- feature columns describe the case,
- the target column contains the outcome,
- and the task is to predict that target for an unseen row.

### Say

> "A table has explicit structure. Each row might be a customer or quote, while
> each column has a particular meaning: age, mileage, region, vehicle type,
> premium, and so on."

> "Those columns are heterogeneous. Some are continuous numbers, some are
> categories, some are booleans, and some values may be missing. The model has to
> reason about their relationships while respecting the row and column structure."

> "The target is the outcome we want to estimate. A classification target could be
> accepted versus rejected. A regression target could be cost, demand, or claim
> severity."

### Simple example

> "If historical rows include quote attributes and an accepted column, an unseen
> row contains the same quote attributes but no known accepted value. The model's
> job is to estimate that missing outcome."

### Transition

> "Now let us look at where the foundation-model part enters."

---

## Slide 3: Pretrain on many tasks, adapt through context

### What the slide shows

The slide separates two stages:

1. pretraining, which happens before this notebook and produces reusable model
   weights;
2. in-context prediction, where labeled rows from the current table describe
   the new task.

### Say

> "The expensive general learning happens during pretraining across many tabular
> tasks. We then load those existing weights rather than initialize and train a
> new neural network for our insurance table."

> "For the new task, we provide labeled context rows. Each context row includes
> the features and the known target. We then provide query rows that have the
> same feature schema but whose targets we want to predict."

> "TabFM uses the context to infer what relationship is being requested and
> returns predictions for the query rows."

### Explain `fit()` carefully

> "The API uses the familiar method name `fit`, but in this workflow it prepares
> the table and stores the labeled context. It does not update the pretrained
> TabFM weights."

This is zero-shot in the **weight-update sense**. It does not mean that the
model receives no examples: the 100 labeled context rows are essential.

### Important distinction

This is not ordinary few-shot prompting with sentences. The context is a typed
table with feature columns and labels, and the model was designed for tabular
prediction.

### Transition

> "The easiest way to understand the practical difference is to compare the two
> modeling workflows."

---

## Slide 4: Traditional ML versus the TabFM path

### What the slide shows

Both approaches start with data and end with validated predictions, but they
start the modeling step from different places.

### Say

> "In a traditional ML pipeline, the team prepares features, selects candidate
> algorithms, trains task-specific parameters, tunes hyperparameters, and compares
> the resulting models."

> "With TabFM, we begin with pretrained model weights, supply context rows, and
> immediately obtain candidate predictions. That can reduce the amount of custom
> model-building required before we learn whether a use case is promising."

### What does not disappear

Both paths still require:

- a meaningful and correctly defined target;
- clean, representative, legally usable data;
- separation between context or training rows and holdout rows;
- appropriate baselines and metrics;
- monitoring, security, governance, and human oversight.

### Keep the comparison honest

> "A foundation model is not automatically the best model. Logistic regression,
> gradient-boosted trees, or another task-specific method may be cheaper, faster,
> easier to explain, or more accurate."

The benefit being tested is **reuse and speed to a candidate baseline**, not
guaranteed superiority.

### Transition

> "So where might this different starting point be useful?"

---

## Slide 5: One interface, many tabular tasks

### What the slide shows

The slide gives examples of classification and regression tasks and then states
where a tabular foundation model may be especially worth testing.

### Say

> "Classification asks which category applies: will a customer churn, is a
> transaction suspicious, will a borrower default, or will a customer accept an
> offer?"

> "Regression estimates a numeric outcome: sales, demand, cost, duration, or
> severity."

> "A reusable tabular model is particularly interesting when a team wants a rapid
> baseline, has many related prediction tasks, has limited labeled context, or
> wants to test an interactive decision workflow before investing in a bespoke
> model."

### Selection criteria

Do not choose TabFM based only on model novelty. Compare:

- predictive quality and calibration;
- inference latency and computational cost;
- available context size and data scale;
- explainability and audit requirements;
- licensing and deployment restrictions;
- performance of simpler conventional methods.

### Licensing boundary

The published TabFM weights used by this notebook are non-commercial. The demo
must not be presented as a deployable commercial insurance solution.

### Transition

> "The notebook uses one insurance table and two targets to make this concrete:
> quote acceptance, then expected loss cost."

---

## Slide 6: Demo - score a quote and test alternatives

### What the slide shows

The slide combines the Demo 1 workflow with its observed holdout metrics:

- TabFM ROC AUC: `0.622`;
- logistic-regression ROC AUC: `0.592`;
- no-skill ROC AUC: `0.500`;
- TabFM and logistic average precision are almost equal.

It also notes that the notebook then reuses the **same quote** for named-offer
ranking and expected-loss regression. Do **not** invent Demo 2 numbers on the
slide; read MAE live from the notebook.

### Say

> "Our first synthetic task asks whether a customer will accept an
> automobile-insurance quote. TabFM receives 100 labeled context rows and scores
> unseen holdout quotes."

> "The ROC AUC of 0.622 is above the no-skill value of 0.5 and modestly above the
> same-context logistic regression result of 0.592. Average precision is nearly
> identical for TabFM and logistic regression. This is a modest positive result,
> not a dramatic win."

> "After that evidence gate, an agent can enter an eligible quote, see its
> acceptance score, retrieve similar historical cases through a separate
> nearest-neighbor layer, and compare alternative discounts, deductibles, and
> named offers."

> "The notebook then asks a second question of the same table: expected loss
> cost. That uses the same 100 context rows, a regression checkpoint, and its
> own evidence gate. Do not quote a Demo 2 metric until that gate has run."

### State the workflow boundary

The first target is **customer acceptance**, not underwriting approval. The
second target is **expected loss cost**, not an approved price. The demo does
not help an agent persuade a review team to accept an ineligible customer.
Underwriting, eligibility, pricing, fairness, compliance, and margin
constraints would be separate controls applied before or around the scenario
analysis.

### State the evidence boundary

- The dataset is synthetic.
- Classification metrics demonstrate modest held-out ranking signal for this run.
- Regression metrics must be read from the live notebook; do not invent them.
- Similar-case retrieval is not generated by TabFM.
- Named offers are model-ranked packages, not approved products.
- What-if score changes are model sensitivity, not causal effects.
- The result does not establish production accuracy, calibration, or ROI.

### Transition

> "The final slide separates the foundation-model promise from the evidence
> needed to trust itâ€”including when we reuse the same table for a new target."

---

## Slide 7: Pretrained does not mean pre-validated

### What the slide shows

The slide summarizes the reusable technical loop:

1. load a pretrained model;
2. provide labeled context rows;
3. predict unseen query rows;
4. compare those predictions with baselines on held-out data;
5. treat the result as useful only if the evidence supports it.

Reuse across tasks is the promise. A new target on the same table still needs
its own evidence gate.

### Say

> "The promise is reuse. Instead of starting every tabular task with newly
> trained model parameters, we start with a pretrained model that can interpret
> a new task from labeled context."

> "The practical benefit may be faster experimentation and a shorter path to a
> credible candidate baseline. In the notebook, that reuse is concrete: we keep
> the quote table and swap the label from acceptance to expected loss."

> "But pretrained does not mean proven for our table, and it does not mean
> proven for the next target either. Trust still comes from unseen data,
> appropriate baselines, uncertainty checks, calibration testing, operational
> constraints, and governance."

### Closing claim

> "A tabular foundation model gives us a new starting point, not an automatic
> production solution. The right question is not 'Is foundation AI impressive?'
> It is 'On this table, for this target, under these constraints, does the
> pretrained model provide enough measured value to justify using it?'"

### Optional IBM context

If the IBM article is relevant to the audience, add only:

> "IBM's LDM article inspired the decision-support interaction used in this
> demo. The article does not disclose TabFM or establish that IBM's
> implementation uses the same architecture."

---

# Part II: Notebook walkthrough

The deck establishes the idea first. Now switch to the Molab notebook and walk
through it **by section**, not by stale cell numbers. Markdown cells introduce
each step; executable cells stay visible.

Before a live presentation:

- run every cell once, including Demo 2;
- allow time for **two** TabFM checkpoints (classification, then regression);
- keep the Molab session open.

If the session is slow, pause after Demo 1. The regression checkpoint loads
only when Demo 2 runs.

---

## Setup and framing

### What it does

The PEP 723 block and import cell load marimo, pandas, NumPy, sklearn, and
TabFM. Opening markdown states the IBM interaction pattern, announces **two
tasks**, and repeats the architecture boundary.

### Say

> "This notebook is designed for Molab, so the Chromebook does not need local
> Python. The inline dependency block tells the hosted runtime what to install."

> "We recreate the quote-acceptance workflow first. Then we ask a second
> question of the same table: expected loss cost. That is the reuse claim from
> the slides, made concrete."

### Emphasize

- The classification checkpoint may take time to download.
- The regression checkpoint downloads later, in Demo 2.
- A hosted GPU is helpful if Molab offers one.
- `fit()` stores context; it does not update pretrained weights.

### What not to claim

Do not say IBM uses TabFM. Do not say zero-shot means no examples.

### Relevant code

```python
from tabfm import TabFMClassifier, TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
```

---

## Synthetic quote history

### What it does

`make_quote_history()` builds 1,800 reproducible rows. Each row has quote
attributes plus two hidden targets:

- `accepted` â€” noisy classification label;
- `expected_loss_cost` â€” noisy regression label driven by risk and coverage,
  **not** by `discount_pct`.

### Say

> "IBM's production example uses proprietary historical quote data, so we cannot
> reproduce it here. The generator gives us a controlled table. The model sees
> examples and labels, not the formulas used to generate the outcomes."

> "There are two hidden processes. Acceptance depends on price, deductible,
> tenure, bundling, and a few nonlinear interactions. Expected loss depends on
> risk and coverage. Discount is deliberately absent from the loss process so
> we can show that extra discount mainly moves acceptance, not cost."

### What not to claim

This is synthetic evidence. It demonstrates mechanism and evaluation, not real
insurance performance. The 1,800 rows are not IBM's reported 15 million quotes.

### Relevant code

```python
def make_quote_history(rows: int = 1800, seed: int = 7):
    ...
    frame["accepted"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    frame["expected_loss_cost"] = (
        (expected_loss + rng.normal(0, 35, rows)).clip(lower=40)
    ).round(0)
```

---

## Demo 1: will the customer accept this quote?

### What it does

Defines one feature schema for classification, regression, and retrieval.
`accepted` is the Demo 1 label. `expected_loss_cost` is **not** a feature.

Then the notebook:

1. creates a stratified 75/25 train/holdout split;
2. samples 100 stratified context rows from the training portion;
3. loads the TabFM **classification** checkpoint;
4. calls `fit` on the context rows;
5. scores the untouched holdout;
6. fits an untuned logistic regression and a class-prior dummy on the same
   100 rows.

Demo 2 later reuses `X_context` and `X_test` with cost labels from the same
indices.

### Say

> "This is the table contract. Both demos receive the same quote attributes.
> Demo 1 predicts whether the customer accepted the quote."

> "The holdout rows are not shown to TabFM as context. The 100 rows are a
> representative context sample, not a conventional training set. No TabFM
> weights are updated here."

> "Logistic regression sees the same small context table. That comparison
> answers whether TabFM is doing something useful rather than merely producing
> a plausible-looking probability."

### What not to claim

Do not call this â€œtraining TabFMâ€ without qualification. Say â€œpreparing the
context and scoring with TabFM.â€ Do not describe acceptance as underwriting
approval.

### Relevant code

```python
tabfm_classifier = TabFMClassifier(
    model=tabfm_v1_0_0.load(model_type="classification")
)
tabfm_classifier.fit(X_context, y_context.to_numpy())
y_cost_context = y_cost.loc[X_context.index]
y_cost_test = y_cost.loc[X_test.index]
```

---

## Evidence gate: classification

### What it does

Reports ROC AUC, average precision, AP lift over prevalence, and a bootstrap
interval for TabFM ROC AUC. The gate passes only if the AUC interval sits
above 0.5 and average precision beats holdout prevalence.

### Say

> "A probability alone is not enough. Average precision must be compared with
> how common acceptance is. ROC AUC must beat 0.5. The interval makes us less
> likely to celebrate a chance fluctuation."

> "On the recorded synthetic run, TabFM reached about 0.622 ROC AUC versus
> 0.592 for logistic regression and 0.500 for the no-skill baseline. That is a
> modest pass, not a dramatic win. If this run fails the gate, stop claiming
> useful Demo 1 scores."

### What not to claim

Do not treat a live quote percentage as calibrated. Do not say TabFM beat a
tuned production model.

---

## Live quote, next-best offers, and similar cases

### What it does

After the gate, the notebook shows:

- candidate-quote controls;
- the current TabFM acceptance score;
- a discount Ã— deductible what-if grid;
- named **next-best offers** scored by the same classifier
  (current terms, extra discount, higher deductible, home bundle, coverage
  upgrade â€” skipping options the quote already uses);
- similar historical quotes from a separate nearest-neighbor index.

### Say

> "The intended user is an agent discussing terms with a customer. The score
> estimates whether the customer will accept the quote, not whether
> underwriting will approve the risk."

> "The named-offer table is the same classifier asked a product question:
> extra discount, higher deductible, home bundle, or a coverage upgrade. These
> are model-ranked alternatives, not approved packages."

> "Similar quotes come from standardized distance over attributes. That
> retrieval layer is sklearn, not TabFM. Do not describe it as the foundation
> model explaining itself."

### Who uses this workflow?

An insurance **agent, broker, or sales/service advisor**. Underwriting,
eligibility, fraud checks, margin, and approval authority remain separate
controls. Use quote comparison only among terms that are eligible to offer.

### What not to claim

What-if changes are model sensitivity, not causal effects. Offers are not
products. Retrieval is not TabFM.

### Relevant code

```python
offer_quotes = build_named_offers(candidate_quote.iloc[0])
offer_quotes["tabfm_acceptance_probability"] = (
    tabfm_classifier.predict_proba(offer_quotes[feature_columns])[
        :, positive_index
    ]
)
```

---

## Demo 2: expected loss cost

### What it does

Loads a **second** checkpoint with `model_type="regression"` and
`TabFMRegressor`. Fits on the same 100 context rows using `expected_loss_cost`.
Compares holdout predictions with a mean dummy and untuned linear regression.

### Say

> "We did not train a new network. We swapped the label column and asked a
> different question of the same table."

> "This cell may pause while the regression weights download. That is why we
> kept Demo 1 on the classification checkpoint only."

> "The mean baseline always predicts the average cost. Linear regression is
> the simple conventional comparison. TabFM has to beat the mean before we
> treat live cost numbers as demonstrated signal."

### What not to claim

Do not read a Demo 2 MAE from the slides. Do not call expected loss an
approved premium, a claim reserve, or an underwriting decision.

### Relevant code

```python
tabfm_regressor = TabFMRegressor(
    model=tabfm_v1_0_0.load(model_type="regression")
)
tabfm_regressor.fit(X_context, y_cost_context.to_numpy())
```

---

## Evidence gate: regression

### What it does

Reports MAE, RMSE, and MAE lift versus the mean baseline. The gate passes only
if TabFM MAE is **strictly below** the mean baseline. A second label notes
whether TabFM is within 10% MAE of linear regression.

### Say

> "Read the table aloud. If TabFM's MAE is below the mean baseline, we can use
> the live cost as a second-task demonstration on this synthetic data. If it
> is not, say so and do not treat the dollar amount as evidence."

> "Even a pass is not a production result. It is the same discipline we used
> for classification, applied to a numeric target."

### What not to claim

Do not invent a number if the cell has not run. Do not claim TabFM is better
than a tuned GBM.

---

## Tradeoff: acceptance versus expected loss

### What it does

Scores the **same** candidate, term grid, and named offers with the
regressor. Extra discount should mainly lift acceptance; a higher deductible
can move both numbers because deductible is in both hidden processes.

### Say

> "Now the agent sees two numbers on one quote: chance the customer accepts,
> and expected loss associated with those terms."

> "Change the discount. Acceptance should move more than cost, because
> discount is not in the hidden loss formula. Change the deductible. Both
> scores may move."

> "These are model sensitivities, not causal effects, and they are not an
> instruction to sell the cheapest or the most acceptable package."

### What not to claim

Do not present the tradeoff as optimization, pricing authority, or fairness
clearance.

---

## Limits

### What it does

States what the prototype establishes if **both** gates pass, and what it
does not.

### Say

> "The strongest conclusion is conditional. If both evidence gates pass, this
> synthetic experiment shows that TabFM can provide held-out signal from a
> small context table on two targets without task-specific weight updates."

> "It does not prove that IBM uses TabFM, that the synthetic relationship
> matches real customers, that the scores are calibrated, or that TabFM beats
> tuned production models."

### Close the notebook

> "The next credible step is to replace the synthetic generator with a
> governed historical quote table that includes both labels, repeat both
> evidence gates, and only then consider a controlled business experiment."

---

# Final claims checklist

## Safe claims

- The demo recreates the insurance interaction pattern described by IBM.
- TabFM is a pretrained foundation model for tabular classification and
  regression.
- TabFM uses labeled rows as context without task-specific weight updates.
- The notebook evaluates unseen rows before interpreting live scores.
- One feature schema supports two tasks; each task has its own evidence gate.
- The UI combines scoring, named-offer ranking, what-if exploration, and
  separate retrieval.
- Similar-case retrieval is a nearest-neighbor layer, not TabFM.
- Molab lets the notebook run from a Chromebook without local Python.

## Claims to avoid

- IBM SQL Data Insights is powered by TabFM.
- IBM necessarily uses a transformer.
- Synthetic results reproduce IBM's 7% closing-rate improvement.
- A raw TabFM percentage or dollar amount is automatically calibrated.
- Similar-case retrieval is TabFM's explanation.
- Named offers are approved products or underwriting decisions.
- Expected loss is a causal effect of changing terms.
- TabFM is better than conventional models without a benchmark.
- Passing Demo 1 means Demo 2 is validated.
- Zero-shot means no examples, no validation, or no data preparation.
- The public TabFM weights may be used commercially.

## Sources

- IBM LDM article:
  <https://www.ibm.com/think/news/meet-large-database-models-ldms>
- Google Research TabFM article:
  <https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/>
- TabFM repository:
  <https://github.com/google-research/tabfm>
- TabFM PyTorch model card:
  <https://huggingface.co/google/tabfm-1.0.0-pytorch>
- Molab guide:
  <https://github.com/marimo-team/marimo/blob/main/docs/guides/molab.md>
