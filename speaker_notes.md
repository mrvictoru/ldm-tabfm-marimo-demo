# Speaker notes: recreating IBM's LDM quote workflow with TabFM

## 1. The short version

Use this as the opening:

> "IBM's article describes a system that uses historical insurance quotes to
> estimate whether a customer will accept a new quote, retrieve similar prior
> cases, and recalculate the odds when a salesperson changes the deductible or
> discount. This notebook recreates that interaction pattern with Google's
> pretrained TabFM model."

Then state the boundary:

> "This is an independent recreation. I am not claiming that IBM SQL Data
> Insights uses TabFM, or even that it uses the same model architecture. My
> hypothesis is that a zero-shot tabular foundation model is a plausible way to
> implement the kind of workflow IBM describes, and this demo tests that idea."

That distinction is essential. IBM's public article explains the product behavior
and business result, but it does not identify TabFM as its underlying model.

---

## 2. What IBM's article actually describes

The strongest concrete example in the IBM article is Swiss Mobiliar's insurance
quote workflow:

1. A salesperson enters a candidate automobile insurance quote.
2. The system finds similar previous cases.
3. It estimates the probability that the customer will accept the quote.
4. The salesperson changes terms such as the deductible or discount.
5. The system recalculates the acceptance odds.
6. The salesperson uses those comparisons to choose a more tailored quote.

IBM reports that the production system used approximately 15 million historical
quote records with several dozen attributes. IBM also reports a 7% improvement in
the closing rate over six months.

Do not imply that this notebook reproduces that scale or result. The notebook
reproduces the **decision-time loop**:

> current case -> predicted outcome -> similar cases -> alternative terms ->
> recalculated outcome

That is the LDM-style interaction pattern we are demonstrating.

---

## 3. Why TabFM is relevant

Google describes TabFM as a foundation model for classification and regression on
tabular data. It supports mixed numerical and categorical columns and uses
in-context learning.

The important technical distinction:

> "`TabFMClassifier.fit()` does not train new model weights for this insurance
> dataset. It prepares the columns and supplies labeled historical rows as context
> to the pretrained model. Predictions are produced without task-specific
> fine-tuning or hyperparameter search."

This aligns with the part of the LDM vision concerned with moving rapidly from a
historical table to a predictive workflow. It does **not** mean that evaluation,
data quality, governance, or domain expertise disappear.

Google's published TabFM weights are non-commercial. This prototype is for
technical demonstration and evaluation, not deployment.

---

## 4. What changed from the earlier version

The earlier notebook led with fraud detection. Its holdout ROC AUC was 0.467 and
its average precision was approximately equal to fraud prevalence. Those results
did not demonstrate a useful TabFM signal. A probability appearing in the UI was
therefore easy to mistake for evidence that the model worked.

The rewritten notebook:

- uses IBM's primary insurance example instead of an indirect fraud example,
- evaluates TabFM before presenting its current-case score,
- compares it with a no-skill class-prior baseline,
- compares it with an untuned logistic regression using the same 100 rows,
- includes a bootstrap confidence interval for ROC AUC,
- labels the run as a pass or "not established,"
- separates TabFM scoring from nearest-neighbor retrieval,
- and clearly marks the historical quote data as synthetic.

The presentation should never skip the evidence gate.

---

## 5. How to explain the synthetic data

IBM's 15 million quote records are proprietary, so this repository cannot use
them. The notebook creates a reproducible table with realistic-looking fields:

- customer and vehicle attributes,
- prior claims and customer tenure,
- coverage tier and region,
- reference and quoted premiums,
- deductible and discount,
- and whether the quote was accepted.

The accepted/rejected outcome comes from a hidden nonlinear and noisy process.
TabFM is not given that formula. It sees only labeled examples and must recover a
pattern that generalizes to held-out rows.

Say:

> "Synthetic data lets us test the mechanics reproducibly, but it is not evidence
> of performance on real customers. The credible next step is to replace this
> generator with an approved historical quote table and rerun the same evaluation."

Do not describe synthetic results as proof of ROI, production readiness, or
insurance-domain accuracy.

---

## 6. The evidence gate

The benchmark table includes three signals:

1. **TabFM** using 100 labeled context rows without weight updates.
2. **Untuned logistic regression** trained on the same 100 rows.
3. **Class-prior baseline**, which assigns everyone the same probability.

### ROC AUC

ROC AUC asks whether accepted quotes tend to receive higher scores than rejected
quotes across possible thresholds.

- `1.0`: perfect ranking
- `0.5`: random ranking
- below `0.5`: worse than random ranking

The notebook also reports a bootstrap 95% interval. Its evidence gate requires the
lower end of that interval to exceed `0.5`. This is intentionally stricter than
celebrating one point estimate above chance.

### Average precision

Average precision focuses on the precision-recall trade-off and must be compared
with the positive-class prevalence.

For example, if 40% of held-out quotes are accepted:

- an average precision near `0.40` is roughly no-skill,
- `0.60` is meaningful lift,
- and the notebook reports the ratio as AP lift over prevalence.

### How to present the verdict

If the notebook says:

> **PASS: held-out TabFM signal demonstrated**

you may say:

> "On this synthetic held-out set, TabFM learned a ranking signal from 100 context
> rows without dataset-specific weight training."

If it says:

> **NOT ESTABLISHED**

say:

> "The workflow runs, but this execution did not establish useful predictive
> performance. We should not interpret the individual probabilities as reliable."

Do not hide or talk around a failed gate.

---

## 7. Why show logistic regression

The conventional baseline prevents the demo from implying that any non-random
score makes TabFM special.

Both models receive the same 100 labeled rows:

- TabFM supplies them as context to a pretrained foundation model.
- Logistic regression estimates task-specific coefficients from those rows.

Possible interpretations:

- **TabFM beats logistic regression:** strong evidence of useful zero-shot
  performance on this synthetic task.
- **They are within 0.03 ROC AUC:** the notebook labels TabFM competitive. This
  demonstrates comparable ranking with less task-specific model setup, not
  predictive superiority.
- **Logistic regression is clearly better:** TabFM is not the best model for this
  task as configured. The LDM-style UI remains valid, but the TabFM value claim is
  weak.
- **Neither beats no-skill:** the run demonstrates only the interface.

The comparison is intentionally modest. It is not a substitute for tuned boosted
trees, cross-validation, calibration analysis, or a production benchmark.

---

## 8. The live quote result

The current result shows:

- TabFM acceptance probability,
- the quote's percentile among held-out TabFM scores,
- historical acceptance prevalence,
- quoted premium,
- deductible,
- and discount.

The percentile is useful because a raw probability can be misread as calibrated.
For example:

> "This quote is at the 80th percentile of model scores"

is a ranking statement. It is often safer than:

> "This customer has exactly an 80% chance of accepting."

Only describe the score as model-driven evidence if the holdout gate passed.

---

## 9. The what-if table

The notebook generates alternatives across discounts and deductibles, then asks
TabFM to rescore them.

Say:

> "This is the behavior IBM describes: a salesperson can change commercial terms
> and immediately compare the modeled acceptance odds before choosing an offer."

Also state:

> "The highest modeled acceptance probability is not automatically the best
> business quote. A real solution must include margin, underwriting, fairness,
> compliance, and customer-treatment constraints."

Watch whether the results behave sensibly:

- larger discounts should generally not reduce modeled acceptance,
- deductible effects may depend on customer and coverage attributes,
- and repeated or erratic reversals should be treated as a warning.

What-if outputs show model sensitivity, not causality. Historical association does
not prove that changing a term will cause the predicted improvement.

---

## 10. Similar historical quotes

The retrieval layer uses standardized Euclidean distance over numeric and
one-hot-encoded categorical fields.

It is deliberately separate from TabFM.

Say:

> "TabFM provides the acceptance score. A conventional nearest-neighbor index
> retrieves similar historical quotes. The notebook combines those components,
> but it does not pretend that TabFM performs every function."

The similar rows help a user inspect:

- whether comparable cases exist,
- what terms they received,
- and whether they accepted.

Similarity is contextual evidence, not an explanation of TabFM's internal
reasoning and not a guarantee of the same outcome.

---

## 11. Molab and Chromebook setup

This project is designed to run in Molab's hosted environment. The Chromebook
only needs a browser; it does not need a local Python installation.

Before presenting:

1. Open the notebook in Molab.
2. Allow the inline dependencies to install.
3. Use a hosted GPU runtime if available because the PyTorch checkpoint is large.
4. Run all cells once so the TabFM weights are downloaded and cached.
5. Confirm the evidence-gate result.
6. Exercise the discount and deductible controls.
7. Keep the Molab tab open and avoid relying on a fresh model download during the
   live presentation.

The model weights use a non-commercial license, so this is a demonstration rather
than a commercial deployment.

---

## 12. Suggested presentation flow

### Opening

> "IBM describes an insurance quoting system that turns a historical database
> into decision support at the moment a salesperson is preparing an offer."

### Hypothesis

> "Google's TabFM suggests a practical way to recreate that interaction pattern:
> provide examples from a new table as context and make predictions without
> training task-specific model weights."

### Data boundary

> "IBM's real records are proprietary, so this notebook uses synthetic quote
> history. It can demonstrate the mechanism, not IBM's production result."

### Evaluation

> "Before showing a probability, we test whether TabFM beats chance on unseen
> rows, show uncertainty, and compare it with a conventional baseline."

### Live interaction

> "Now we enter a quote, adjust the discount or deductible, recalculate the score,
> and inspect similar historical cases."

### Close

> "The demo supports the architectural hypothesis if TabFM passes the held-out
> evidence gate. The business hypothesis still requires representative real data,
> governance, and a production experiment."

---

## 13. Claims checklist

### Safe claims

- The notebook recreates the interaction pattern described in IBM's article.
- TabFM is a pretrained foundation model for tabular classification and
  regression.
- TabFM uses labeled rows as context without task-specific weight updates.
- The notebook evaluates held-out ranking before interpreting live scores.
- The UI combines model scoring, what-if exploration, and separate case retrieval.
- Molab lets the demo run from a Chromebook without local Python.

### Claims to avoid

- IBM SQL Data Insights is TabFM.
- IBM necessarily uses a transformer.
- Synthetic performance proves production insurance performance.
- A displayed probability is automatically calibrated or useful.
- Similar-case retrieval explains TabFM.
- TabFM is better than traditional models without a benchmark.
- Zero-shot means no examples, no evaluation, or no data work.
- The public TabFM weights can be used commercially.

---

## 14. Source links

- IBM: <https://www.ibm.com/think/news/meet-large-database-models-ldms>
- Google Research:
  <https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/>
- TabFM repository: <https://github.com/google-research/tabfm>
- TabFM PyTorch model card:
  <https://huggingface.co/google/tabfm-1.0.0-pytorch>
- Molab guide:
  <https://github.com/marimo-team/marimo/blob/main/docs/guides/molab.md>
