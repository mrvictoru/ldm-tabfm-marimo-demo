# Speaker notes: IBM LDM-style quote workflow with TabFM

These notes are organized in the same order as the presentation:

1. walk through the seven slides,
2. then walk through the notebook cell by cell.

The central story is IBM's insurance example: use historical quotes to estimate
acceptance, retrieve similar prior quotes, and test different commercial terms
before choosing an offer.

---

# Part I: Slide-by-slide presentation

## Slide 1: From dashboards to decision support

### What the slide shows

The title introduces the shift from retrospective reporting to an interactive
decision workflow.

### Say

> "Most analytics teams are good at explaining what already happened: dashboards,
> reports, trends, and summaries. IBM's LDM article describes a different
> direction: using historical records to help someone decide what to do in a live
> case."

> "This demo applies that idea to insurance quoting. We enter a candidate quote,
> estimate the likelihood of acceptance, inspect similar historical quotes, and
> test alternatives such as a different deductible or discount."

### Important boundary

> "This is an independent recreation of the interaction pattern. It is not IBM's
> SQL Data Insights product, and IBM's article does not say that IBM uses TabFM."

### Transition

> "First, let us make the LDM idea concrete."

---

## Slide 2: Report the past or assist the decision?

### What the slide shows

The left side represents traditional analytics:

- what happened,
- how many,
- where did performance change,
- output: a dashboard.

The right side represents decision-time analytics:

- what does this new case look like,
- how promising is it,
- what similar cases exist,
- what happens if an input changes.

### Say

> "The distinction is not that dashboards are unhelpful. The distinction is when
> the analytics becomes useful. A dashboard summarizes history. An LDM-style
> workflow brings historical intelligence into the moment when a person is
> preparing a decision."

> "For an insurance salesperson, the question is not only what the portfolio did
> last month. It is: for this customer and this quote, what acceptance outcome is
> plausible, what similar quotes have worked, and what if I change the terms?"

### Explain the histogram

> "The historical records create the reference distribution. The new quote is
> placed into that context rather than being presented as an isolated number."

### Transition

> "The next slide separates what we are reproducing from what we are not claiming."

---

## Slide 3: We reproduced the pattern, not the architecture

### What the slide shows

The slide lists the workflow:

1. enter a candidate insurance quote,
2. estimate customer acceptance,
3. retrieve similar historical quotes,
4. change the deductible or discount,
5. validate the score on held-out data.

### Say

> "The IBM article gives us a business interaction pattern, not enough technical
> detail to reconstruct IBM's internal architecture. We therefore reproduce the
> observable workflow with public tools."

> "TabFM supplies the tabular prediction. A separate nearest-neighbor layer
> retrieves comparable quotes. Marimo supplies the interactive controls."

### Be precise about the hypothesis

> "The hypothesis is that a zero-shot tabular foundation model is a plausible way
> to get from a historical table to this kind of decision-support prototype
> quickly. The notebook tests that hypothesis instead of assuming it."

### Do not say

- "IBM uses TabFM."
- "This is IBM SQL Data Insights."
- "The notebook reproduces IBM's 15 million-row production system."

### Transition

> "Now let us look at why a tabular foundation model is relevant."

---

## Slide 4: Rows, columns, attributes—not essays

### What the slide shows

The code loads TabFM, supplies historical quote rows as context, and scores
unseen rows.

### Say

> "TabFM is designed for structured data: numerical and categorical columns such
> as age, vehicle type, coverage, premium, deductible, and discount."

> "The important detail is that this is zero-shot in the model-training sense.
> The `fit` call prepares the table and supplies labeled examples as context. It
> does not update the pretrained model weights for this particular quote table."

> "That is attractive for rapid prototyping because we can try a new table
> without building a task-specific neural network or running a hyperparameter
> search first."

### Caveat

> "Zero-shot does not mean no data preparation or no validation. We still need a
> representative context, a clean target, a separate holdout, and evidence that
> the predictions are better than chance."

### Transition

> "That leads to the feature-engineering question."

---

## Slide 5: Less handcrafted modeling—not zero

### What the slide shows

The slide contrasts the traditional modeling path with the promise of a
pretrained tabular model.

### Say

> "The benefit we are testing is reduced custom modeling work, not the elimination
> of all data work."

> "The notebook still defines the quote schema, creates the premium fields, checks
> the target, separates context from holdout data, and builds the retrieval
> features. TabFM reduces the need to train and tune a new model from scratch; it
> does not remove domain knowledge or governance."

### Better claim

> "The honest claim is that a pretrained model may shorten the path from a
> historical table to a credible prototype. Whether it is accurate enough for a
> real business use case must be measured."

### Transition

> "The next slide is the live interaction that mirrors IBM's insurance example."

---

## Slide 6: Enter a quote, compare terms, recalculate

### What the slide shows

The slide shows a quote being built, scored, and compared with alternative
deductibles and discounts. It also labels retrieval as a separate component.

### Say

> "This is the decision-time loop. We enter a quote, calculate its premium,
> obtain a TabFM acceptance score, and then generate alternative commercial
> terms."

> "The presenter can show a concrete what-if question: what happens to the
> modeled acceptance probability if the discount increases, or if the deductible
> changes?"

> "The similar-quote table provides historical context. It is not an explanation
> of TabFM's internal reasoning; it is a separate retrieval tool that helps a
> salesperson inspect comparable outcomes."

### How to handle the result

Before discussing a live percentage, point to the evidence status:

> "The notebook first checks whether TabFM has a held-out signal. If the evidence
> gate fails, the displayed probability is a UI output, not a validated business
> signal."

If the gate passes:

> "On this synthetic holdout, TabFM demonstrated a ranking signal. That supports
> the prototype hypothesis, but it is not evidence of production calibration or
> customer ROI."

If the gate does not pass:

> "The workflow runs, but this execution did not establish useful predictive
> performance. We can still demonstrate the interaction, but we should not sell
> the score as reliable."

### Business caveat

> "The highest acceptance score is not automatically the best quote. A real
> decision would also consider margin, underwriting rules, fairness, compliance,
> and customer treatment."

### Transition

> "The final slide summarizes the architecture and the limits of the claim."

---

## Slide 7: What an LDM-style system does inside a business

### What the slide shows

The slide summarizes the reusable pattern:

1. start with structured historical data,
2. score a current case,
3. retrieve similar prior cases,
4. edit inputs and rerun,
5. validate evidence before trusting the score.

### Say

> "The reusable idea is not a particular screen or model name. It is the loop:
> historical records become context, a current case receives a score, similar
> cases make the output concrete, and what-if edits support a decision."

> "TabFM is one possible scoring component. Retrieval and UI are separate
> components. The combination is what makes this an LDM-style prototype."

### Closing claim

> "If TabFM passes the evidence gate, this prototype supports the idea that a
> pretrained tabular model can quickly power this interaction on the synthetic
> task. The business claim still requires real quote data, governance, and a
> controlled production experiment."

---

# Part II: Notebook cell-by-cell walkthrough

The deck establishes the idea first. Now switch to the Molab notebook and walk
through the cells in execution order.

## Cell 1: Imports and inline dependencies

### Code

The first lines contain the PEP 723 dependency block. The import cell loads
marimo, pandas, NumPy, sklearn, and TabFM.

### Say

> "This notebook is designed for Molab, so the Chromebook does not need local
> Python. The inline dependency block tells the hosted runtime what to install."

> "We use TabFM for zero-shot tabular classification, sklearn for evaluation and
> comparison, and marimo for the reactive interface."

### Emphasize

- The PyTorch checkpoint may take time to download.
- A hosted GPU is helpful if Molab offers one.
- Run the notebook once before presenting.

### Relevant code

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.13", "numpy>=1.26", "pandas>=2.2",
#                 "scikit-learn>=1.5", "tabfm[pytorch]>=1.0.1"]
# ///
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
```

---

## Cell 2: Notebook framing

### What it does

This markdown cell introduces the insurance workflow and states that the project
does not reproduce IBM's product or architecture.

### Say

> "This is the notebook's contract with the audience: we are recreating the
> interaction pattern, not claiming access to IBM's implementation."

### Emphasize

IBM describes the behavior publicly. It does not identify TabFM as the model
behind SQL Data Insights.

### Relevant code

```python
mo.md("""
# Recreating IBM's LDM quote workflow with Google TabFM
...
""")
```

---

## Cell 3: What the demo is designed to prove

### What it does

This cell lists the evidence requirements:

- held-out signal,
- no-skill comparison,
- conventional comparison,
- decision-time behavior,
- historical grounding.

### Say

> "A probability alone is not enough. The notebook separates the user experience
> from the question of whether the model has learned a useful ranking."

> "The `fit` call supplies context rows to TabFM, but it does not fine-tune the
> pretrained weights."

### Relevant code

```python
mo.md("""
## What this demo is designed to prove
- Held-out signal
- No-skill and conventional comparisons
- Decision-time what-if behavior
""")
```

---

## Cell 4: Generate the historical quote table

### What it does

`make_quote_history()` creates reproducible synthetic insurance records with:

- customer age,
- vehicle age,
- annual mileage,
- prior claims,
- tenure,
- home-policy bundling,
- coverage tier,
- vehicle type,
- region,
- deductible,
- discount,
- reference premium,
- quoted premium,
- price-to-reference ratio,
- and an accepted/rejected outcome.

### Say

> "IBM's production example uses proprietary historical quote data, so we cannot
> reproduce it here. Instead, this generator gives us a controlled table where
> the label comes from a hidden nonlinear and noisy process."

> "The model sees examples and labels, not the formula used to generate the
> outcome. That lets us test whether it can recover a signal."

### Important limitation

> "This is synthetic evidence. It demonstrates the mechanism and the evaluation
> workflow, not real insurance performance."

### Relevant code

```python
def make_quote_history(rows: int = 1800, seed: int = 7):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({...})
    frame["quoted_premium"] = (
        frame["reference_premium"] * (1 - frame["discount_pct"] / 100)
    ).round(0)
    frame["accepted"] = rng.binomial(1, acceptance_probability)
    return frame
```

---

## Cell 5: Show the historical table statistics

### What it does

Displays row count, acceptance rate, and data provenance.

### Say

> "This gives the audience the base rate before we discuss average precision. A
> model's average precision must be compared with how common acceptance is."

> "The 1,800 rows here are a notebook-sized synthetic history, not IBM's reported
> 15 million quote records."

### Relevant code

```python
quote_history = make_quote_history()
mo.md(f"""
## Historical quote table
- **Rows:** `{len(quote_history):,}`
- **Accepted:** `{quote_history["accepted"].mean():.1%}`
""")
```

---

## Cell 6: Choose feature and target columns

### What it does

Builds `X` from quote attributes and `y` from `accepted`. It identifies the
categorical and numeric columns.

### Say

> "This is the table contract. The model receives the quote attributes, while
> acceptance is the label we want to predict."

> "We keep categorical fields such as coverage, vehicle, and region as categorical
> data. Numeric fields include premium, mileage, deductible, and discount."

### Emphasize

The `price_to_reference` field makes the commercial price relationship explicit.
This is still light feature engineering; zero-shot does not mean zero schema
design.

### Relevant code

```python
feature_columns = [
    "customer_age", "vehicle_age", "annual_mileage", "prior_claims",
    "tenure_years", "bundled_home", "coverage_tier", "vehicle_type",
    "region", "reference_premium", "deductible", "discount_pct",
    "quoted_premium", "price_to_reference",
]
X = quote_history[feature_columns].copy()
y = quote_history["accepted"].astype(int)
```

---

## Cell 7: Split data and run TabFM

### What it does

The cell:

1. creates a stratified 75/25 train/holdout split,
2. samples 100 stratified context rows from the training portion,
3. loads the TabFM classification checkpoint,
4. calls `fit` on the context rows,
5. scores the untouched holdout,
6. and keeps the acceptance-class probability.

### Say

> "The holdout rows are not shown to TabFM as context. That separation is what
> lets us ask whether the score generalizes beyond the examples it saw."

> "The 100 rows are a representative context sample, not a conventional training
> set. No TabFM model weights are updated here."

### Be careful

Do not call this “training TabFM” without qualification. Say “preparing the
context and scoring with TabFM.”

### Relevant code

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
X_context, _, y_context, _ = train_test_split(
    X_train, y_train, train_size=100, stratify=y_train, random_state=42
)
classifier = TabFMClassifier(model=tabfm_v1_0_0.load(
    model_type="classification"
))
classifier.fit(X_context, y_context.to_numpy())
scores = classifier.predict_proba(X_test)
```

---

## Cell 8: Fit the logistic comparison

### What it does

Builds a standard preprocessing pipeline and an untuned logistic-regression
classifier using the same 100 context rows and the same holdout.

### What `logistic_scores` means

`logistic_scores` is a pandas `Series` containing one estimated probability for
each row in `X_test`. In this notebook, the positive class is quote acceptance,
so a value such as `0.82` means:

> "Given the features of this quote and the examples in `X_context`, the
> logistic-regression benchmark estimates an 82% chance that the quote will be
> accepted."

It is not the final yes/no decision. A threshold such as `0.50` could turn the
probability into a class prediction, but the notebook keeps the continuous
scores because ROC AUC and average precision evaluate how well the model ranks
quotes across all possible thresholds.

For example, suppose the held-out data contains these three quotes:

| Test-row index | Actual outcome | `logistic_scores` |
| ---: | --- | ---: |
| 101 | accepted | 0.82 |
| 102 | rejected | 0.21 |
| 103 | accepted | 0.64 |

The scores rank the two accepted quotes above the rejected quote, which is the
kind of ordering ROC AUC rewards. They are not guaranteed to be perfectly
calibrated: a score of `0.82` does not mean that exactly 82 out of 100
otherwise identical quotes will be accepted.

The `[:, 1]` in `predict_proba(X_test)[:, 1]` selects the probability of the
positive class, `accepted = 1`, rather than the probability of rejection.
Because the Series uses `index=X_test.index`, each score can be matched back to
the corresponding held-out quote.

### Say

> "This comparison answers whether TabFM is doing something useful rather than
> merely producing a plausible-looking probability."

> "Both methods see the same small context table. Logistic regression is a simple
> conventional baseline, not a fully tuned production competitor."

### Interpret possible outcomes

- TabFM stronger: evidence of useful zero-shot signal on this synthetic task.
- Similar performance: TabFM may reduce setup, but is not demonstrably superior.
- Logistic regression stronger: TabFM is not the best choice as configured.
- Both near chance: the interface works, but the predictive claim fails.

### Relevant code

```python
logistic_classifier = make_pipeline(
    logistic_preprocessor,
    LogisticRegression(max_iter=1_000),
)
logistic_classifier.fit(X_context, y_context)
logistic_scores = pd.Series(
    logistic_classifier.predict_proba(X_test)[:, 1],
    index=X_test.index,
    name="logistic_acceptance_probability",
)
```

---

## Cell 9: Define evaluation helpers

### What it does

Defines metrics and a bootstrap confidence interval for TabFM ROC AUC.

### Say

> "ROC AUC measures ranking: do accepted quotes tend to receive higher scores
> than rejected quotes? A value of 0.5 is random ranking."

> "Average precision focuses on the precision-recall trade-off. We compare it with
> the acceptance prevalence rather than treating 0.5 as a universal baseline."

> "The bootstrap interval makes the evidence gate less dependent on one lucky
> point estimate."

### Relevant code

```python
def evaluate_signal(labels, scores):
    return {
        "ROC AUC": roc_auc_score(labels, scores),
        "Average precision": average_precision_score(labels, scores),
    }

tabfm_auc_low, tabfm_auc_high = bootstrap_auc_interval(y_test, tabfm_scores)
```

---

## Cell 10: Build the benchmark and evidence gate

### What it does

Calculates metrics for:

1. TabFM,
2. logistic regression,
3. a class-prior no-skill baseline.

It then labels the result:

- `PASS: held-out TabFM signal demonstrated`,
- `SIGNAL ONLY`,
- `COMPETITIVE`,
- or `NOT ESTABLISHED`.

### Say

> "This is the most important evaluation cell. We do not declare success because a
> number appeared on screen."

The signal gate requires:

- the lower bootstrap ROC AUC bound to exceed 0.5,
- and average precision to exceed holdout prevalence.

The competitive label additionally checks whether TabFM is within 0.03 ROC AUC
of logistic regression.

### Presentation rule

Always read this verdict before showing the live quote result. If it says
`NOT ESTABLISHED`, do not describe the live percentage as validated risk.

### Interpreting an example result

For example:

| Signal | ROC AUC | Average precision | AP lift |
| --- | ---: | ---: | ---: |
| TabFM | 0.622 | 0.762 | 1.12x |
| Logistic regression | 0.592 | 0.761 | 1.12x |
| Class-prior baseline | 0.500 | 0.682 | 1.00x |

This is a **modest positive result**, not a strong one:

- TabFM's ROC AUC of `0.622` is above random ranking (`0.500`), so accepted
  quotes tend to receive somewhat higher scores than rejected quotes.
- The logistic benchmark is also above chance at `0.592`. TabFM is ahead by
  `0.030` ROC AUC, which meets this notebook's "within 0.03" comparison rule
  after the signal gate passes. This does not establish a practically
  meaningful advantage over logistic regression.
- The holdout acceptance prevalence is `0.682`. TabFM's average precision of
  `0.762` is about 12% higher than that prevalence (`0.762 / 0.682 = 1.12`),
  indicating better-than-baseline ranking in the precision-recall view.
- The class-prior baseline has ROC AUC `0.500` and average precision equal to
  prevalence, as expected for a constant score that contains no quote-specific
  information.

The table alone does **not** tell us whether the evidence gate passed. We must
also inspect the bootstrap lower bound for TabFM ROC AUC. If that lower bound
is above `0.5`, this run supports the narrower claim "TabFM showed held-out
signal on this synthetic dataset." If it is at or below `0.5`, the apparent
advantage may be sampling noise and the notebook should report
`NOT ESTABLISHED`.

### Relevant code

```python
signal_demonstrated = (
    tabfm_auc_low > 0.5
    and tabfm_metrics["Average precision"] > prevalence
)
evidence_label = (
    "PASS: held-out TabFM signal demonstrated"
    if signal_demonstrated
    else "NOT ESTABLISHED: do not claim TabFM is useful on this run"
)
```

---

## Cell 11: Build the similarity layer

### What it does

Standardizes numeric fields, one-hot encodes categorical fields, and builds a
nearest-neighbor index over the full historical quote table.

### How the matrix and index work

The original quote table mixes numbers and text categories, but distance
calculation needs a numeric matrix. The `ColumnTransformer` creates one row per
historical quote and one numeric column per transformed feature:

- numeric fields such as age, mileage, premium, deductible, and discount are
  standardized so large-unit fields do not automatically dominate the distance;
- categorical fields such as coverage tier, vehicle type, and region are
  one-hot encoded, so each category becomes a `0` or `1` indicator column.

For example, a row with `vehicle_type = "Sedan"` might become indicators such
as `vehicle_type_Sedan = 1` and `vehicle_type_Truck = 0`. The resulting
`similarity_matrix` is therefore a rectangular array: rows are historical
quotes, and columns are the transformed numeric and categorical attributes.
It is a representation for retrieval, not a model prediction.

`NearestNeighbors` then builds `similarity_index` from that matrix. The index
stores the transformed historical rows in a structure that can efficiently
answer: "Which eight rows are closest to this new quote?" It does not train
TabFM, change the acceptance model, or calculate the TabFM probability.

At decision time, the candidate quote goes through the same preprocessor:

```python
candidate_vector = similarity_preprocessor.transform(
    candidate_quote[feature_columns]
)
distances, neighbor_positions = similarity_index.kneighbors(candidate_vector)
similar_quotes = quote_history.iloc[neighbor_positions[0]]
```

`candidate_vector` is one transformed row. `neighbor_positions` contains the
integer row positions of the eight nearest historical quotes; `distances`
contains their corresponding distance values. The notebook uses the positions
to retrieve rows from `quote_history` and displays them as "similar historical
quotes."

### What `metric="euclidean"` means

Euclidean distance is the straight-line distance between two transformed rows:

```text
sqrt((a1 - b1)^2 + (a2 - b2)^2 + ... + (an - bn)^2)
```

A smaller value means the quotes are more similar under this representation.
Because numeric fields are standardized and categorical fields are indicators,
the metric combines differences across all of those columns. It is a
transparent heuristic, not a learned notion of business similarity. Changing
the metric or feature weights could change which quotes are retrieved.

### Say

> "This is the retrieval component. It is intentionally separate from TabFM. It
> finds comparable quotes using a transparent distance calculation."

> "The similar cases provide context for a salesperson, but they are not a
> post-hoc explanation of the foundation model."

### Relevant code

```python
similarity_preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_columns),
    ("categorical", OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    ), categorical_columns),
])
similarity_matrix = similarity_preprocessor.fit_transform(X)
similarity_index = NearestNeighbors(n_neighbors=8, metric="euclidean")
similarity_index.fit(similarity_matrix)
```

---

## Cell 12: Create the quote controls

### What it does

Creates marimo controls for:

- coverage tier,
- vehicle type,
- region,
- customer age,
- vehicle age,
- annual mileage,
- prior claims,
- tenure,
- bundled home policy,
- deductible,
- and discount.

### Say

> "These controls represent the decision-time inputs a salesperson can discuss
> with a customer. The interface is what turns a batch prediction into an
> interactive workflow."

### Demo tip

Start from the seeded quote, then change only one or two terms at a time so the
audience can connect the input change with the recalculated output.

### Relevant code

```python
coverage = mo.ui.dropdown(
    options=sorted(quote_history["coverage_tier"].unique().tolist()),
    value=str(seed_quote["coverage_tier"]),
    label="Coverage tier",
)
customer_age = mo.ui.slider(
    21, 75, value=int(seed_quote["customer_age"]),
    label="Customer age",
)
```

---

## Cell 13: Arrange the controls

### What it does

Places the widgets into a readable candidate-quote panel.

### Say

> "This cell is presentation plumbing. It puts the business inputs together so
> the user can edit a quote without reading the implementation."

Move quickly through this cell.

### Relevant code

```python
discount_pct = mo.ui.slider(
    0, 20, value=int(seed_quote["discount_pct"]),
    label="Discount (%)", show_value=True
)
deductible = mo.ui.dropdown(
    options=[250, 500, 750, 1000, 1500],
    value=int(seed_quote["deductible"]),
    label="Deductible",
)
```

---

## Cell 14: Define the quote builder

### What it does

`build_quote()` applies the pricing formula and returns a one-row DataFrame with
the same schema used by the historical data.

### Say

> "The candidate quote must use the same columns and transformations as the
> historical table. This function keeps the interactive row compatible with the
> TabFM classifier and the similarity index."

### Important distinction

The pricing formula is a deterministic demo rule. The acceptance outcome is not
known for the candidate quote; TabFM supplies an estimate.

### Relevant code

```python
quoted_premium = reference_premium * (1 - discount_pct / 100)
return pd.DataFrame([{
    "quoted_premium": round(quoted_premium, 0),
    "price_to_reference": round(
        quoted_premium / reference_premium, 4
    ),
    # remaining quote fields use the same schema as history
}])
```

---

## Cell 15: Score the current quote

### What it does

Builds the current candidate row from the widgets and calls
`tabfm_classifier.predict_proba()`.

### Say

> "This is the live scoring moment. We construct one quote with the same schema as
> the historical examples and ask TabFM for its acceptance probability."

> "The percentage is a model output, not automatically a calibrated probability.
> The evidence gate and the score percentile help us interpret it more carefully."

### Relevant code

```python
candidate_probability = float(
    tabfm_classifier.predict_proba(candidate_quote[feature_columns])[
        0, positive_index
    ]
)
```

---

## Cell 16: Generate what-if scenarios

### What it does

Creates combinations of discounts and deductibles, scores them with TabFM, and
sorts the alternatives by predicted acceptance probability.

### Say

> "This directly recreates the quote-optimization behavior in IBM's article. We
> hold the customer and coverage context constant while testing different
> commercial terms."

> "These are modeled scenarios, not causal guarantees. A larger discount may be
> associated with higher acceptance in the historical data, but that does not
> prove that changing the discount will cause the same improvement for a real
> customer."

### What to inspect

- Does the ranking respond to discount and deductible changes?
- Are the changes directionally plausible?
- Are there erratic reversals that should be treated as model instability?

### Relevant code

```python
scenario_quotes["tabfm_acceptance_probability"] = (
    tabfm_classifier.predict_proba(
        scenario_quotes[feature_columns]
    )[:, positive_index]
)
scenario_results = scenario_quotes.sort_values(
    "tabfm_acceptance_probability", ascending=False
)
```

---

## Cell 17: Show the current quote result

### What it does

Displays:

- TabFM acceptance probability,
- percentile among holdout scores,
- historical acceptance rate,
- quoted premium,
- deductible,
- discount,
- and the evidence-gate interpretation.

### Say

> "This is the decision-support summary. Notice that we show the historical base
> rate and score percentile alongside the raw model output."

> "If the evidence gate failed, this cell explicitly tells us not to treat the
> percentage as demonstrated useful prediction."

### Avoid

Do not say:

> "This customer has exactly an X% chance of accepting."

Prefer:

> "The model assigns this quote an X% acceptance score, at the Yth percentile
> relative to the holdout scores."

### Interpreting an example result

For an output such as:

```text
TabFM acceptance probability: 70.2%
Holdout score percentile: 48.7%
Historical acceptance rate: 68.2%
Quoted annual premium: $950
Deductible: $500
Discount: 20%
```

the fields mean:

- **TabFM acceptance probability (`70.2%`)**: TabFM's score for this one
  candidate quote, with acceptance as the positive class. It is an estimated
  model score, not a guaranteed outcome and not necessarily a calibrated
  real-world probability.
- **Holdout score percentile (`48.7%`)**: the candidate's score is greater than
  or equal to approximately 48.7% of the held-out quote scores. This places it
  near the middle of the model's scoring range. It is a relative ranking
  statistic, not a 48.7% acceptance probability and not the percentage of
  customers who will accept.
- **Historical acceptance rate (`68.2%`)**: the fraction of held-out labels
  that are accepted. In this notebook it is the holdout prevalence used for
  comparison with the no-skill baseline. It is a portfolio-level base rate,
  not the candidate-specific prediction.
- **Quoted annual premium (`$950`)**, **deductible (`$500`)**, and **discount
  (`20%`)**: the commercial terms of the candidate row being scored. They are
  inputs to the score, not independent evidence that the customer will accept.

The 70.2% score is only slightly above the 68.2% base rate, so this particular
quote is not an unusually high-scoring case despite having a score above 70%.
The 48.7th percentile reinforces that interpretation: relative to the
holdout-score distribution, it is close to average.

Finally, **"The holdout evidence gate passed"** refers to the overall evaluation,
not a guarantee about this quote. The notebook reached that message because
TabFM's bootstrap ROC AUC lower bound exceeded 0.5 and its average precision
exceeded holdout prevalence. This supports the narrower claim that TabFM showed
held-out ranking signal on this synthetic dataset; it does not prove that this
individual quote will be accepted or that the model is ready for production.

### Relevant code

```python
score_percentile = float(
    (tabfm_scores <= candidate_probability).mean()
)
historical_rate = float(y_test.mean())
```

---

## Cell 18: Show what-if quote optimization

### What it does

Displays the highest-ranked candidate combinations of discount and deductible.

### Say

> "The table makes the scenario analysis concrete. It gives the salesperson a
> small set of alternatives to discuss rather than requiring a separate modeling
> workflow for every quote."

> "In a real implementation, this ranking would be constrained by margin,
> underwriting, fairness, compliance, and business policy."

### Relevant code

```python
display_scenarios = scenario_results.copy()
display_scenarios["tabfm_acceptance_probability"] = (
    display_scenarios["tabfm_acceptance_probability"]
    .map(lambda value: f"{value:.1%}")
)
mo.vstack([mo.md("## What-if quote optimization"),
           display_scenarios.head(10)])
```

---

## Cell 19: Show similar historical quotes

### What it does

Retrieves the nearest historical quote rows and displays their terms and
acceptance outcomes.

### Say

> "Now we add historical grounding. The user can see what similar quotes looked
> like and whether those prior quotes were accepted."

> "This is not TabFM explaining itself. It is a separate retrieval layer that
> makes the decision more inspectable."

### Relevant code

```python
candidate_vector = similarity_preprocessor.transform(
    candidate_quote[feature_columns]
)
_, neighbor_positions = similarity_index.kneighbors(candidate_vector)
similar_quotes = quote_history.iloc[neighbor_positions[0]]
```

---

## Cell 20: Final scope and limitations

### What it does

States what the prototype establishes and what it does not.

### Say

> "The strongest conclusion is conditional. If the evidence gate passes, this
> synthetic experiment shows that TabFM can provide a held-out signal from a
> small context table without task-specific weight updates."

> "It does not prove that IBM uses TabFM, that the synthetic relationship matches
> real customers, that the score is calibrated, or that TabFM beats tuned
> production models."

### Close the notebook

> "The next credible step is to replace the synthetic generator with a governed
> historical quote table, repeat the evaluation across time-based holdouts, and
> run a controlled business experiment."

### Relevant code

```python
mo.md("""
## What the prototype establishes—and what it does not
...
""")
```

---

# Final claims checklist

## Safe claims

- The demo recreates the insurance interaction pattern described by IBM.
- TabFM is a pretrained foundation model for tabular classification and
  regression.
- TabFM uses labeled rows as context without task-specific weight updates.
- The notebook evaluates unseen rows before interpreting live scores.
- The UI combines scoring, what-if exploration, and separate retrieval.
- Molab lets the notebook run from a Chromebook without local Python.

## Claims to avoid

- IBM SQL Data Insights is powered by TabFM.
- IBM necessarily uses a transformer.
- Synthetic results reproduce IBM's 7% closing-rate improvement.
- A raw TabFM percentage is automatically calibrated.
- Similar-case retrieval is TabFM's explanation.
- TabFM is better than conventional models without a benchmark.
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
