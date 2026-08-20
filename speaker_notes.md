# Speaker notes for the LDM-style TabFM demo

This document is meant to support a presentation, not just a notebook walkthrough. The goal is to help you explain **why this matters for analytics teams**, **how it connects to the IBM LDM concept**, and **why a pretrained tabular foundation model like TabFM is a practical way to demonstrate that idea today**.

---

## 1. Core presentation message

If you want one clear message for the audience, use this:

> "This demo shows that we can turn historical business data into an interactive decision-support system. Instead of only reporting the past, we can score a new case, compare it with similar historical cases, and test possible changes before making a decision."

That is the main value proposition.

For this audience, the important idea is not "we used Python."  
The important idea is:

- analytics can become interactive,
- historical data can become operational,
- and pretrained tabular AI can help smaller teams build advanced decision tools faster.

---

## 2. How to frame the IBM article

The IBM article describes **large database models (LDMs)** as systems that help users work directly with enterprise data in a smarter, more decision-oriented way.

The key ideas from the article that matter for this presentation are:

1. **Use historical records as intelligence, not just storage**  
   The database is not only where data sits. It becomes the source of examples, patterns, and comparisons.

2. **Evaluate a current case against historical context**  
   Instead of analyzing after the fact, the user can ask: "What is the likely outcome of this case right now?"

3. **Support scenario exploration**  
   A user can try alternative inputs and see how the outcome changes. In other words, the system helps with what-if thinking rather than only retrospective analysis.

4. **Bring back similar prior cases**  
   The system helps the user understand the current case in context.

5. **Move faster from idea to business value**  
   One of the most compelling points in the article is speed: teams can operationalize historical data more quickly.

When you present, say this plainly:

> "What IBM is pointing toward is a shift from dashboards that describe the past to systems that actively assist a decision in the present."

---

## 3. How our notebook maps to the LDM idea

Be precise here. Do not claim that the notebook is literally IBM's LDM system.

A good way to say it:

> "This notebook is an LDM-style demonstration. It recreates the business functions described in the article using a pretrained tabular model, interactive controls, and historical-case retrieval."

What we reproduced:

- **Fraud / anomaly triage**  
  Score a new transaction and surface suspicious patterns.

- **Scenario exploration**  
  Change transaction attributes and instantly recalculate the signal.

- **Similar historical cases**  
  Show comparable prior examples to support investigation.

- **Insurance quote optimization**  
  Change pricing-related inputs and estimate likelihood of acceptance.

- **Retail healthier-alternative recommendations**  
  Find similar products with a better health profile.

This is the key bridge statement:

> "The notebook is not trying to replicate IBM's product architecture. It is demonstrating the same business interaction pattern with tools we can access today."

---

## 4. Why TabFM matters in this story

This section is important because it explains why the demo is more than a toy.

TabFM matters because:

- it is a **pretrained foundation model for tabular data**,
- it reduces the need to build everything from scratch,
- it reduces how much manual model-building work has to happen before a team can start testing value,
- it is a credible example of how foundation-model thinking applies beyond text,
- and it helps show that advanced analytics can become more accessible even when local talent or ML infrastructure is limited.

Suggested wording:

> "A lot of AI discussion is dominated by text models. But many business decisions are not made from essays or chat logs. They are made from rows, columns, attributes, and historical records. TabFM is interesting because it is built for that kind of data."

You can also make the strategic point:

> "For teams in markets where deep ML specialization is still limited, pretrained tabular models are promising because they lower the barrier to building decision-support systems from structured business data."

---

## 5. The feature-engineering angle: why this is appealing

This is one of the strongest ideas to surface in the presentation.

Part of the appeal of the LDM vision is that it promises to reduce some of the traditional handwork usually associated with data-science projects:

- manual feature engineering,
- repeated feature extraction pipelines,
- custom model selection for every use case,
- and long delays between "we have the data" and "we have a usable decision tool."

That does **not** mean domain knowledge disappears. It means more of the intelligence can come from the model plus the historical data, rather than from months of bespoke feature crafting.

Suggested wording:

> "Traditionally, getting from database rows to a useful predictive workflow often requires a lot of data-science work: choosing features, engineering transformations, training custom models, and iterating many times."

Then:

> "What makes the LDM idea attractive is the possibility of skipping much of that custom effort, or at least reducing it significantly, by using a model designed to work directly with structured historical data."

### How honest we should be about this demo

Our notebook **partly** demonstrates that idea, but not perfectly.

It shows the promise because:

- we use a pretrained tabular model rather than building a bespoke model architecture from zero,
- we keep the workflow relatively close to the original transaction columns,
- and we get a working decision-support experience without a heavy, enterprise-scale feature platform.

But we should also acknowledge:

- the demo still adds a few simple derived fields, such as balance deltas and error measures,
- those fields are there to make the example clearer and stronger,
- so this is better described as **reduced manual feature engineering**, not **zero feature engineering**.

The best presentation line is:

> "This demo does not eliminate feature engineering entirely, but it does show the direction: less handcrafted modeling work, faster prototyping, and more value coming directly from structured historical data plus a pretrained tabular model."

---

## 6. How to close the deck

The last slide should feel like a **technical summary**, not a product pitch.

The main point to land is:

### An LDM-style system turns structured historical data into an interactive workflow

Instead of ending with a dashboard or a static prediction, the system can:

1. take a **live business case** as input,
2. compare it against **historical rows**,
3. produce a **score or signal**,
4. retrieve **similar prior cases** for context,
5. and let the user run **what-if edits** before acting.

That is the mechanism.

For this audience, say it plainly:

> "The interesting part is not just that a model gives a score. The interesting part is that historical data becomes operational context. A user can bring a new case, compare it with similar past records, test changes, and make a better-informed decision in the same workflow."

That maps well to the IBM article and the IBM video because both emphasize:

- structured rows and columns as the core data asset,
- similarity and retrieval over historical records,
- live scoring of a current case,
- and scenario exploration before action.

### How to explain our specific demo honestly

Be explicit that our prototype is built from three parts:

1. **TabFM** for tabular pattern recognition and scoring,
2. a **helper retrieval layer** for similar-case lookup,
3. **marimo** for the interactive what-if interface.

Suggested wording:

> "In this demo, TabFM handles the tabular scoring, a retrieval layer brings back comparable historical cases, and marimo provides the interface for interactive scenario testing."

Then add the business interpretation:

> "That combination is what makes the workflow useful. The analyst is not staring at an unexplained score. They can inspect comparable cases and test changes before deciding."

---

## 7. Best positioning for this audience

Your audience is in data analytics, but may not have deep machine-learning experience. So the tone should be:

- technical enough to feel credible,
- practical enough to feel achievable,
- and honest about what is automated versus what is still engineered.

Avoid sounding like you are selling a platform.

Avoid phrases that overclaim, such as:

- "this changes everything,"
- "fully replaces data scientists,"
- or "no feature engineering is needed."

Better framing:

> "This is a credible prototype of a new analytics workflow. It shows how a pretrained tabular model, historical record retrieval, and an interactive UI can be combined to support live decisions."

And:

> "The value is not magic automation. The value is that more of the pattern-recognition work can come from the model plus historical data, so teams can prototype decision-support workflows faster."

---

## 8. Suggested opening talk track

You can use something close to this at the start:

> "Most analytics teams today are very good at showing what already happened. Dashboards, reports, summaries, trends. Those are valuable, but they usually stop short of helping a user decide what to do in a live case."
>
> "What interested me about the IBM LDM article is that it describes a different pattern: use historical database records to actively support a decision while the user is making it."
>
> "This demo is my attempt to recreate that pattern in a practical way using a pretrained tabular foundation model called TabFM and an interactive marimo notebook."
>
> "What makes this especially interesting is that we can do this without building a completely custom ML stack from scratch. That is where pretrained tabular models become very relevant."
>
> "The point is not that this notebook is a production platform. The point is that the interaction model is powerful: score a case, compare it to similar past cases, and test possible changes immediately."

---

## 9. Cell-by-cell speaker guide

Use this while walking through the notebook.

### Notebook framing cells

Use the first three markdown sections as your setup story before you go step by step.

Code snippet:

```python
mo.md("""
# LDM-style fraud triage with TabFM
...
- **Risk scoring:** predict fraud probability for a transaction
- **What-if analysis:** change transaction attributes and recompute the score
- **Similar prior cases:** retrieve the closest historical transactions
""")
```

Say:

> "Before we get into the mechanics, the notebook tells the audience exactly what this is: an LDM-style workflow with scoring, what-if analysis, and similar-case retrieval."

What to emphasize:

- The notebook is organized around business interactions, not just model training.
- The "How this maps to the IBM article" table is useful for setting expectations early.
- The "How to read the outputs" section helps you define what the fraud score and rule-based flags mean before the live interaction starts.

### Step 1: Load the notebook tools

Code snippet:

```python
import marimo as mo
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
```

Say:

> "This is only setup. We load the notebook, data, modeling, and similarity tools. The point is not the imports themselves, but the workflow they enable."

What to emphasize:

- Move quickly here.
- The important idea is that one notebook combines UI, modeling, and retrieval.

### Step 2: Load a sample of transaction data

Code snippet:

```python
DATA_URL = (
    "https://huggingface.co/datasets/"
    "CiferAI/Cifer-Fraud-Detection-Dataset-AF/resolve/main/"
    "Cifer-Fraud-Detection-Dataset-AF-part-1-14.csv"
)
RAW_COLUMNS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
```

Say:

> "Here we point to the public fraud dataset and define the transaction columns we want to work with."

Then add:

> "In a real enterprise implementation, this would be your own historical case table."

What to emphasize:

- Historical rows are the raw material.
- The demo starts from structured business data, not unstructured text.

### Step 2 continued: enrich the transactions and build the what-if row logic

Code snippet:

```python
def _enrich_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    enriched["hour_of_day"] = enriched["step"] % 24
    enriched["origin_delta"] = enriched["oldbalanceOrg"] - enriched["newbalanceOrig"]
    enriched["dest_delta"] = enriched["newbalanceDest"] - enriched["oldbalanceDest"]
    enriched["origin_balance_error"] = (enriched["origin_delta"] - enriched["amount"]).abs()
    enriched["dest_balance_error"] = (enriched["dest_delta"] - enriched["amount"]).abs()
    enriched["log_amount"] = np.log1p(enriched["amount"])
```

Say:

> "This is the small amount of feature engineering in the demo. We create a few fields that capture whether the balance movement looks normal or suspicious."

Then add:

> "So the honest claim is not zero feature engineering. The honest claim is much less bespoke work than a traditional end-to-end modeling project."

What to emphasize:

- This is transparent, simple analytics engineering.
- The balance error fields are especially useful because they capture suspicious money-movement patterns.

### Step 2 continued: stream a balanced interactive sample

Code snippet:

```python
@functools.lru_cache(maxsize=1)
def load_cifer_sample(
    normal_rows: int = 2400,
    fraud_rows: int = 300,
    chunk_size: int = 150_000,
    random_state: int = 42,
) -> pd.DataFrame:
```

Say:

> "The notebook does not load the entire dataset. It streams the file in chunks and keeps a manageable interactive sample so the demo stays usable."

What to emphasize:

- This is a demo design decision, not a conceptual limitation.
- The sample still gives enough history for scoring and retrieval.

### Step 3: Create the working dataset for the demo

Code snippet:

```python
modeling_df = load_cifer_sample()
```

Say:

> "This gives us the working historical transaction table for the notebook. Think of it as the case history the model and retrieval layer can learn from."

What to emphasize:

- This is the historical memory of the workflow.
- Everything later depends on this table.

### Debug preview cell right after Step 3

Code snippet:

```python
print("Loaded dataframe columns:")
print(modeling_df.columns.tolist())
print("\nTop 5 rows:")
print(modeling_df.head(5).to_string(index=False))
```

Say:

> "This is just a quick inspection cell so we can confirm the dataset loaded correctly."

What to emphasize:

- You can scroll past this quickly in a presentation.
- It is useful as a sanity check, not as a main storytelling moment.

### Step 4: Review the sample statistics

Code snippet:

```python
fraud_rate = modeling_df["isFraud"].mean()
flagged_rate = modeling_df["isFlaggedFraud"].mean()
```

Say:

> "This gives us a quick profile of the sample: how much fraud is in it, and how often the built-in rule flag would trigger."

What to emphasize:

- This grounds the audience in the data mix.
- It also helps distinguish the simple rule flag from the learned model score.

### Step 5: Choose the features the model will use

Code snippet:

```python
feature_columns = [
    "step", "hour_of_day", "type", "amount", "log_amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud", "origin_delta", "dest_delta",
    "origin_balance_error", "dest_balance_error",
]
```

Say:

> "This cell defines the transaction attributes the model will consider when deciding whether a case looks fraud-like."

What to emphasize:

- These are still close to the original business columns.
- This is part of the reduced manual feature-engineering story.

### Step 6: Train the fraud-risk model and test it on held-out data

Code snippet:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
tabfm_model = tabfm_v1_0_0.load(model_type="classification")
classifier = TabFMClassifier(model=tabfm_model)
classifier.fit(X_context, y_context.to_numpy())
```

Say:

> "This is where TabFM learns from historical examples and turns those rows into a predictive fraud signal."

Then add:

> "The notebook also tries to make the context rows more representative by stratifying and matching the real fraud ratio, instead of feeding the model a simplistic random slice."

What to emphasize:

- Prediction is one component of the overall workflow.
- The important outcome is a usable signal, not a claim of perfect fraud detection.

### Step 7: Review how well the model performed on a test slice

Code snippet:

```python
tabfm_auc = roc_auc_score(y_test, fraud_scores)
tabfm_ap = average_precision_score(y_test, fraud_scores)
baseline_auc = roc_auc_score(y_test, baseline_scores)
baseline_ap = average_precision_score(y_test, baseline_scores)
```

Say:

> "These metrics are just a quick sanity check. They show that the model is producing a meaningful signal on held-out data."

What to emphasize:

- Keep this high level unless someone asks for details.
- The comparison with `isFlaggedFraud` is useful because it shows the notebook is doing more than a simple hard-coded rule.

### Step 8: Build the case-comparison layer

Code snippet:

```python
similarity_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ]
)
similarity_index = NearestNeighbors(metric="euclidean", n_neighbors=8)
```

Say:

> "This is the retrieval layer. It builds the similarity machinery that lets the notebook find the closest historical transactions for any new case."

What to emphasize:

- This is central to the LDM-style story.
- The audience should understand that the notebook is not only scoring, but also comparing.

### Step 9: Create the interactive controls

Code snippet:

```python
transaction_type = mo.ui.dropdown(...)
step_value = mo.ui.slider(...)
amount_value = mo.ui.number(...)
oldbalance_org_value = mo.ui.number(...)
oldbalance_dest_value = mo.ui.number(...)
origin_updates_normally = mo.ui.switch(...)
dest_updates_normally = mo.ui.switch(...)
```

Say:

> "This is where the workflow becomes operational. A user can directly edit the live case instead of passively looking at a report."

What to emphasize:

- This is the shift from analytics to decision support.
- Ask the audience to imagine a fraud desk, underwriting screen, or triage console.

### Step 10: Arrange the controls in one place

Code snippet:

```python
controls = mo.vstack(
    [
        mo.md("## Interactive what-if controls"),
        transaction_type,
        step_value,
        amount_value,
        oldbalance_org_value,
        oldbalance_dest_value,
        origin_updates_normally,
        dest_updates_normally,
    ]
)
```

Say:

> "This cell is simple, but it matters for usability. It puts the editable inputs into one clean control panel."

What to emphasize:

- The UI matters because the point is human decision support.
- A good interface is part of making historical data operational.

### Step 11: Score the edited transaction

Code snippet:

```python
candidate_frame = build_candidate_row(...)
candidate_features = candidate_frame[feature_columns]
candidate_probability = float(
    classifier.predict_proba(candidate_features)[0][positive_index]
)
_, neighbor_positions = similarity_index.kneighbors(candidate_similarity)
similar_cases = modeling_df.iloc[neighbor_positions[0]].copy()
```

Say:

> "This is the core live workflow: build a candidate case from the UI, score it with TabFM, and immediately pull back similar historical cases."

What to emphasize:

- This is the best place to explain the full loop.
- Prediction and retrieval happen together, not as separate disconnected tasks.

### Step 12: Show the current risk result

Code snippet:

```python
- **TabFM fraud probability:** `{candidate_probability:.1%}`
- **Rule-based flag (`isFlaggedFraud`):** `{current_flag}`
- **Origin balance error:** `{candidate_frame.loc[0, "origin_balance_error"]:.2f}`
- **Destination balance error:** `{candidate_frame.loc[0, "dest_balance_error"]:.2f}`
```

Say:

> "This score is a decision-support signal. Higher means the case looks more similar to past fraud-like cases. It is not an automatic verdict."

What to emphasize:

- Say the model "estimates" or "signals."
- Use the balance errors and rule flag as lightweight explanation aids.

### Step 13: Show similar past cases

Code snippet:

```python
similar_cases[["tabfm_demo_distance_rank", *DISPLAY_COLUMNS]]
```

Say:

> "This is one of the most important parts of the notebook. We are not only returning a score. We are returning comparable historical cases for context."

Then add:

> "That makes the system easier to trust because the analyst is not forced to treat the score as a black box."

What to emphasize:

- Similar-case retrieval is central to the LDM idea.
- This supports explanation, investigation, and confidence.

### Step 14: Show the context rows passed to TabFM

Code snippet:

```python
context_preview = modeling_df.loc[X_context.index].copy()
context_preview["context_label"] = y_context.values
```

Say:

> "This section makes the in-context learning setup more visible by showing examples the model actually saw as context."

What to emphasize:

- This helps demystify the model.
- It reinforces that the notebook is grounded in historical examples.

### Insurance extension overview

Start the extension with the notebook heading:

```python
mo.md("""
# Future extension 1: insurance quote optimization
""")
```

Say:

> "Now the notebook shows that the same interaction pattern can be reused for another business problem: insurance quote optimization."

### Insurance step 1: generate historical quote data

Code snippet:

```python
def make_insurance_dataset(rows: int = 1200, seed: int = 7) -> pd.DataFrame:
    ...
    quote_df["accepted"] = rng.binomial(1, probability)
```

Say:

> "This synthetic dataset stands in for a historical book of insurance quotes, including customer, vehicle, pricing, and whether the quote was accepted."

What to emphasize:

- This is a synthetic example, but the workflow is realistic.
- The same structure applies: historical rows, score, what-if edits, similar cases.

### Insurance step 2: fit a quote-acceptance model

Code snippet:

```python
(
    insurance_classifier,
    insurance_positive_index,
    ...,
) = fit_tabfm_binary(insurance_X, insurance_y, random_state=19)
```

Say:

> "This trains a model that estimates the chance a quote will be accepted."

Then add:

> "That matters because now a user can test quote changes before deciding what to offer."

What to emphasize:

- This directly matches one of the strongest IBM examples.
- The business value is optimization, not just prediction.

### Insurance step 3: edit a quote and rescore it

Code snippet:

```python
insurance_acceptance_probability = float(
    insurance_classifier.predict_proba(insurance_candidate_features)[0][
        insurance_positive_index
    ]
)
```

Say:

> "Here the presenter can change deductible, discount, and other quote details, then immediately see how likely the quote is to be accepted."

What to emphasize:

- This is a live what-if pricing workflow.
- It is a good example of how historical data can directly support a business decision.

### Retail extension overview

Start this extension with the notebook heading:

```python
mo.md("""
# Future extension 2: retail healthier alternatives
""")
```

Say:

> "The final section shows the same pattern in a recommendation setting: start from one product, find similar ones, and rank the healthier alternatives."

### Retail step 1: generate a product catalog

Code snippet:

```python
def make_retail_dataset(rows: int = 420, seed: int = 11) -> pd.DataFrame:
    ...
    retail_df["healthy_fit"] = rng.binomial(1, health_probability)
    retail_df["nutrition_score"] = (...).round(2)
```

Say:

> "This synthetic catalog gives each product structured attributes like flavor, texture, nutrition, and price, plus a label for whether it tends to be a strong healthier alternative."

What to emphasize:

- Again, the point is the interaction pattern.
- The data is structured rows and columns, just like the IBM framing.

### Retail step 2: fit the healthier-alternative model

Code snippet:

```python
(
    retail_classifier,
    retail_positive_index,
    ...,
) = fit_tabfm_binary(retail_X, retail_y, random_state=23)
```

Say:

> "This model learns which products look like strong healthier alternatives."

What to emphasize:

- The model score is only one part.
- Similarity is still needed so the alternatives stay relevant to the shopper's intent.

### Retail step 3: pick a product and rank alternatives

Code snippet:

```python
healthier_pool["overall_rank_score"] = (
    health_weight * healthier_pool["healthy_fit_probability"]
    + (1 - health_weight) * healthier_pool["similarity_score"]
)
recommended_products = healthier_pool.sort_values(
    "overall_rank_score", ascending=False
).head(6)
```

Say:

> "This is a nice example of recommendation with a business objective. We are not only finding similar products. We are combining similarity with a health-oriented ranking goal."

What to emphasize:

- Recommendation is not only nearest neighbor search.
- It can be steered toward healthier, safer, cheaper, or more profitable alternatives depending on the business goal.

---

## 10. The strongest business points to say out loud

If you want to sound strategic and convincing, keep returning to these points:

### 1. This is about operationalizing historical data

> "Most companies already have the data. The opportunity is to make that data usable at decision time."

### 2. This is not limited to one domain

> "Any repeated business decision with historical examples is a candidate: fraud, underwriting, pricing, quote acceptance, recommendations, anomaly review, contract review, and more."

### 3. This lowers the barrier for analytics teams

> "Pretrained tabular models mean teams do not always need to build everything from zero to start exploring this capability."

### 4. This can reduce classic data-science bottlenecks

> "One reason this is exciting is that it can reduce how much manual feature engineering and one-off model development is needed before a team can test a useful business workflow."

### 5. This is a bridge from analytics to intelligent applications

> "Instead of analytics ending in a dashboard, analytics can become an interactive system that helps someone act."

---

## 11. What not to overclaim

Be confident, but careful.

Do **not** claim:

- that this notebook is a production LDM platform,
- that TabFM is the same thing as IBM SQL DI,
- that foundation models remove the need for domain knowledge,
- that all feature engineering disappears automatically,
- that synthetic demos prove business ROI,
- or that the score should replace human judgment.

Better wording:

> "This is a credible prototype of the interaction pattern."

And:

> "The value of the demo is that it makes the concept concrete and shows how modern tabular models can support this style of workflow."

And, for the feature-engineering topic:

> "The right claim is not 'no data-science work is needed.' The right claim is 'the amount of custom work needed to reach a useful prototype can be much lower.'"

---

## 12. Suggested closing talk track

You can close with something like this:

> "What excites me about this space is that it gives analytics teams a path beyond static reporting. With historical structured data, interactive interfaces, and pretrained tabular models, we can start building systems that do not just describe the business, but help guide decisions inside it."
>
> "And if the model can absorb more of the pattern-recognition work that usually requires heavy feature engineering and custom model design, then smaller teams can participate in this shift much faster."
>
> "That is the real promise I see in the LDM idea: not only smarter models, but better decision workflows."

---

## 13. Short version for slide-design handoff

If a design agent or slide builder needs the distilled message, use this:

### Presentation theme
From dashboards to decision support

### Problem
Most analytics work explains the past but does not directly help a user make a live decision.

### LDM-style opportunity
Use historical structured data plus AI to:
- score a new case,
- retrieve similar prior cases,
- and test what-if changes in real time.

### Why TabFM
A pretrained foundation model for tabular data makes this concept faster to prototype, easier to demonstrate, and less dependent on heavy custom model-building.

### Feature-engineering message
The promise is not "no expertise needed."  
The promise is "less manual feature engineering and less bespoke model work before a team can build a useful prototype."

### Demo proof points
- Fraud triage
- Insurance quote optimization
- Retail healthier-alternative recommendation

### Strategic message
Smaller or less ML-mature markets can still begin building advanced decision-support experiences by combining domain data, interactive tooling, and pretrained tabular models.
