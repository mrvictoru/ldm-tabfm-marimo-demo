# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy==2.5.1",
#     "pandas==3.0.5",
#     "scikit-learn==1.9.0",
#     "tabfm==1.0.1",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="wide", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 1: Load the notebook tools

    This first step brings in the Python libraries used by the demo: data tools, machine-learning helpers, and the marimo interface that makes the controls work.
    """)
    return


@app.cell
def _():
    import functools

    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    return (
        ColumnTransformer,
        NearestNeighbors,
        OneHotEncoder,
        StandardScaler,
        TabFMClassifier,
        average_precision_score,
        functools,
        mo,
        np,
        pd,
        roc_auc_score,
        tabfm_v1_0_0,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # LDM-style fraud triage with TabFM

    This notebook recreates the main **large database model** behaviors described in the IBM article with a public Hugging Face fraud dataset and a pretrained **TabFM** foundation model:

    - **Risk scoring:** predict fraud probability for a transaction
    - **What-if analysis:** change transaction attributes and recompute the score
    - **Similar prior cases:** retrieve the closest historical transactions

    **Dataset:** `CiferAI/Cifer-Fraud-Detection-Dataset-AF` on Hugging Face
    **Model:** `google/tabfm-1.0.0-pytorch` classification checkpoint

    **Notes**
    - First run downloads public model weights from Hugging Face.
    - The pretrained TabFM weights are released under Google's **non-commercial** license.
    - The full Cifer dataset is very large, so this notebook intentionally samples a smaller interactive subset for molab/marimo use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How this maps to the IBM article

    | Article workflow | Status in this notebook | How it is reproduced |
    | --- | --- | --- |
    | Fraud / anomaly triage | **Implemented** | TabFM predicts fraud probability from transaction attributes |
    | What-if risk recomputation | **Implemented** | marimo widgets rebuild the transaction row and rescore it reactively |
    | Similar prior records | **Implemented** | nearest-neighbor retrieval over historical transactions |
    | Insurance quote optimization | **Implemented below** | synthetic quote data with quote acceptance scoring, what-if editing, and similar past quotes |
    | Retail healthier alternatives | **Implemented below** | synthetic product data with TabFM healthy-fit scoring, similarity, and alternative ranking |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How to read the outputs

    - **TabFM fraud probability**: a value between 0 and 1. Higher values mean the model thinks the transaction looks more like fraud based on the context rows it saw.
    - **Rule-based flag (`isFlaggedFraud`)**: a simple heuristic flag from the dataset. It is useful as a baseline, but it is not the same thing as the model's learned probability.
    - **Origin / destination balance error**: how far the edited balances are from the expected balance-change pattern. Large mismatches are often suspicious.
    - **Similar historical transactions**: the closest rows from the sampled dataset. These are useful for case comparison and investigation, not as a guarantee of fraud.

    In an LDM-style demo, the story is: “I can score a new case, inspect similar past cases, and see how a small change in the inputs changes the risk signal.”
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 2: Load a sample of transaction data

    This section downloads a manageable sample of fraud data from a public Hugging Face dataset. The notebook keeps the number of rows small so the demo stays fast and easy to use in molab.
    """)
    return


@app.cell
def _(functools, np, pd):
    DATA_URL = (
        "https://huggingface.co/datasets/"
        "CiferAI/Cifer-Fraud-Detection-Dataset-AF/resolve/main/"
        "Cifer-Fraud-Detection-Dataset-AF-part-1-14.csv"
    )
    RAW_COLUMNS = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]
    DISPLAY_COLUMNS = [
        "type",
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "isFraud",
        "origin_balance_error",
        "dest_balance_error",
    ]

    # Adds derived balance/hour features
    """Derive extra columns that help explain a transaction's risk.

    Computes the hour-of-day, the balance change on each side, and
    "balance error" terms that measure how far the recorded balances are
    from what a normal transaction of this amount would produce.
    Large balance errors are a common red flag in fraud analysis.
    """
    def _enrich_transactions(frame: pd.DataFrame) -> pd.DataFrame:

        enriched = frame.copy()
        enriched["hour_of_day"] = enriched["step"] % 24
        enriched["origin_delta"] = (
            enriched["oldbalanceOrg"] - enriched["newbalanceOrig"]
        )
        enriched["dest_delta"] = (
            enriched["newbalanceDest"] - enriched["oldbalanceDest"]
        )
        enriched["origin_balance_error"] = (
            enriched["origin_delta"] - enriched["amount"]
        ).abs()
        enriched["dest_balance_error"] = (
            enriched["dest_delta"] - enriched["amount"]
        ).abs()
        enriched["log_amount"] = np.log1p(enriched["amount"])
        return enriched

    # Streams a balanced sample from the Cifer CSV
    """Stream the remote Cifer CSV and return a balanced, interactive sample.

    Reads the file in chunks (to keep memory small), sampling fraud and
    non-fraud rows separately until the requested counts are reached, then
    shuffles and enriches the result. Cached so the heavy download only
    happens once per session.
    """
    @functools.lru_cache(maxsize=1)
    def load_cifer_sample(
        normal_rows: int = 2400,
        fraud_rows: int = 300,
        chunk_size: int = 150_000,
        random_state: int = 42,
    ) -> pd.DataFrame:

        positives = []
        negatives = []
        positive_count = 0
        negative_count = 0

        reader = pd.read_csv(
            DATA_URL,
            usecols=RAW_COLUMNS,
            chunksize=chunk_size,
        )
        for chunk in reader:
            fraud_chunk = chunk.loc[chunk["isFraud"] == 1]
            clean_chunk = chunk.loc[chunk["isFraud"] == 0]

            if positive_count < fraud_rows and not fraud_chunk.empty:
                needed = fraud_rows - positive_count
                sampled = fraud_chunk.sample(
                    n=min(needed, len(fraud_chunk)),
                    random_state=random_state,
                )
                positives.append(sampled)
                positive_count += len(sampled)

            if negative_count < normal_rows and not clean_chunk.empty:
                needed = normal_rows - negative_count
                sampled = clean_chunk.sample(
                    n=min(needed, len(clean_chunk)),
                    random_state=random_state,
                )
                negatives.append(sampled)
                negative_count += len(sampled)

            if positive_count >= fraud_rows and negative_count >= normal_rows:
                break

        sampled_frame = pd.concat([*positives, *negatives], ignore_index=True)
        sampled_frame = sampled_frame.sample(
            frac=1.0,
            random_state=random_state,
        ).reset_index(drop=True)
        return _enrich_transactions(sampled_frame)

    # Builds a what-if transaction row
    """Construct a one-row transaction from the what-if UI inputs.

    Recomputes the expected post-transaction balances from the edited
    amount. If "updates normally" is off for a side, it deliberately keeps
    the old balance, creating a mismatch that the model can flag as
    suspicious. Returns the enriched row ready for scoring.
    """
    def build_candidate_row(
        *,
        step: float,
        txn_type: str,
        amount: float,
        oldbalance_org: float,
        oldbalance_dest: float,
        origin_updates_normally: bool,
        dest_updates_normally: bool,
    ) -> pd.DataFrame:

        expected_origin = max(oldbalance_org - amount, 0.0)
        expected_dest = oldbalance_dest + amount
        newbalance_orig = expected_origin if origin_updates_normally else oldbalance_org
        newbalance_dest = expected_dest if dest_updates_normally else oldbalance_dest
        is_flagged = int(txn_type == "TRANSFER" and amount > 200_000)

        candidate = pd.DataFrame(
            [
                {
                    "step": int(step),
                    "type": txn_type,
                    "amount": float(amount),
                    "oldbalanceOrg": float(oldbalance_org),
                    "newbalanceOrig": float(newbalance_orig),
                    "oldbalanceDest": float(oldbalance_dest),
                    "newbalanceDest": float(newbalance_dest),
                    "isFlaggedFraud": is_flagged,
                    "isFraud": 0,
                }
            ]
        )
        return _enrich_transactions(candidate)

    return DATA_URL, DISPLAY_COLUMNS, build_candidate_row, load_cifer_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 3: Create the working dataset for the demo

    This step runs the data-loading function and gives the notebook a clean table to work with. You can think of this as the "historical cases" the model will compare against.
    """)
    return


@app.cell
def _(load_cifer_sample):
    modeling_df = load_cifer_sample()
    return (modeling_df,)


@app.cell
def _(modeling_df):
    print("Loaded dataframe columns:")
    print(modeling_df.columns.tolist())
    print("\nTop 5 rows:")
    print(modeling_df.head(5).to_string(index=False))
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 4: Review the sample statistics

    This section summarizes the sample so the audience can see what kind of data the model is using. It highlights how often fraud appears and how often the simple rule-based flag would catch a case.
    """)
    return


@app.cell
def _(DATA_URL, mo, modeling_df):
    fraud_rate = modeling_df["isFraud"].mean()
    flagged_rate = modeling_df["isFlaggedFraud"].mean()
    mo.md(
        f"""
        ## Public dataset slice used by the notebook

        - **Source shard:** `{DATA_URL.split('/')[-1]}`
        - **Rows sampled for interactivity:** `{len(modeling_df):,}`
        - **Fraud share in sample:** `{fraud_rate:.1%}`
        - **Rule-flagged share in sample:** `{flagged_rate:.1%}`

        The notebook drops the sender and recipient IDs and keeps the transaction attributes that a user can reasonably edit in a what-if workflow.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 5: Choose the features the model will use

    This cell decides which transaction details the model should look at. In plain terms, it says: “These are the fields we care about when judging whether a transaction looks suspicious.”
    """)
    return


@app.cell
def _(modeling_df):
    feature_columns = [
        "step",
        "hour_of_day",
        "type",
        "amount",
        "log_amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "origin_delta",
        "dest_delta",
        "origin_balance_error",
        "dest_balance_error",
    ]
    categorical_columns = ["type"]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    X = modeling_df[feature_columns].copy()
    y = modeling_df["isFraud"].astype(int).copy()
    return X, categorical_columns, feature_columns, numeric_columns, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 6: Train the fraud-risk model and test it on held-out data

    This is the key machine-learning step. The notebook builds a **stratified, class-ratio-aware** set of historical cases as context and asks the TabFM model to estimate the fraud probability for new examples.

    Key improvements over a naïve random sample:
    - **Larger context** (up to 150 per class instead of 50) gives the model more signal.
    - **Stratified by transaction type** so TRANSFER / CASH_OUT fraud patterns are represented.
    - **Class ratio matches real data** (~11% fraud) instead of an artificial 50/50 split that over-predicted fraud.
    """)
    return


@app.cell
def _(TabFMClassifier, X, pd, tabfm_v1_0_0, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    # Build a richer, stratified context that reflects real fraud patterns.
    # TabFM is an in-context learner: more diverse, representative context rows
    # give it a stronger signal.
    #
    # Strategy:
    #   - Use up to 150 rows per class (vs the previous cap of 50) so the model
    #     sees more variety.
    #   - Stratify by transaction type within each class so TRANSFER / CASH_OUT
    #     fraud patterns are represented, not just randomly sampled rows.
    #   - Weight fraud vs. normal rows to reflect the real ~11% fraud rate in the
    #     dataset instead of an artificial 50/50 split, which biased the model
    #     toward over-predicting fraud.

    fraud_rate_in_train = float(y_train.mean())
    context_fraud_rows = min(150, int(y_train.sum()))
    context_normal_rows = min(
        int(context_fraud_rows * (1.0 - fraud_rate_in_train) / fraud_rate_in_train),
        int((y_train == 0).sum()),
    )

    def _stratified_sample(mask, n, rng=42):
        """Sample n rows from X_train[mask], stratified by transaction type."""
        pool = X_train[mask].copy()
        pool["_y"] = y_train[mask]
        types = pool["type"].unique()
        per_type = max(1, n // len(types))
        parts = []
        for t in types:
            t_pool = pool[pool["type"] == t]
            take = min(per_type, len(t_pool))
            parts.append(t_pool.sample(n=take, random_state=rng))
        combined = pd.concat(parts)
        # top up / trim to exactly n
        if len(combined) < n:
            remaining = pool.drop(index=combined.index, errors="ignore")
            extra = min(n - len(combined), len(remaining))
            if extra > 0:
                combined = pd.concat(
                    [combined, remaining.sample(n=extra, random_state=rng)]
                )
        combined = combined.sample(n=min(n, len(combined)), random_state=rng)
        return combined.drop(columns=["_y"]).index.tolist()

    context_index = _stratified_sample(y_train == 1, context_fraud_rows)
    context_index += _stratified_sample(y_train == 0, context_normal_rows)

    X_context = X_train.loc[context_index].sample(frac=1.0, random_state=42)
    y_context = y_train.loc[X_context.index]

    tabfm_model = tabfm_v1_0_0.load(model_type="classification")
    classifier = TabFMClassifier(model=tabfm_model)
    classifier.fit(X_context, y_context.to_numpy())

    test_probabilities = classifier.predict_proba(X_test)
    class_order = list(classifier.classes_)
    positive_index = class_order.index(1)
    fraud_scores = pd.Series(
        test_probabilities[:, positive_index],
        index=X_test.index,
        name="tabfm_fraud_probability",
    )
    baseline_scores = X_test["isFlaggedFraud"].astype(float)
    return (
        X_context,
        baseline_scores,
        classifier,
        fraud_scores,
        positive_index,
        y_context,
        y_test,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 7: Review how well the model performed on a test slice

    This section shows a quick benchmark so the audience can see whether the model is producing a meaningful signal. It compares the TabFM score to the simple rule-based flag.
    """)
    return


@app.cell
def _(
    average_precision_score,
    baseline_scores,
    fraud_scores,
    mo,
    roc_auc_score,
    y_test,
):
    tabfm_auc = roc_auc_score(y_test, fraud_scores)
    tabfm_ap = average_precision_score(y_test, fraud_scores)
    baseline_auc = roc_auc_score(y_test, baseline_scores)
    baseline_ap = average_precision_score(y_test, baseline_scores)

    mo.md(
        f"""
        ## Quick sanity check on the sampled holdout set

        | Signal | ROC AUC | Average precision |
        | --- | ---: | ---: |
        | TabFM fraud probability | {tabfm_auc:.3f} | {tabfm_ap:.3f} |
        | Built-in rule flag (`isFlaggedFraud`) | {baseline_auc:.3f} | {baseline_ap:.3f} |

        This is not meant as a production benchmark; it only checks that the interactive demo has a meaningful signal on a sampled slice of the public dataset.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 8: Build the case-comparison layer

    This section creates a simple similarity tool. It helps the notebook find past transactions that look most like the new case being examined.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    NearestNeighbors,
    OneHotEncoder,
    StandardScaler,
    X,
    categorical_columns,
    feature_columns,
    mo,
    numeric_columns,
):
    similarity_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )
    similarity_matrix = similarity_preprocessor.fit_transform(X)
    similarity_index = NearestNeighbors(metric="euclidean", n_neighbors=8)
    similarity_index.fit(similarity_matrix)

    mo.md(
        f"""
        ## Similarity layer ready

        - **Reference set:** `{len(X):,}` historical transactions
        - **Features used:** `{len(feature_columns)}` (`{len(numeric_columns)}` numeric + `{len(categorical_columns)}` categorical)
        - **Neighbors retrieved per case:** `{similarity_index.n_neighbors}`

        Every scored case is compared against this index to surface the most similar historical transactions.
        """
    )
    return similarity_index, similarity_preprocessor


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 9: Create the interactive controls

    This section builds the sliders and dropdowns that let a non-technical audience change the transaction details in a simple way. These controls are the main way the demo becomes interactive.
    """)
    return


@app.cell
def _(X, mo, modeling_df):
    # Seed controls from a median-representative non-fraud row rather than
    # the arbitrary first row, so the default what-if case is more typical.
    normal_rows = modeling_df.loc[modeling_df["isFraud"] == 0]
    numeric_seed_cols = ["amount", "oldbalanceOrg", "oldbalanceDest"]
    medians = normal_rows[numeric_seed_cols].median()
    diffs = (normal_rows[numeric_seed_cols] - medians).abs().sum(axis=1)
    safe_seed = normal_rows.loc[diffs.idxmin()]
    amount_cap = float(X["amount"].quantile(0.995))
    origin_cap = float(X["oldbalanceOrg"].quantile(0.995))
    dest_cap = float(X["oldbalanceDest"].quantile(0.995))

    transaction_type = mo.ui.dropdown(
        options=sorted(modeling_df["type"].unique().tolist()),
        value=str(safe_seed["type"]),
        label="Transaction type",
    )
    step_value = mo.ui.slider(
        start=1,
        stop=744,
        step=1,
        value=int(safe_seed["step"]),
        label="Hour in simulation window",
        show_value=True,
        include_input=True,
        debounce=True,
    )
    amount_value = mo.ui.number(
        start=0,
        stop=max(250_000.0, amount_cap),
        step=100.0,
        value=float(safe_seed["amount"]),
        label="Transaction amount",
        debounce=True,
    )
    oldbalance_org_value = mo.ui.number(
        start=0,
        stop=max(250_000.0, origin_cap),
        step=100.0,
        value=float(safe_seed["oldbalanceOrg"]),
        label="Origin balance before transaction",
        debounce=True,
    )
    oldbalance_dest_value = mo.ui.number(
        start=0,
        stop=max(250_000.0, dest_cap),
        step=100.0,
        value=float(safe_seed["oldbalanceDest"]),
        label="Destination balance before transaction",
        debounce=True,
    )
    origin_updates_normally = mo.ui.switch(
        value=True,
        label="Origin balance updates normally",
    )
    dest_updates_normally = mo.ui.switch(
        value=True,
        label="Destination balance updates normally",
    )
    return (
        amount_value,
        dest_updates_normally,
        oldbalance_dest_value,
        oldbalance_org_value,
        origin_updates_normally,
        step_value,
        transaction_type,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 10: Arrange the controls in one place

    This step places the controls into a simple panel so the audience can focus on the inputs without getting lost in the notebook layout.
    """)
    return


@app.cell
def _(
    amount_value,
    dest_updates_normally,
    mo,
    oldbalance_dest_value,
    oldbalance_org_value,
    origin_updates_normally,
    step_value,
    transaction_type,
):
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
    controls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 11: Score the edited transaction

    This is the moment where the demo becomes a decision-support tool. The notebook takes the values from the controls, builds a new transaction row, and asks the model for a fraud probability.
    """)
    return


@app.cell
def _(
    amount_value,
    build_candidate_row,
    classifier,
    dest_updates_normally,
    feature_columns,
    modeling_df,
    oldbalance_dest_value,
    oldbalance_org_value,
    origin_updates_normally,
    pd,
    positive_index,
    similarity_index,
    similarity_preprocessor,
    step_value,
    transaction_type,
):
    candidate_frame = build_candidate_row(
        step=step_value.value,
        txn_type=transaction_type.value,
        amount=amount_value.value,
        oldbalance_org=oldbalance_org_value.value,
        oldbalance_dest=oldbalance_dest_value.value,
        origin_updates_normally=origin_updates_normally.value,
        dest_updates_normally=dest_updates_normally.value,
    )
    candidate_features = candidate_frame[feature_columns]
    candidate_probability = float(
        classifier.predict_proba(candidate_features)[0][positive_index]
    )
    candidate_similarity = similarity_preprocessor.transform(candidate_features)
    _, neighbor_positions = similarity_index.kneighbors(candidate_similarity)
    similar_cases = modeling_df.iloc[neighbor_positions[0]].copy()
    similar_cases["tabfm_demo_distance_rank"] = range(1, len(similar_cases) + 1)
    similar_cases = similar_cases.reset_index(drop=True)

    suspicion_checks = pd.DataFrame(
        [
            {
                "signal": "Large transfer rule",
                "value": bool(candidate_frame.loc[0, "isFlaggedFraud"]),
            },
            {
                "signal": "Origin balance mismatch",
                "value": candidate_frame.loc[0, "origin_balance_error"] > 1.0,
            },
            {
                "signal": "Destination balance mismatch",
                "value": candidate_frame.loc[0, "dest_balance_error"] > 1.0,
            },
        ]
    )
    return (
        candidate_frame,
        candidate_probability,
        similar_cases,
        suspicion_checks,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 12: Show the current risk result

    This section turns the model score into a simple result summary. It shows the probability, the rule-based flag, and a few balance-based signals that help explain the score.
    """)
    return


@app.cell
def _(candidate_frame, candidate_probability, mo, suspicion_checks):
    current_flag = int(candidate_frame.loc[0, "isFlaggedFraud"])
    result_box = mo.md(
        f"""
        ## Current triage result

        - **TabFM fraud probability:** `{candidate_probability:.1%}`
        - **Rule-based flag (`isFlaggedFraud`):** `{current_flag}`
        - **Origin balance error:** `{candidate_frame.loc[0, "origin_balance_error"]:.2f}`
        - **Destination balance error:** `{candidate_frame.loc[0, "dest_balance_error"]:.2f}`
        """
    ).callout(
        kind="danger" if candidate_probability >= 0.5 else "info",
    )

    mo.vstack([result_box, suspicion_checks])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 13: Show similar past cases

    This section presents the closest historical transactions. It is the case-comparison part of the demo: instead of looking at a single score alone, the user can inspect similar examples.
    """)
    return


@app.cell
def _(DISPLAY_COLUMNS, mo, similar_cases):
    mo.vstack(
        [
            mo.md("## Similar historical transactions"),
            similar_cases[["tabfm_demo_distance_rank", *DISPLAY_COLUMNS]],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 14: Show the context rows used by the model

    This last section makes the model's reasoning more transparent by showing the historical examples the model saw as context. These are the examples that helped the model form its judgment.
    """)
    return


@app.cell
def _(X_context, mo, modeling_df, y_context):
    context_preview = modeling_df.loc[X_context.index].copy()
    context_preview["context_label"] = y_context.values
    mo.vstack(
        [
            mo.md("## Context rows passed to TabFM"),
            context_preview[
                [
                    "type",
                    "amount",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "isFlaggedFraud",
                    "context_label",
                ]
            ].head(12),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Future extension 1: insurance quote optimization

    This extension mirrors the IBM article's insurance example. A user can edit an insurance quote, see the predicted chance of acceptance, and compare the quote to similar past cases.
    """)
    return


app._unparsable_cell(
    """
    # In context learning for TabFM + returns eval pieces
    def fit_tabfm_binary(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        random_state: int = 42,
        context_cap: int = 50,
    ):
    \"\"\"Train a TabFM classifier and return everything needed for scoring.

    Splits the data into train/test, samples a balanced set of per-class
    \"context\" rows (the few examples TabFM conditions on), fits the
    classifier, and scores the holdout set. Returns the fitted model,
    the positive-class index, the context rows/labels, and test scores
    so callers can report AUC / average precision.
    \"\"\"
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=random_state,
        )

        context_per_class = min(context_cap, y_train.value_counts().min())
        context_index = []
        for class_value in sorted(y_train.unique()):
            sampled_index = y_train[y_train == class_value].sample(
                n=context_per_class,
                random_state=random_state,
            ).index
            context_index.extend(sampled_index.tolist())

        X_context = X_train.loc[context_index].sample(frac=1.0, random_state=random_state)
        y_context = y_train.loc[X_context.index]

        classifier = TabFMClassifier(
            model=tabfm_v1_0_0.load(model_type=\"classification\")
        )
        classifier.fit(X_context, y_context.to_numpy())
        positive_index = list(classifier.classes_).index(1)
        test_probabilities = classifier.predict_proba(X_test)[:, positive_index]
        test_scores = pd.Series(
            test_probabilities,
            index=X_test.index,
            name=\"tabfm_probability\",
        )
        return classifier, positive_index, X_context, y_context, X_test, y_test, test_scores

    # Builds preprocessor + NN index
    def build_similarity_tools(
        frame: pd.DataFrame,
        categorical_columns: list[str],
        numeric_columns: list[str],
        *,
        n_neighbors: int = 8,
    ):
    \"\"\"Build the preprocessing pipeline and nearest-neighbor index.

    Scales numeric columns and one-hot encodes categorical columns into a
    single feature matrix, then fits a NearestNeighbors index over it.
    Used to retrieve the closest historical rows for any new case.
    \"\"\"
        preprocessor = ColumnTransformer(
            transformers=[
                (\"numeric\", StandardScaler(), numeric_columns),
                (
                    \"categorical\",
                    OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False),
                    categorical_columns,
                ),
            ]
        )
        matrix = preprocessor.fit_transform(frame)
        index = NearestNeighbors(metric=\"euclidean\", n_neighbors=n_neighbors)
        index.fit(matrix)
        return preprocessor, matrix, index
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Insurance step 1: generate historical quote data

    This synthetic dataset acts like a past book of insurance quotes. It includes customer, vehicle, and price information plus a label showing whether a quote was accepted.
    """)
    return


app._unparsable_cell(
    """
    def make_insurance_dataset(rows: int = 1200, seed: int = 7) -> pd.DataFrame:
    \"\"\"Generate a synthetic book of historical insurance quotes.

    Creates customer, vehicle, and coverage attributes, derives premium and
    price-per-mile fields with a fixed pricing formula, then simulates an
    accepted/rejected label from a logistic model. Seeded so the demo data
    is reproducible.
    \"\"\"
        rng = np.random.default_rng(seed)
        coverage_tiers = np.array([\"Basic\", \"Standard\", \"Premium\"])
        vehicle_types = np.array([\"Sedan\", \"SUV\", \"Truck\", \"EV\"])
        regions = np.array([\"Urban\", \"Suburban\", \"Rural\"])

        quote_df = pd.DataFrame(
            {
                \"driver_age\": rng.integers(21, 76, rows),
                \"vehicle_age\": rng.integers(0, 16, rows),
                \"annual_mileage\": rng.integers(6000, 26000, rows),
                \"accident_count\": rng.choice([0, 1, 2, 3], size=rows, p=[0.62, 0.23, 0.11, 0.04]),
                \"loyalty_years\": rng.integers(0, 16, rows),
                \"bundle_home\": rng.integers(0, 2, rows),
                \"coverage_tier\": rng.choice(coverage_tiers, size=rows, p=[0.3, 0.45, 0.25]),
                \"vehicle_type\": rng.choice(vehicle_types, size=rows, p=[0.4, 0.3, 0.15, 0.15]),
                \"region\": rng.choice(regions, size=rows, p=[0.38, 0.42, 0.2]),
            }
        )

        tier_factor = quote_df[\"coverage_tier\"].map(
            {\"Basic\": 0.9, \"Standard\": 1.15, \"Premium\": 1.45}
        )
        vehicle_factor = quote_df[\"vehicle_type\"].map(
            {\"Sedan\": 1.0, \"SUV\": 1.12, \"Truck\": 1.18, \"EV\": 1.08}
        )
        region_factor = quote_df[\"region\"].map(
            {\"Urban\": 1.12, \"Suburban\": 1.0, \"Rural\": 0.93}
        )

        quote_df[\"base_premium\"] = (
            520
            + quote_df[\"vehicle_age\"] * 18
            + quote_df[\"annual_mileage\"] * 0.014
            + quote_df[\"accident_count\"] * 185
            + (75 - quote_df[\"driver_age\"]).clip(lower=0) * 4
        ) * tier_factor * vehicle_factor * region_factor
        quote_df[\"base_premium\"] = quote_df[\"base_premium\"].round(0)

        quote_df[\"deductible\"] = rng.choice(
            [250, 500, 750, 1000, 1500],
            size=rows,
            p=[0.12, 0.35, 0.18, 0.25, 0.10],
        )
        quote_df[\"discount_pct\"] = rng.integers(0, 21, rows)
        quote_df[\"premium_after_discount\"] = (
            quote_df[\"base_premium\"] * (1 - quote_df[\"discount_pct\"] / 100)
        ).round(0)
        quote_df[\"price_per_mile\"] = (
            quote_df[\"premium_after_discount\"] / quote_df[\"annual_mileage\"]
        ).round(4)

        logit = (
            2.4
            - 0.0019 * quote_df[\"premium_after_discount\"]
            + 0.0005 * quote_df[\"deductible\"]
            + 0.05 * quote_df[\"discount_pct\"]
            + 0.10 * quote_df[\"loyalty_years\"]
            + 0.45 * quote_df[\"bundle_home\"]
            - 0.55 * quote_df[\"accident_count\"]
            - 0.10 * (quote_df[\"coverage_tier\"] == \"Premium\").astype(float)
            + 0.14 * (quote_df[\"region\"] == \"Rural\").astype(float)
            + rng.normal(0, 0.55, rows)
        )
        probability = 1 / (1 + np.exp(-logit))
        quote_df[\"accepted\"] = rng.binomial(1, probability)
        return quote_df

    def build_insurance_quote(
        *,
        driver_age: float,
        vehicle_age: float,
        annual_mileage: float,
        accident_count: float,
        loyalty_years: float,
        bundle_home: bool,
        coverage_tier: str,
        vehicle_type: str,
        region: str,
        deductible: float,
        discount_pct: float,
    ) -> pd.DataFrame:
    \"\"\"Build a one-row insurance quote from the what-if UI inputs.

    Applies the same pricing formula used by make_insurance_dataset so the
    edited quote is consistent with the historical book of quotes, and can
    be scored by the trained acceptance model.
    \"\"\"
        tier_factor = {\"Basic\": 0.9, \"Standard\": 1.15, \"Premium\": 1.45}[coverage_tier]
        vehicle_factor = {\"Sedan\": 1.0, \"SUV\": 1.12, \"Truck\": 1.18, \"EV\": 1.08}[vehicle_type]
        region_factor = {\"Urban\": 1.12, \"Suburban\": 1.0, \"Rural\": 0.93}[region]
        base_premium = (
            520
            + vehicle_age * 18
            + annual_mileage * 0.014
            + accident_count * 185
            + max(75 - driver_age, 0) * 4
        ) * tier_factor * vehicle_factor * region_factor
        premium_after_discount = base_premium * (1 - discount_pct / 100)
        return pd.DataFrame(
            [
                {
                    \"driver_age\": float(driver_age),
                    \"vehicle_age\": float(vehicle_age),
                    \"annual_mileage\": float(annual_mileage),
                    \"accident_count\": float(accident_count),
                    \"loyalty_years\": float(loyalty_years),
                    \"bundle_home\": int(bundle_home),
                    \"coverage_tier\": coverage_tier,
                    \"vehicle_type\": vehicle_type,
                    \"region\": region,
                    \"base_premium\": round(base_premium, 0),
                    \"deductible\": float(deductible),
                    \"discount_pct\": float(discount_pct),
                    \"premium_after_discount\": round(premium_after_discount, 0),
                    \"price_per_mile\": round(premium_after_discount / annual_mileage, 4),
                }
            ]
        )
    """,
    name="_"
)


@app.cell
def _(make_insurance_dataset):
    insurance_df = make_insurance_dataset()
    return (insurance_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Insurance step 2: fit a quote-acceptance model

    The quote model learns the pattern behind accepted versus rejected quotes. In a live demo, this is what lets you show how small quote changes affect likely customer response.
    """)
    return


@app.cell
def _(build_similarity_tools, fit_tabfm_binary, insurance_df):
    insurance_feature_columns = [
        "driver_age",
        "vehicle_age",
        "annual_mileage",
        "accident_count",
        "loyalty_years",
        "bundle_home",
        "coverage_tier",
        "vehicle_type",
        "region",
        "base_premium",
        "deductible",
        "discount_pct",
        "premium_after_discount",
        "price_per_mile",
    ]
    insurance_categorical_columns = ["coverage_tier", "vehicle_type", "region"]
    insurance_numeric_columns = [
        column
        for column in insurance_feature_columns
        if column not in insurance_categorical_columns
    ]
    insurance_X = insurance_df[insurance_feature_columns].copy()
    insurance_y = insurance_df["accepted"].astype(int).copy()
    (
        insurance_classifier,
        insurance_positive_index,
        insurance_context,
        insurance_context_labels,
        insurance_X_test,
        insurance_y_test,
        insurance_test_scores,
    ) = fit_tabfm_binary(insurance_X, insurance_y, random_state=19)
    (
        insurance_similarity_preprocessor,
        _insurance_similarity_matrix,
        insurance_similarity_index,
    ) = build_similarity_tools(
        insurance_X,
        insurance_categorical_columns,
        insurance_numeric_columns,
    )
    return (
        insurance_classifier,
        insurance_feature_columns,
        insurance_positive_index,
        insurance_similarity_index,
        insurance_similarity_preprocessor,
        insurance_test_scores,
        insurance_y_test,
    )


@app.cell
def _(
    average_precision_score,
    insurance_test_scores,
    insurance_y_test,
    mo,
    roc_auc_score,
):
    mo.md(f"""
    ## Insurance extension: model check

    - **Holdout ROC AUC:** `{roc_auc_score(insurance_y_test, insurance_test_scores):.3f}`
    - **Holdout average precision:** `{average_precision_score(insurance_y_test, insurance_test_scores):.3f}`

    This extension uses the same LDM-style pattern as the fraud demo: score a new case, adjust inputs, and compare with similar past records.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Insurance step 3: edit a quote and rescore it

    These controls let a presenter show the most intuitive part of the insurance use case: if price, deductible, or discount changes, the predicted chance of acceptance changes too.
    """)
    return


@app.cell
def _(insurance_df, mo):
    insurance_seed = insurance_df.iloc[0]
    insurance_coverage = mo.ui.dropdown(
        options=sorted(insurance_df["coverage_tier"].unique().tolist()),
        value=str(insurance_seed["coverage_tier"]),
        label="Coverage tier",
    )
    insurance_vehicle = mo.ui.dropdown(
        options=sorted(insurance_df["vehicle_type"].unique().tolist()),
        value=str(insurance_seed["vehicle_type"]),
        label="Vehicle type",
    )
    insurance_region = mo.ui.dropdown(
        options=sorted(insurance_df["region"].unique().tolist()),
        value=str(insurance_seed["region"]),
        label="Region",
    )
    insurance_driver_age = mo.ui.slider(
        start=21,
        stop=75,
        step=1,
        value=int(insurance_seed["driver_age"]),
        label="Driver age",
        show_value=True,
    )
    insurance_vehicle_age = mo.ui.slider(
        start=0,
        stop=15,
        step=1,
        value=int(insurance_seed["vehicle_age"]),
        label="Vehicle age",
        show_value=True,
    )
    insurance_mileage = mo.ui.number(
        start=6000,
        stop=30000,
        step=500,
        value=float(insurance_seed["annual_mileage"]),
        label="Annual mileage",
    )
    insurance_accidents = mo.ui.slider(
        start=0,
        stop=3,
        step=1,
        value=int(insurance_seed["accident_count"]),
        label="Prior accidents",
        show_value=True,
    )
    insurance_loyalty = mo.ui.slider(
        start=0,
        stop=15,
        step=1,
        value=int(insurance_seed["loyalty_years"]),
        label="Years with insurer",
        show_value=True,
    )
    insurance_bundle = mo.ui.switch(
        value=bool(insurance_seed["bundle_home"]),
        label="Home policy bundled",
    )
    insurance_deductible = mo.ui.slider(
        start=250,
        stop=1500,
        step=250,
        value=int(insurance_seed["deductible"]),
        label="Deductible",
        show_value=True,
    )
    insurance_discount = mo.ui.slider(
        start=0,
        stop=20,
        step=1,
        value=int(insurance_seed["discount_pct"]),
        label="Discount percent",
        show_value=True,
    )
    insurance_controls = mo.vstack(
        [
            mo.md("## Insurance quote controls"),
            insurance_coverage,
            insurance_vehicle,
            insurance_region,
            insurance_driver_age,
            insurance_vehicle_age,
            insurance_mileage,
            insurance_accidents,
            insurance_loyalty,
            insurance_bundle,
            insurance_deductible,
            insurance_discount,
        ]
    )
    insurance_controls
    return (
        insurance_accidents,
        insurance_bundle,
        insurance_coverage,
        insurance_deductible,
        insurance_discount,
        insurance_driver_age,
        insurance_loyalty,
        insurance_mileage,
        insurance_region,
        insurance_vehicle,
        insurance_vehicle_age,
    )


@app.cell
def _(
    build_insurance_quote,
    insurance_accidents,
    insurance_bundle,
    insurance_classifier,
    insurance_coverage,
    insurance_deductible,
    insurance_df,
    insurance_discount,
    insurance_driver_age,
    insurance_feature_columns,
    insurance_loyalty,
    insurance_mileage,
    insurance_positive_index,
    insurance_region,
    insurance_similarity_index,
    insurance_similarity_preprocessor,
    insurance_vehicle,
    insurance_vehicle_age,
):
    insurance_candidate = build_insurance_quote(
        driver_age=insurance_driver_age.value,
        vehicle_age=insurance_vehicle_age.value,
        annual_mileage=insurance_mileage.value,
        accident_count=insurance_accidents.value,
        loyalty_years=insurance_loyalty.value,
        bundle_home=insurance_bundle.value,
        coverage_tier=insurance_coverage.value,
        vehicle_type=insurance_vehicle.value,
        region=insurance_region.value,
        deductible=insurance_deductible.value,
        discount_pct=insurance_discount.value,
    )
    insurance_candidate_features = insurance_candidate[insurance_feature_columns]
    insurance_acceptance_probability = float(
        insurance_classifier.predict_proba(insurance_candidate_features)[0][
            insurance_positive_index
        ]
    )
    insurance_similarity = insurance_similarity_preprocessor.transform(
        insurance_candidate_features
    )
    _, insurance_neighbor_positions = insurance_similarity_index.kneighbors(
        insurance_similarity
    )
    insurance_similar_quotes = insurance_df.iloc[insurance_neighbor_positions[0]].copy()
    insurance_similar_quotes["similarity_rank"] = range(
        1, len(insurance_similar_quotes) + 1
    )
    return (
        insurance_acceptance_probability,
        insurance_candidate,
        insurance_similar_quotes,
    )


@app.cell
def _(insurance_acceptance_probability, insurance_candidate, mo):
    mo.md(
        f"""
        ## Insurance extension: current quote result

        - **Predicted quote acceptance probability:** `{insurance_acceptance_probability:.1%}`
        - **Base premium:** `${float(insurance_candidate.loc[0, "base_premium"]):,.0f}`
        - **Premium after discount:** `${float(insurance_candidate.loc[0, "premium_after_discount"]):,.0f}`
        - **Price per mile:** `${float(insurance_candidate.loc[0, "price_per_mile"]):.4f}`

        This is the IBM-style insurance workflow: edit the quote, recompute the odds, and compare the result with similar past quotes.
        """
    ).callout(
        kind="success" if insurance_acceptance_probability >= 0.5 else "warn",
    )
    return


@app.cell
def _(insurance_similar_quotes, mo):
    mo.vstack(
        [
            mo.md("## Similar historical insurance quotes"),
            insurance_similar_quotes[
                [
                    "similarity_rank",
                    "coverage_tier",
                    "vehicle_type",
                    "region",
                    "driver_age",
                    "vehicle_age",
                    "premium_after_discount",
                    "deductible",
                    "discount_pct",
                    "accepted",
                ]
            ],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Future extension 2: retail healthier alternatives

    This extension mirrors the article's retail example. A user starts from one product and the notebook suggests similar alternatives that look healthier according to nutrition and model score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Retail step 1: generate a product catalog

    This dataset acts like a structured product table. Each row describes a product's taste-style, nutrition profile, price, and a synthetic label saying whether it tends to be a strong healthier alternative.
    """)
    return


app._unparsable_cell(
    """
    def make_retail_dataset(rows: int = 420, seed: int = 11) -> pd.DataFrame:
    \"\"\"Generate a synthetic product catalog for the healthier-alternatives demo.

    Creates category, flavor, texture, brand, nutrition, and price columns,
    then simulates a healthy-fit label from a logistic model plus a numeric
    nutrition score. Seeded so the catalog is reproducible.
    \"\"\"
        rng = np.random.default_rng(seed)
        categories = np.array([\"Cereal\", \"Granola\", \"Snack Bar\", \"Crackers\"])
        flavor_profiles = np.array([\"Nutty\", \"Fruity\", \"Chocolate\", \"Plain\"])
        textures = np.array([\"Crunchy\", \"Soft\", \"Light\"])
        brands = np.array([\"Northfield\", \"Harvest\", \"Bright\", \"Peak\"])

        retail_df = pd.DataFrame(
            {
                \"category\": rng.choice(categories, size=rows, p=[0.35, 0.2, 0.25, 0.2]),
                \"flavor_profile\": rng.choice(flavor_profiles, size=rows),
                \"texture\": rng.choice(textures, size=rows),
                \"brand_family\": rng.choice(brands, size=rows),
                \"sugar_g\": rng.integers(2, 22, rows),
                \"fiber_g\": rng.integers(1, 12, rows),
                \"protein_g\": rng.integers(2, 16, rows),
                \"calories\": rng.integers(80, 240, rows),
                \"price_usd\": rng.uniform(2.5, 8.5, rows).round(2),
                \"sodium_mg\": rng.integers(40, 360, rows),
                \"whole_grain\": rng.integers(0, 2, rows),
            }
        )
        retail_df[\"product_name\"] = [
            f\"{retail_df.loc[i, 'brand_family']} {retail_df.loc[i, 'category']} {i + 1}\"
            for i in retail_df.index
        ]
        health_logit = (
            0.55 * retail_df[\"fiber_g\"]
            + 0.38 * retail_df[\"protein_g\"]
            - 0.35 * retail_df[\"sugar_g\"]
            - 0.012 * retail_df[\"calories\"]
            - 0.18 * retail_df[\"price_usd\"]
            - 0.002 * retail_df[\"sodium_mg\"]
            + 0.75 * retail_df[\"whole_grain\"]
            + 0.20 * (retail_df[\"category\"] == \"Granola\").astype(float)
            + rng.normal(0, 1.1, rows)
        )
        health_probability = 1 / (1 + np.exp(-(health_logit - 2.0)))
        retail_df[\"healthy_fit\"] = rng.binomial(1, health_probability)
        retail_df[\"nutrition_score\"] = (
            retail_df[\"fiber_g\"] * 2.2
            + retail_df[\"protein_g\"] * 1.6
            + retail_df[\"whole_grain\"] * 4.0
            - retail_df[\"sugar_g\"] * 1.5
            - retail_df[\"calories\"] * 0.03
            - retail_df[\"sodium_mg\"] * 0.01
        ).round(2)
        return retail_df
    """,
    name="_"
)


@app.cell
def _(make_retail_dataset):
    retail_df = make_retail_dataset()
    return (retail_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Retail step 2: fit the healthier-alternative model

    The model here learns which product rows look like strong healthier alternatives. Later, the notebook will combine that score with product similarity to produce recommendation-style suggestions.
    """)
    return


@app.cell
def _(build_similarity_tools, fit_tabfm_binary, retail_df):
    retail_feature_columns = [
        "category",
        "flavor_profile",
        "texture",
        "brand_family",
        "sugar_g",
        "fiber_g",
        "protein_g",
        "calories",
        "price_usd",
        "sodium_mg",
        "whole_grain",
        "nutrition_score",
    ]
    retail_categorical_columns = ["category", "flavor_profile", "texture", "brand_family"]
    retail_numeric_columns = [
        column for column in retail_feature_columns if column not in retail_categorical_columns
    ]
    retail_X = retail_df[retail_feature_columns].copy()
    retail_y = retail_df["healthy_fit"].astype(int).copy()
    (
        retail_classifier,
        retail_positive_index,
        retail_context,
        retail_context_labels,
        retail_X_test,
        retail_y_test,
        retail_test_scores,
    ) = fit_tabfm_binary(retail_X, retail_y, random_state=23)
    retail_similarity_preprocessor, retail_similarity_matrix, _retail_similarity_index = (
        build_similarity_tools(
            retail_X,
            retail_categorical_columns,
            retail_numeric_columns,
            n_neighbors=20,
        )
    )
    return (
        retail_classifier,
        retail_feature_columns,
        retail_positive_index,
        retail_similarity_preprocessor,
        retail_test_scores,
        retail_y_test,
    )


@app.cell
def _(
    average_precision_score,
    mo,
    retail_test_scores,
    retail_y_test,
    roc_auc_score,
):
    mo.md(f"""
    ## Retail extension: model check

    - **Holdout ROC AUC:** `{roc_auc_score(retail_y_test, retail_test_scores):.3f}`
    - **Holdout average precision:** `{average_precision_score(retail_y_test, retail_test_scores):.3f}`

    This extension uses the same structure as the article's shopping example: start from one item, find similar items, and rank the healthier alternatives.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Retail step 3: pick a product and rank alternatives

    The presenter chooses one starting product, then adjusts how strongly the ranking should favor “healthier” versus “most similar.” This makes the recommendation story easy to explain to non-technical users.
    """)
    return


@app.cell
def _(mo, retail_df):
    retail_product_options = {
        f"{row.product_name} ({row.category})": int(index)
        for index, row in retail_df.head(250).iterrows()
    }
    retail_product = mo.ui.dropdown(
        options=retail_product_options,
        value=next(iter(retail_product_options)),
        label="Starting product",
        searchable=True,
    )
    retail_health_priority = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=65,
        label="How strongly to favor healthier alternatives",
        show_value=True,
    )
    retail_controls = mo.vstack(
        [
            mo.md("## Retail alternative finder controls"),
            retail_product,
            retail_health_priority,
        ]
    )
    retail_controls
    return retail_health_priority, retail_product


@app.cell
def _(
    np,
    pd,
    retail_classifier,
    retail_df,
    retail_feature_columns,
    retail_health_priority,
    retail_positive_index,
    retail_product,
    retail_similarity_preprocessor,
):
    selected_product = retail_df.iloc[int(retail_product.value)].copy()
    selected_feature_row = retail_df.loc[[int(retail_product.value)], retail_feature_columns]
    selected_vector = retail_similarity_preprocessor.transform(selected_feature_row)[0]

    candidate_pool = retail_df.loc[
        (retail_df["category"] == selected_product["category"])
        & (retail_df.index != int(retail_product.value))
    ].copy()
    candidate_vectors = retail_similarity_preprocessor.transform(
        candidate_pool[retail_feature_columns]
    )
    candidate_pool["distance"] = np.linalg.norm(candidate_vectors - selected_vector, axis=1)
    candidate_pool["is_healthier"] = (
        (candidate_pool["sugar_g"] <= selected_product["sugar_g"])
        & (candidate_pool["calories"] <= selected_product["calories"])
        & (candidate_pool["fiber_g"] >= selected_product["fiber_g"])
    )

    healthier_pool = candidate_pool.loc[candidate_pool["is_healthier"]].copy()
    if healthier_pool.empty:
        healthier_pool = candidate_pool.nsmallest(8, "distance").copy()

    healthier_features = healthier_pool[retail_feature_columns]
    healthier_pool["healthy_fit_probability"] = retail_classifier.predict_proba(
        healthier_features
    )[:, retail_positive_index]

    max_distance = max(float(healthier_pool["distance"].max()), 1e-6)
    healthier_pool["similarity_score"] = 1 - healthier_pool["distance"] / max_distance
    health_weight = retail_health_priority.value / 100
    healthier_pool["overall_rank_score"] = (
        health_weight * healthier_pool["healthy_fit_probability"]
        + (1 - health_weight) * healthier_pool["similarity_score"]
    )
    recommended_products = healthier_pool.sort_values(
        "overall_rank_score", ascending=False
    ).head(6)
    selected_product_df = pd.DataFrame([selected_product])
    return recommended_products, selected_product_df


@app.cell
def _(mo, selected_product_df):
    selected_row = selected_product_df.iloc[0]
    mo.md(
        f"""
        ## Retail extension: selected product

        - **Product:** `{selected_row["product_name"]}`
        - **Category:** `{selected_row["category"]}`
        - **Sugar:** `{selected_row["sugar_g"]}g`
        - **Fiber:** `{selected_row["fiber_g"]}g`
        - **Calories:** `{selected_row["calories"]}`
        - **Nutrition score:** `{selected_row["nutrition_score"]}`

        The alternatives below stay close to this product while trying to improve the nutrition profile.
        """
    )
    return


@app.cell
def _(mo, recommended_products):
    mo.vstack(
        [
            mo.md("## Retail extension: healthier similar alternatives"),
            recommended_products[
                [
                    "product_name",
                    "category",
                    "flavor_profile",
                    "texture",
                    "sugar_g",
                    "fiber_g",
                    "protein_g",
                    "calories",
                    "price_usd",
                    "healthy_fit_probability",
                    "overall_rank_score",
                ]
            ],
        ]
    )
    return


if __name__ == "__main__":
    app.run()
