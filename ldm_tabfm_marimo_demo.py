# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.13",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "scikit-learn>=1.5",
#     "tabfm[pytorch]>=1.0.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    return (
        ColumnTransformer,
        DummyClassifier,
        LogisticRegression,
        NearestNeighbors,
        OneHotEncoder,
        StandardScaler,
        TabFMClassifier,
        average_precision_score,
        make_pipeline,
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
    # Recreating IBM's LDM quote workflow with Google TabFM

    IBM describes an insurance workflow in which a salesperson:

    1. enters a candidate quote,
    2. estimates the probability that the customer will accept it,
    3. retrieves similar historical quotes,
    4. changes terms such as the deductible or discount,
    5. and recalculates the odds before choosing an offer.

    This notebook recreates that **interaction pattern** with Google's pretrained
    **TabFM** tabular foundation model and a marimo interface.

    > **Important boundary:** This is an independent technical prototype. It does
    > not reproduce IBM SQL Data Insights, and it does not establish that IBM's
    > product uses TabFM or the same architecture.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What this demo is designed to prove

    The useful claim is not merely that TabFM can output a probability. The demo
    must show all of the following:

    - **Held-out signal:** TabFM ranks accepted quotes above rejected quotes.
    - **A no-skill comparison:** its predictions beat a class-prior baseline.
    - **A conventional comparison:** an untuned logistic model is shown beside it.
    - **Decision-time behavior:** changing quote terms recalculates the score.
    - **Historical grounding:** similar prior quotes are displayed with outcomes.

    TabFM is zero-shot in the model-training sense: `fit()` prepares the table and
    supplies labeled rows as in-context examples; it does not update the pretrained
    model weights. We still need representative context rows and held-out evaluation.
    """)
    return


@app.cell
def _(np, pd):
    def make_quote_history(rows: int = 1800, seed: int = 7):
        """Create reproducible demonstration data for the quote workflow."""
        rng = np.random.default_rng(seed)
        coverage_values = np.array(["Basic", "Standard", "Premium"])
        vehicle_values = np.array(["Sedan", "SUV", "Truck", "EV"])
        region_values = np.array(["Urban", "Suburban", "Rural"])

        frame = pd.DataFrame(
            {
                "customer_age": rng.integers(21, 76, rows),
                "vehicle_age": rng.integers(0, 16, rows),
                "annual_mileage": rng.integers(5_000, 30_001, rows),
                "prior_claims": rng.choice(
                    [0, 1, 2, 3], rows, p=[0.61, 0.25, 0.10, 0.04]
                ),
                "tenure_years": rng.integers(0, 16, rows),
                "bundled_home": rng.integers(0, 2, rows),
                "coverage_tier": rng.choice(
                    coverage_values, rows, p=[0.28, 0.47, 0.25]
                ),
                "vehicle_type": rng.choice(
                    vehicle_values, rows, p=[0.41, 0.30, 0.14, 0.15]
                ),
                "region": rng.choice(
                    region_values, rows, p=[0.39, 0.41, 0.20]
                ),
                "deductible": rng.choice(
                    [250, 500, 750, 1000, 1500],
                    rows,
                    p=[0.12, 0.34, 0.18, 0.26, 0.10],
                ),
                "discount_pct": rng.integers(0, 21, rows),
            }
        )

        coverage_factor = frame["coverage_tier"].map(
            {"Basic": 0.88, "Standard": 1.15, "Premium": 1.48}
        )
        vehicle_factor = frame["vehicle_type"].map(
            {"Sedan": 1.00, "SUV": 1.13, "Truck": 1.20, "EV": 1.08}
        )
        region_factor = frame["region"].map(
            {"Urban": 1.13, "Suburban": 1.00, "Rural": 0.92}
        )
        frame["reference_premium"] = (
            (
                410
                + frame["vehicle_age"] * 22
                + frame["annual_mileage"] * 0.017
                + frame["prior_claims"] * 240
                + (35 - frame["customer_age"]).clip(lower=0) * 8
            )
            * coverage_factor
            * vehicle_factor
            * region_factor
        ).round(0)
        frame["quoted_premium"] = (
            frame["reference_premium"] * (1 - frame["discount_pct"] / 100)
        ).round(0)
        frame["price_to_reference"] = (
            frame["quoted_premium"] / frame["reference_premium"]
        ).round(4)

        # The hidden response process is deliberately nonlinear and noisy. The
        # notebook evaluates whether a model can recover it from examples.
        logit = (
            -0.35
            - 5.0 * (frame["price_to_reference"] - 0.90)
            + 0.0010 * frame["deductible"]
            + 0.075 * frame["tenure_years"]
            + 0.55 * frame["bundled_home"]
            - 0.42 * frame["prior_claims"]
            - 0.42 * (frame["coverage_tier"] == "Premium").astype(float)
            + 0.35
            * (
                (frame["vehicle_type"] == "EV")
                & (frame["region"] == "Urban")
            ).astype(float)
            - 0.55
            * (
                (frame["customer_age"] < 28)
                & (frame["vehicle_type"] == "Truck")
            ).astype(float)
            + rng.normal(0, 0.65, rows)
        )
        acceptance_probability = 1 / (1 + np.exp(-logit))
        frame["accepted"] = rng.binomial(1, acceptance_probability)
        return frame

    quote_history = make_quote_history()
    return make_quote_history, quote_history


@app.cell(hide_code=True)
def _(mo, quote_history):
    mo.md(
        f"""
        ## Historical quote table

        - **Rows:** `{len(quote_history):,}`
        - **Accepted:** `{quote_history["accepted"].mean():.1%}`
        - **Source:** reproducible synthetic demonstration data

        IBM reports using roughly 15 million real quote records. This notebook uses
        synthetic data because those records are proprietary. Therefore, it can
        demonstrate the mechanism and evaluation discipline, but not IBM's reported
        business impact or production-scale performance.
        """
    )
    return


@app.cell
def _(quote_history):
    feature_columns = [
        "customer_age",
        "vehicle_age",
        "annual_mileage",
        "prior_claims",
        "tenure_years",
        "bundled_home",
        "coverage_tier",
        "vehicle_type",
        "region",
        "reference_premium",
        "deductible",
        "discount_pct",
        "quoted_premium",
        "price_to_reference",
    ]
    categorical_columns = ["coverage_tier", "vehicle_type", "region"]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    X = quote_history[feature_columns].copy()
    y = quote_history["accepted"].astype(int)
    return X, categorical_columns, feature_columns, numeric_columns, y


@app.cell
def _(
    DummyClassifier,
    TabFMClassifier,
    X,
    pd,
    tabfm_v1_0_0,
    train_test_split,
    y,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    # TabFM defaults to a bounded 100-row context. Use a representative,
    # stratified sample and keep the holdout completely separate.
    X_context, _, y_context, _ = train_test_split(
        X_train,
        y_train,
        train_size=100,
        stratify=y_train,
        random_state=42,
    )

    tabfm_model = tabfm_v1_0_0.load(model_type="classification")
    tabfm_classifier = TabFMClassifier(model=tabfm_model)
    tabfm_classifier.fit(X_context, y_context.to_numpy())
    positive_index = list(tabfm_classifier.classes_).index(1)
    tabfm_scores = pd.Series(
        tabfm_classifier.predict_proba(X_test)[:, positive_index],
        index=X_test.index,
        name="tabfm_acceptance_probability",
    )

    dummy_classifier = DummyClassifier(strategy="prior")
    dummy_classifier.fit(X_context, y_context)
    dummy_scores = pd.Series(
        dummy_classifier.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="class_prior_probability",
    )
    return (
        X_context,
        X_test,
        dummy_scores,
        positive_index,
        tabfm_classifier,
        tabfm_scores,
        y_context,
        y_test,
    )


@app.cell
def _(
    ColumnTransformer,
    LogisticRegression,
    OneHotEncoder,
    StandardScaler,
    X_context,
    X_test,
    categorical_columns,
    make_pipeline,
    numeric_columns,
    pd,
    y_context,
):
    logistic_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )
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
    return logistic_classifier, logistic_scores


@app.cell
def _(average_precision_score, np, roc_auc_score):
    def bootstrap_auc_interval(labels, scores, repeats=400, seed=91):
        rng = np.random.default_rng(seed)
        labels_array = np.asarray(labels)
        scores_array = np.asarray(scores)
        values = []
        for _ in range(repeats):
            positions = rng.integers(0, len(labels_array), len(labels_array))
            sampled_labels = labels_array[positions]
            if np.unique(sampled_labels).size == 2:
                values.append(
                    roc_auc_score(sampled_labels, scores_array[positions])
                )
        return tuple(np.quantile(values, [0.025, 0.975]))

    def evaluate_signal(labels, scores):
        return {
            "ROC AUC": roc_auc_score(labels, scores),
            "Average precision": average_precision_score(labels, scores),
        }

    return bootstrap_auc_interval, evaluate_signal


@app.cell
def _(
    bootstrap_auc_interval,
    dummy_scores,
    evaluate_signal,
    logistic_scores,
    pd,
    tabfm_scores,
    y_test,
):
    prevalence = float(y_test.mean())
    tabfm_metrics = evaluate_signal(y_test, tabfm_scores)
    logistic_metrics = evaluate_signal(y_test, logistic_scores)
    dummy_metrics = evaluate_signal(y_test, dummy_scores)
    tabfm_auc_low, tabfm_auc_high = bootstrap_auc_interval(y_test, tabfm_scores)

    benchmark_table = pd.DataFrame(
        [
            {
                "Signal": "TabFM (100 context rows, no weight updates)",
                **tabfm_metrics,
                "AP lift over prevalence": (
                    tabfm_metrics["Average precision"] / prevalence
                ),
            },
            {
                "Signal": "Untuned logistic regression (same 100 rows)",
                **logistic_metrics,
                "AP lift over prevalence": (
                    logistic_metrics["Average precision"] / prevalence
                ),
            },
            {
                "Signal": "No-skill class-prior baseline",
                **dummy_metrics,
                "AP lift over prevalence": (
                    dummy_metrics["Average precision"] / prevalence
                ),
            },
        ]
    )

    signal_demonstrated = (
        tabfm_auc_low > 0.5
        and tabfm_metrics["Average precision"] > prevalence
    )
    competitive_with_logistic = (
        signal_demonstrated
        and tabfm_metrics["ROC AUC"] >= logistic_metrics["ROC AUC"] - 0.03
    )
    evidence_label = (
        "PASS: held-out TabFM signal demonstrated"
        if signal_demonstrated
        else "NOT ESTABLISHED: do not claim TabFM is useful on this run"
    )
    comparison_label = (
        "COMPETITIVE: TabFM is within 0.03 ROC AUC of logistic regression"
        if competitive_with_logistic
        else (
            "SIGNAL ONLY: TabFM works, but logistic regression is materially stronger"
            if signal_demonstrated
            else "NO COMPARISON CLAIM: the TabFM signal gate did not pass"
        )
    )
    return (
        benchmark_table,
        comparison_label,
        evidence_label,
        prevalence,
        signal_demonstrated,
        tabfm_auc_high,
        tabfm_auc_low,
        tabfm_metrics,
    )


@app.cell(hide_code=True)
def _(
    benchmark_table,
    comparison_label,
    evidence_label,
    mo,
    prevalence,
    tabfm_auc_high,
    tabfm_auc_low,
):
    evidence_kind = "success" if evidence_label.startswith("PASS") else "warn"
    mo.vstack(
        [
            mo.md("## Evidence gate: does TabFM add a useful signal?"),
            mo.md(
                f"""
                **{evidence_label}**

                **{comparison_label}**

                - Holdout acceptance prevalence: `{prevalence:.3f}`
                - TabFM ROC AUC 95% bootstrap interval:
                  `[{tabfm_auc_low:.3f}, {tabfm_auc_high:.3f}]`

                Average precision must be compared with prevalence. ROC AUC must
                beat `0.5`. The interval makes the notebook less likely to celebrate
                a chance fluctuation.
                """
            ).callout(kind=evidence_kind),
            benchmark_table.style.format(
                {
                    "ROC AUC": "{:.3f}",
                    "Average precision": "{:.3f}",
                    "AP lift over prevalence": "{:.2f}x",
                }
            ),
        ]
    )
    return


@app.cell
def _(
    ColumnTransformer,
    NearestNeighbors,
    OneHotEncoder,
    StandardScaler,
    X,
    categorical_columns,
    numeric_columns,
):
    similarity_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )
    similarity_matrix = similarity_preprocessor.fit_transform(X)
    similarity_index = NearestNeighbors(n_neighbors=8, metric="euclidean")
    similarity_index.fit(similarity_matrix)
    return similarity_index, similarity_preprocessor


@app.cell
def _(mo, quote_history):
    seed_quote = quote_history.iloc[0]
    coverage = mo.ui.dropdown(
        options=sorted(quote_history["coverage_tier"].unique().tolist()),
        value=str(seed_quote["coverage_tier"]),
        label="Coverage tier",
    )
    vehicle = mo.ui.dropdown(
        options=sorted(quote_history["vehicle_type"].unique().tolist()),
        value=str(seed_quote["vehicle_type"]),
        label="Vehicle type",
    )
    region = mo.ui.dropdown(
        options=sorted(quote_history["region"].unique().tolist()),
        value=str(seed_quote["region"]),
        label="Region",
    )
    customer_age = mo.ui.slider(
        21, 75, value=int(seed_quote["customer_age"]), label="Customer age"
    )
    vehicle_age = mo.ui.slider(
        0, 15, value=int(seed_quote["vehicle_age"]), label="Vehicle age"
    )
    annual_mileage = mo.ui.number(
        5_000,
        30_000,
        value=float(seed_quote["annual_mileage"]),
        step=500,
        label="Annual mileage",
    )
    prior_claims = mo.ui.slider(
        0, 3, value=int(seed_quote["prior_claims"]), label="Prior claims"
    )
    tenure_years = mo.ui.slider(
        0, 15, value=int(seed_quote["tenure_years"]), label="Tenure (years)"
    )
    bundled_home = mo.ui.switch(
        value=bool(seed_quote["bundled_home"]), label="Home policy bundled"
    )
    deductible = mo.ui.dropdown(
        options=[250, 500, 750, 1000, 1500],
        value=int(seed_quote["deductible"]),
        label="Deductible",
    )
    discount_pct = mo.ui.slider(
        0,
        20,
        value=int(seed_quote["discount_pct"]),
        label="Discount (%)",
        show_value=True,
    )
    controls = mo.vstack(
        [
            mo.md("## Candidate quote"),
            mo.hstack([coverage, vehicle, region]),
            mo.hstack([customer_age, vehicle_age, annual_mileage]),
            mo.hstack([prior_claims, tenure_years, bundled_home]),
            mo.hstack([deductible, discount_pct]),
        ]
    )
    controls
    return (
        annual_mileage,
        bundled_home,
        coverage,
        customer_age,
        deductible,
        discount_pct,
        prior_claims,
        region,
        tenure_years,
        vehicle,
        vehicle_age,
    )


@app.cell
def _(pd):
    def build_quote(
        *,
        customer_age,
        vehicle_age,
        annual_mileage,
        prior_claims,
        tenure_years,
        bundled_home,
        coverage_tier,
        vehicle_type,
        region,
        deductible,
        discount_pct,
    ):
        coverage_factor = {
            "Basic": 0.88,
            "Standard": 1.15,
            "Premium": 1.48,
        }[coverage_tier]
        vehicle_factor = {
            "Sedan": 1.00,
            "SUV": 1.13,
            "Truck": 1.20,
            "EV": 1.08,
        }[vehicle_type]
        region_factor = {
            "Urban": 1.13,
            "Suburban": 1.00,
            "Rural": 0.92,
        }[region]
        reference_premium = (
            410
            + vehicle_age * 22
            + annual_mileage * 0.017
            + prior_claims * 240
            + max(35 - customer_age, 0) * 8
        ) * coverage_factor * vehicle_factor * region_factor
        quoted_premium = reference_premium * (1 - discount_pct / 100)
        return pd.DataFrame(
            [
                {
                    "customer_age": float(customer_age),
                    "vehicle_age": float(vehicle_age),
                    "annual_mileage": float(annual_mileage),
                    "prior_claims": float(prior_claims),
                    "tenure_years": float(tenure_years),
                    "bundled_home": int(bundled_home),
                    "coverage_tier": coverage_tier,
                    "vehicle_type": vehicle_type,
                    "region": region,
                    "reference_premium": round(reference_premium, 0),
                    "deductible": float(deductible),
                    "discount_pct": float(discount_pct),
                    "quoted_premium": round(quoted_premium, 0),
                    "price_to_reference": round(
                        quoted_premium / reference_premium, 4
                    ),
                }
            ]
        )

    return (build_quote,)


@app.cell
def _(
    annual_mileage,
    build_quote,
    bundled_home,
    coverage,
    customer_age,
    deductible,
    discount_pct,
    feature_columns,
    positive_index,
    prior_claims,
    region,
    tabfm_classifier,
    tenure_years,
    vehicle,
    vehicle_age,
):
    candidate_quote = build_quote(
        customer_age=customer_age.value,
        vehicle_age=vehicle_age.value,
        annual_mileage=annual_mileage.value,
        prior_claims=prior_claims.value,
        tenure_years=tenure_years.value,
        bundled_home=bundled_home.value,
        coverage_tier=coverage.value,
        vehicle_type=vehicle.value,
        region=region.value,
        deductible=deductible.value,
        discount_pct=discount_pct.value,
    )
    candidate_probability = float(
        tabfm_classifier.predict_proba(candidate_quote[feature_columns])[
            0, positive_index
        ]
    )
    return candidate_probability, candidate_quote


@app.cell
def _(
    build_quote,
    candidate_quote,
    deductible,
    discount_pct,
    feature_columns,
    pd,
    positive_index,
    tabfm_classifier,
):
    scenario_rows = []
    base = candidate_quote.iloc[0]
    discount_values = sorted(
        {
            max(0, int(discount_pct.value) - 5),
            int(discount_pct.value),
            min(20, int(discount_pct.value) + 5),
            min(20, int(discount_pct.value) + 10),
        }
    )
    deductible_values = sorted(
        {int(deductible.value), 500, 1000, 1500}
    )
    for scenario_discount in discount_values:
        for scenario_deductible in deductible_values:
            scenario_rows.append(
                build_quote(
                    customer_age=base["customer_age"],
                    vehicle_age=base["vehicle_age"],
                    annual_mileage=base["annual_mileage"],
                    prior_claims=base["prior_claims"],
                    tenure_years=base["tenure_years"],
                    bundled_home=bool(base["bundled_home"]),
                    coverage_tier=base["coverage_tier"],
                    vehicle_type=base["vehicle_type"],
                    region=base["region"],
                    deductible=scenario_deductible,
                    discount_pct=scenario_discount,
                )
            )
    scenario_quotes = pd.concat(scenario_rows, ignore_index=True)
    scenario_quotes["tabfm_acceptance_probability"] = (
        tabfm_classifier.predict_proba(scenario_quotes[feature_columns])[
            :, positive_index
        ]
    )
    scenario_results = scenario_quotes[
        [
            "discount_pct",
            "deductible",
            "quoted_premium",
            "tabfm_acceptance_probability",
        ]
    ].sort_values(
        "tabfm_acceptance_probability",
        ascending=False,
    )
    return (scenario_results,)


@app.cell
def _(
    candidate_probability,
    candidate_quote,
    mo,
    prevalence,
    signal_demonstrated,
    tabfm_scores,
):
    score_percentile = float((tabfm_scores <= candidate_probability).mean())
    result_kind = "success" if signal_demonstrated else "warn"
    validation_text = (
        "The holdout evidence gate passed, so this score can be used to "
        "demonstrate model-driven ranking on this synthetic dataset."
        if signal_demonstrated
        else "The evidence gate did not pass. Treat this as a UI output, not "
        "as a demonstrated useful prediction."
    )
    mo.md(
        f"""
        ## Current quote result

        - **TabFM acceptance probability:** `{candidate_probability:.1%}`
        - **Holdout score percentile:** `{score_percentile:.1%}`
        - **Historical acceptance rate:** `{prevalence:.1%}`
        - **Quoted annual premium:** `${candidate_quote.loc[0, "quoted_premium"]:,.0f}`
        - **Deductible:** `${candidate_quote.loc[0, "deductible"]:,.0f}`
        - **Discount:** `{candidate_quote.loc[0, "discount_pct"]:.0f}%`

        {validation_text}
        """
    ).callout(kind=result_kind)
    return


@app.cell(hide_code=True)
def _(mo, scenario_results):
    display_scenarios = scenario_results.copy()
    display_scenarios["tabfm_acceptance_probability"] = display_scenarios[
        "tabfm_acceptance_probability"
    ].map(lambda value: f"{value:.1%}")
    mo.vstack(
        [
            mo.md(
                """
                ## What-if quote optimization

                These are model-scored alternatives, not automatic business
                recommendations. A real deployment would also include margin,
                underwriting, fairness, and regulatory constraints.
                """
            ),
            display_scenarios.head(10),
        ]
    )
    return


@app.cell
def _(
    candidate_quote,
    feature_columns,
    quote_history,
    similarity_index,
    similarity_preprocessor,
):
    candidate_vector = similarity_preprocessor.transform(
        candidate_quote[feature_columns]
    )
    _, neighbor_positions = similarity_index.kneighbors(candidate_vector)
    similar_quotes = quote_history.iloc[neighbor_positions[0]].copy()
    similar_quotes.insert(
        0, "similarity_rank", range(1, len(similar_quotes) + 1)
    )
    similar_quote_columns = [
        "similarity_rank",
        "coverage_tier",
        "vehicle_type",
        "region",
        "customer_age",
        "prior_claims",
        "tenure_years",
        "quoted_premium",
        "deductible",
        "discount_pct",
        "accepted",
    ]
    return similar_quote_columns, similar_quotes


@app.cell(hide_code=True)
def _(mo, similar_quote_columns, similar_quotes):
    mo.vstack(
        [
            mo.md(
                """
                ## Similar historical quotes

                This retrieval layer uses standardized distance over quote
                attributes. It is separate from TabFM and is shown explicitly so
                the demo does not attribute every component to the foundation model.
                """
            ),
            similar_quotes[similar_quote_columns],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What the prototype establishes—and what it does not

    **If the evidence gate passes**, this run demonstrates that a pretrained
    tabular foundation model can extract a held-out signal from a small context
    table without dataset-specific weight training or hyperparameter search, then
    power the interaction pattern described in IBM's article.

    It does **not** establish:

    - that IBM SQL Data Insights uses TabFM or a transformer,
    - that synthetic holdout performance transfers to real insurance customers,
    - that TabFM beats tuned production models,
    - that the displayed probabilities are production-calibrated,
    - or that the non-commercial TabFM weights can be deployed commercially.

    The next credible step is to replace `make_quote_history()` with a
    representative, governance-approved quote table and rerun the same evidence
    gate before making any business claim.
    """)
    return


if __name__ == "__main__":
    app.run()
