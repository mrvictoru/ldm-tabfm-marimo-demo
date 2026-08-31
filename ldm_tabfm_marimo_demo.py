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
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        mean_absolute_error,
        mean_squared_error,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from tabfm import (
        TabFMClassifier,
        TabFMRegressor,
        tabfm_v1_0_0_pytorch as tabfm_v1_0_0,
    )

    return (
        ColumnTransformer,
        DummyClassifier,
        DummyRegressor,
        LinearRegression,
        LogisticRegression,
        NearestNeighbors,
        OneHotEncoder,
        StandardScaler,
        TabFMClassifier,
        TabFMRegressor,
        average_precision_score,
        make_pipeline,
        mean_absolute_error,
        mean_squared_error,
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
    **TabFM** tabular foundation model. It then asks a **second question of the
    same table**: expected loss cost.

    > **Important boundary:** This is an independent technical prototype. It does
    > not reproduce IBM SQL Data Insights, and it does not establish that IBM's
    > product uses TabFM or the same architecture.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What this notebook is designed to prove

    A probability or a dollar amount is not enough. The notebook must show:

    **Demo 1 — will the customer accept this quote?**

    - **Held-out ranking signal** on unseen quotes.
    - **No-skill and conventional baselines** beside TabFM.
    - **Decision-time behavior:** changing terms recalculates the score.
    - **Named next-best offers** ranked by the same classifier.
    - **Historical grounding** from a separate nearest-neighbor layer.

    **Demo 2 — what expected loss cost is associated with this quote?**

    - The **same 100 context rows** and the **same candidate quote**.
    - A **new target**, with TabFM loaded as a regressor.
    - **MAE / RMSE versus a mean baseline** and untuned linear regression.
    - A **tradeoff view:** extra discount mainly moves acceptance; deductible
      can move both acceptance and expected cost.

    TabFM is zero-shot in the model-training sense: `fit()` prepares the table
    and supplies labeled rows as in-context examples; it does not update the
    pretrained weights. Each new target still needs its own evidence gate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup

    The first code cell loaded marimo, pandas, NumPy, scikit-learn, and TabFM.
    Two pretrained checkpoints may download the first time you run this notebook:

    1. **classification** for Demo 1,
    2. **regression** later, only when Demo 2 runs.

    Run every cell once before a live talk, and keep the session open.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Synthetic quote history

    IBM's production example uses proprietary quote records, so this generator
    builds a reproducible demonstration table. Each row is one quote. The model
    sees features and labels, **not** the hidden formulas that created
    `accepted` and `expected_loss_cost`.
    """)
    return


@app.cell
def _(np, pd):
    def make_quote_history(rows: int = 1800, seed: int = 7):
        """Create a reproducible quote table with two synthetic targets.

        `accepted` is a noisy classification label. `expected_loss_cost` is a
        noisy regression label driven by risk and coverage, not by discount.
        """
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

        # Hidden acceptance process: nonlinear, noisy, and unknown to the model.
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
        frame["accepted"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

        # Hidden loss-cost process: risk and coverage, not commercial discount.
        coverage_loss_factor = frame["coverage_tier"].map(
            {"Basic": 0.82, "Standard": 1.00, "Premium": 1.28}
        )
        expected_loss = (
            220
            + frame["vehicle_age"] * 28
            + frame["annual_mileage"] * 0.014
            + frame["prior_claims"] * 380
            + (40 - frame["customer_age"]).clip(lower=0) * 7
            + (frame["vehicle_type"] == "Truck").astype(float) * 160
            + (frame["vehicle_type"] == "SUV").astype(float) * 90
            + (frame["region"] == "Urban").astype(float) * 70
            - frame["deductible"] * 0.42
        ) * coverage_loss_factor
        frame["expected_loss_cost"] = (
            (expected_loss + rng.normal(0, 35, rows)).clip(lower=40)
        ).round(0)
        return frame

    quote_history = make_quote_history()
    return make_quote_history, quote_history


@app.cell(hide_code=True)
def _(mo, quote_history):
    mo.md(
        f"""
        ### Historical quote table

        - **Rows:** `{len(quote_history):,}`
        - **Accepted:** `{quote_history["accepted"].mean():.1%}`
        - **Mean expected loss cost:** `${quote_history["expected_loss_cost"].mean():,.0f}`
        - **Source:** reproducible synthetic demonstration data

        IBM reports using roughly 15 million real quote records. This notebook
        uses synthetic data because those records are proprietary. It can
        demonstrate mechanism and evaluation discipline, not IBM's reported
        business impact.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Demo 1: will the customer accept this quote?

    The table contract for both demos is the same feature schema. Demo 1's
    target is `accepted`. `expected_loss_cost` is held out as a later label
    and is **not** a model feature.
    """)
    return


@app.cell
def _(quote_history):
    # Shared feature schema for classification, regression, and retrieval.
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
    y_accept = quote_history["accepted"].astype(int)
    y_cost = quote_history["expected_loss_cost"].astype(float)
    return (
        X,
        categorical_columns,
        feature_columns,
        numeric_columns,
        y_accept,
        y_cost,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Split once, then score with TabFM classification

    1. Hold out 25% of rows for evaluation.
    2. Sample **100 stratified context rows** from the remainder.
    3. Load the classification checkpoint and call `fit()` — this stores
       context, it does **not** update TabFM weights.
    4. Score the untouched holdout.

    Demo 2 will reuse these same row indices with a different label.
    """)
    return


@app.cell
def _(
    DummyClassifier,
    TabFMClassifier,
    X,
    pd,
    tabfm_v1_0_0,
    train_test_split,
    y_accept,
    y_cost,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_accept,
        test_size=0.25,
        stratify=y_accept,
        random_state=42,
    )

    # TabFM defaults to a bounded 100-row context. Keep holdout untouched.
    X_context, _, y_context, _ = train_test_split(
        X_train,
        y_train,
        train_size=100,
        stratify=y_train,
        random_state=42,
    )
    y_cost_context = y_cost.loc[X_context.index]
    y_cost_test = y_cost.loc[X_test.index]

    tabfm_classifier = TabFMClassifier(
        model=tabfm_v1_0_0.load(model_type="classification")
    )
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
        y_cost_context,
        y_cost_test,
        y_test,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Conventional comparison

    An untuned logistic regression sees the **same 100 context rows** and the
    same holdout. This answers whether TabFM is doing more than emitting a
    plausible-looking probability.
    """)
    return


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
    logistic_classifier = make_pipeline(
        ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns,
                ),
            ]
        ),
        LogisticRegression(max_iter=1_000),
    )
    logistic_classifier.fit(X_context, y_context)
    logistic_scores = pd.Series(
        logistic_classifier.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="logistic_acceptance_probability",
    )
    return logistic_classifier, logistic_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evidence gate: classification

    ROC AUC must beat `0.5`. Average precision must beat holdout prevalence.
    A bootstrap interval makes a chance fluctuation harder to celebrate. If
    this gate fails, do not treat live Demo 1 scores as demonstrated signal.
    """)
    return


@app.cell
def _(average_precision_score, np, roc_auc_score):
    def bootstrap_auc_interval(labels, scores, repeats=400, seed=91):
        """95% bootstrap interval for ROC AUC, skipping one-class resamples."""
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

    def evaluate_ranking(labels, scores):
        """Held-out ranking metrics for the acceptance task."""
        return {
            "ROC AUC": roc_auc_score(labels, scores),
            "Average precision": average_precision_score(labels, scores),
        }

    return bootstrap_auc_interval, evaluate_ranking


@app.cell
def _(
    bootstrap_auc_interval,
    dummy_scores,
    evaluate_ranking,
    logistic_scores,
    pd,
    tabfm_scores,
    y_test,
):
    prevalence = float(y_test.mean())
    tabfm_metrics = evaluate_ranking(y_test, tabfm_scores)
    logistic_metrics = evaluate_ranking(y_test, logistic_scores)
    dummy_metrics = evaluate_ranking(y_test, dummy_scores)
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
        "PASS: held-out TabFM ranking signal demonstrated"
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
            mo.md("### Classification evidence"),
            mo.md(
                f"""
                **{evidence_label}**

                **{comparison_label}**

                - Holdout acceptance prevalence: `{prevalence:.3f}`
                - TabFM ROC AUC 95% bootstrap interval:
                  `[{tabfm_auc_low:.3f}, {tabfm_auc_high:.3f}]`

                Average precision must be compared with prevalence. ROC AUC must
                beat `0.5`.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Live quote, next-best offers, and similar cases

    After the gate, an agent can enter an eligible quote, see its score, rank a
    few named packages, and inspect similar historical rows. Retrieval uses
    scikit-learn nearest neighbors, **not** TabFM.
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
    numeric_columns,
):
    # Separate retrieval layer: standardized distance over quote attributes.
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
    similarity_index = NearestNeighbors(n_neighbors=8, metric="euclidean")
    similarity_index.fit(similarity_preprocessor.fit_transform(X))
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
    mo.vstack(
        [
            mo.md("### Candidate quote"),
            mo.hstack([coverage, vehicle, region]),
            mo.hstack([customer_age, vehicle_age, annual_mileage]),
            mo.hstack([prior_claims, tenure_years, bundled_home]),
            mo.hstack([deductible, discount_pct]),
        ]
    )
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
    DEDUCTIBLE_OPTIONS = [250, 500, 750, 1000, 1500]
    COVERAGE_ORDER = ["Basic", "Standard", "Premium"]

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
        """Rebuild premium fields from the same rating factors as the generator."""
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

    def quote_kwargs_from_row(row):
        """Turn a quote row into keyword arguments for `build_quote()`."""
        return {
            "customer_age": row["customer_age"],
            "vehicle_age": row["vehicle_age"],
            "annual_mileage": row["annual_mileage"],
            "prior_claims": row["prior_claims"],
            "tenure_years": row["tenure_years"],
            "bundled_home": bool(row["bundled_home"]),
            "coverage_tier": row["coverage_tier"],
            "vehicle_type": row["vehicle_type"],
            "region": row["region"],
            "deductible": row["deductible"],
            "discount_pct": row["discount_pct"],
        }

    def next_deductible(current):
        """Return the next listed deductible, or None at the top of the menu."""
        for option in DEDUCTIBLE_OPTIONS:
            if option > int(current):
                return option
        return None

    def next_coverage(current):
        """Return the next coverage tier, or None if already Premium."""
        index = COVERAGE_ORDER.index(current)
        if index < len(COVERAGE_ORDER) - 1:
            return COVERAGE_ORDER[index + 1]
        return None

    def build_named_offers(base_row):
        """Build a short menu of packages from the current candidate quote."""
        offers = []

        def add_offer(name, **overrides):
            kwargs = quote_kwargs_from_row(base_row)
            kwargs.update(overrides)
            offer = build_quote(**kwargs)
            offer.insert(0, "offer", name)
            offers.append(offer)

        add_offer("Current terms")
        extra_discount = min(20, int(base_row["discount_pct"]) + 5)
        if extra_discount != int(base_row["discount_pct"]):
            add_offer("Extra discount", discount_pct=extra_discount)
        higher_deductible = next_deductible(base_row["deductible"])
        if higher_deductible is not None:
            add_offer("Higher deductible", deductible=higher_deductible)
        if not bool(base_row["bundled_home"]):
            add_offer("Home bundle on", bundled_home=True)
        upgraded_coverage = next_coverage(base_row["coverage_tier"])
        if upgraded_coverage is not None:
            add_offer("Coverage upgrade", coverage_tier=upgraded_coverage)
        return pd.concat(offers, ignore_index=True)

    return build_named_offers, build_quote, quote_kwargs_from_row


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
    quote_kwargs_from_row,
    tabfm_classifier,
):
    # Discount × deductible grid around the current terms.
    base = candidate_quote.iloc[0]
    discount_values = sorted(
        {
            max(0, int(discount_pct.value) - 5),
            int(discount_pct.value),
            min(20, int(discount_pct.value) + 5),
            min(20, int(discount_pct.value) + 10),
        }
    )
    deductible_values = sorted({int(deductible.value), 500, 1000, 1500})
    scenario_quotes = pd.concat(
        [
            build_quote(
                **{
                    **quote_kwargs_from_row(base),
                    "deductible": scenario_deductible,
                    "discount_pct": scenario_discount,
                }
            )
            for scenario_discount in discount_values
            for scenario_deductible in deductible_values
        ],
        ignore_index=True,
    )
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
    ].sort_values("tabfm_acceptance_probability", ascending=False)
    return scenario_quotes, scenario_results


@app.cell
def _(
    build_named_offers,
    candidate_quote,
    feature_columns,
    positive_index,
    tabfm_classifier,
):
    offer_quotes = build_named_offers(candidate_quote.iloc[0])
    offer_quotes["tabfm_acceptance_probability"] = (
        tabfm_classifier.predict_proba(offer_quotes[feature_columns])[
            :, positive_index
        ]
    )
    offer_results = offer_quotes[
        [
            "offer",
            "coverage_tier",
            "bundled_home",
            "deductible",
            "discount_pct",
            "quoted_premium",
            "tabfm_acceptance_probability",
        ]
    ].sort_values("tabfm_acceptance_probability", ascending=False)
    return offer_quotes, offer_results


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
        "The classification evidence gate passed, so this score can be used "
        "to demonstrate model-driven ranking on this synthetic dataset."
        if signal_demonstrated
        else "The classification evidence gate did not pass. Treat this as a "
        "UI output, not as a demonstrated useful prediction."
    )
    mo.md(
        f"""
        ### Current quote result

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
                ### What-if quote terms

                These are model-scored alternatives, not automatic business
                recommendations. A real deployment would also include margin,
                underwriting, fairness, and regulatory constraints.
                """
            ),
            display_scenarios.head(10),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, offer_results):
    display_offers = offer_results.copy()
    display_offers["tabfm_acceptance_probability"] = display_offers[
        "tabfm_acceptance_probability"
    ].map(lambda value: f"{value:.1%}")
    mo.vstack(
        [
            mo.md(
                """
                ### Next-best offers

                Named packages scored by the **same** TabFM classifier. They are
                model-ranked alternatives, not approved products. Skip a row
                when the current quote already uses that option.
                """
            ),
            display_offers,
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
    _, neighbor_positions = similarity_index.kneighbors(
        similarity_preprocessor.transform(candidate_quote[feature_columns])
    )
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
        "expected_loss_cost",
    ]
    return similar_quote_columns, similar_quotes


@app.cell(hide_code=True)
def _(mo, similar_quote_columns, similar_quotes):
    mo.vstack(
        [
            mo.md(
                """
                ### Similar historical quotes

                This retrieval layer uses standardized distance over quote
                attributes. It is separate from TabFM and is shown so the demo
                does not attribute every component to the foundation model.
                """
            ),
            similar_quotes[similar_quote_columns],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Demo 2: expected loss cost

    Same table. Same 100 context rows. **New target.** TabFM is loaded here as
    a regressor so Demo 1 can finish before the second checkpoint downloads.
    `fit()` again stores context; it does not update pretrained weights.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    DummyRegressor,
    LinearRegression,
    OneHotEncoder,
    StandardScaler,
    TabFMRegressor,
    X_context,
    X_test,
    categorical_columns,
    make_pipeline,
    numeric_columns,
    pd,
    tabfm_v1_0_0,
    y_cost_context,
):
    # Second checkpoint: load only when this cell runs.
    tabfm_regressor = TabFMRegressor(
        model=tabfm_v1_0_0.load(model_type="regression")
    )
    tabfm_regressor.fit(X_context, y_cost_context.to_numpy())
    tabfm_cost_scores = pd.Series(
        tabfm_regressor.predict(X_test),
        index=X_test.index,
        name="tabfm_expected_loss_cost",
    )

    dummy_regressor = DummyRegressor(strategy="mean")
    dummy_regressor.fit(X_context, y_cost_context)
    dummy_cost_scores = pd.Series(
        dummy_regressor.predict(X_test),
        index=X_test.index,
        name="mean_expected_loss_cost",
    )

    linear_regressor = make_pipeline(
        ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns,
                ),
            ]
        ),
        LinearRegression(),
    )
    linear_regressor.fit(X_context, y_cost_context)
    linear_cost_scores = pd.Series(
        linear_regressor.predict(X_test),
        index=X_test.index,
        name="linear_expected_loss_cost",
    )
    return (
        dummy_cost_scores,
        linear_cost_scores,
        tabfm_cost_scores,
        tabfm_regressor,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evidence gate: regression

    Pass only if TabFM MAE is **strictly below** the mean baseline. If this
    gate fails, do not treat live cost numbers as demonstrated signal.
    """)
    return


@app.cell
def _(mean_absolute_error, mean_squared_error, np):
    def evaluate_cost(labels, scores):
        """Held-out error metrics for the expected-loss task."""
        return {
            "MAE": mean_absolute_error(labels, scores),
            "RMSE": float(np.sqrt(mean_squared_error(labels, scores))),
        }

    return (evaluate_cost,)


@app.cell
def _(
    dummy_cost_scores,
    evaluate_cost,
    linear_cost_scores,
    pd,
    tabfm_cost_scores,
    y_cost_test,
):
    tabfm_cost_metrics = evaluate_cost(y_cost_test, tabfm_cost_scores)
    linear_cost_metrics = evaluate_cost(y_cost_test, linear_cost_scores)
    dummy_cost_metrics = evaluate_cost(y_cost_test, dummy_cost_scores)
    holdout_mean_cost = float(y_cost_test.mean())

    cost_benchmark_table = pd.DataFrame(
        [
            {
                "Signal": "TabFM (same 100 context rows, no weight updates)",
                **tabfm_cost_metrics,
                "MAE lift vs mean baseline": (
                    dummy_cost_metrics["MAE"] / tabfm_cost_metrics["MAE"]
                ),
            },
            {
                "Signal": "Untuned linear regression (same 100 rows)",
                **linear_cost_metrics,
                "MAE lift vs mean baseline": (
                    dummy_cost_metrics["MAE"] / linear_cost_metrics["MAE"]
                ),
            },
            {
                "Signal": "No-skill mean baseline",
                **dummy_cost_metrics,
                "MAE lift vs mean baseline": 1.0,
            },
        ]
    )

    cost_signal_demonstrated = (
        tabfm_cost_metrics["MAE"] < dummy_cost_metrics["MAE"]
    )
    competitive_with_linear = (
        cost_signal_demonstrated
        and tabfm_cost_metrics["MAE"] <= linear_cost_metrics["MAE"] * 1.10
    )
    cost_evidence_label = (
        "PASS: held-out TabFM cost signal demonstrated"
        if cost_signal_demonstrated
        else "NOT ESTABLISHED: do not claim TabFM cost predictions are useful on this run"
    )
    cost_comparison_label = (
        "COMPETITIVE: TabFM MAE is within 10% of linear regression"
        if competitive_with_linear
        else (
            "SIGNAL ONLY: TabFM beats the mean, but linear regression is materially stronger"
            if cost_signal_demonstrated
            else "NO COMPARISON CLAIM: the TabFM cost gate did not pass"
        )
    )
    return (
        cost_benchmark_table,
        cost_comparison_label,
        cost_evidence_label,
        cost_signal_demonstrated,
        holdout_mean_cost,
        tabfm_cost_metrics,
    )


@app.cell(hide_code=True)
def _(
    cost_benchmark_table,
    cost_comparison_label,
    cost_evidence_label,
    holdout_mean_cost,
    mo,
):
    cost_evidence_kind = (
        "success" if cost_evidence_label.startswith("PASS") else "warn"
    )
    mo.vstack(
        [
            mo.md("### Regression evidence"),
            mo.md(
                f"""
                **{cost_evidence_label}**

                **{cost_comparison_label}**

                - Holdout mean expected loss: `${holdout_mean_cost:,.0f}`
                - Gate: TabFM MAE must be strictly below the mean baseline.
                """
            ).callout(kind=cost_evidence_kind),
            cost_benchmark_table.style.format(
                {
                    "MAE": "{:.1f}",
                    "RMSE": "{:.1f}",
                    "MAE lift vs mean baseline": "{:.2f}x",
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Tradeoff: acceptance versus expected loss

    Score the **same** candidate, term grid, and named offers with the
    regressor. Extra discount should mainly lift acceptance; a higher
    deductible can move both numbers. These are model sensitivities, not
    causal effects.
    """)
    return


@app.cell
def _(
    candidate_quote,
    feature_columns,
    offer_quotes,
    scenario_quotes,
    tabfm_regressor,
):
    candidate_expected_loss = float(
        tabfm_regressor.predict(candidate_quote[feature_columns])[0]
    )
    scenario_tradeoff = scenario_quotes.copy()
    scenario_tradeoff["tabfm_expected_loss_cost"] = tabfm_regressor.predict(
        scenario_tradeoff[feature_columns]
    )
    scenario_tradeoff = scenario_tradeoff[
        [
            "discount_pct",
            "deductible",
            "quoted_premium",
            "tabfm_acceptance_probability",
            "tabfm_expected_loss_cost",
        ]
    ].sort_values("tabfm_acceptance_probability", ascending=False)

    offer_tradeoff = offer_quotes.copy()
    offer_tradeoff["tabfm_expected_loss_cost"] = tabfm_regressor.predict(
        offer_tradeoff[feature_columns]
    )
    offer_tradeoff = offer_tradeoff[
        [
            "offer",
            "coverage_tier",
            "bundled_home",
            "deductible",
            "discount_pct",
            "quoted_premium",
            "tabfm_acceptance_probability",
            "tabfm_expected_loss_cost",
        ]
    ].sort_values("tabfm_acceptance_probability", ascending=False)
    return candidate_expected_loss, offer_tradeoff, scenario_tradeoff


@app.cell
def _(
    candidate_expected_loss,
    candidate_probability,
    candidate_quote,
    cost_signal_demonstrated,
    mo,
):
    cost_kind = "success" if cost_signal_demonstrated else "warn"
    cost_validation = (
        "The regression evidence gate passed, so this cost can be used to "
        "demonstrate a second-task score on this synthetic dataset."
        if cost_signal_demonstrated
        else "The regression evidence gate did not pass. Treat this cost as a "
        "UI output, not as a demonstrated useful prediction."
    )
    mo.md(
        f"""
        ### Same quote, second number

        - **TabFM acceptance probability:** `{candidate_probability:.1%}`
        - **TabFM expected loss cost:** `${candidate_expected_loss:,.0f}`
        - **Quoted annual premium:** `${candidate_quote.loc[0, "quoted_premium"]:,.0f}`
        - **Deductible:** `${candidate_quote.loc[0, "deductible"]:,.0f}`
        - **Discount:** `{candidate_quote.loc[0, "discount_pct"]:.0f}%`

        {cost_validation}
        """
    ).callout(kind=cost_kind)
    return


@app.cell(hide_code=True)
def _(mo, scenario_tradeoff):
    display_tradeoff = scenario_tradeoff.copy()
    display_tradeoff["tabfm_acceptance_probability"] = display_tradeoff[
        "tabfm_acceptance_probability"
    ].map(lambda value: f"{value:.1%}")
    display_tradeoff["tabfm_expected_loss_cost"] = display_tradeoff[
        "tabfm_expected_loss_cost"
    ].map(lambda value: f"${value:,.0f}")
    mo.vstack(
        [
            mo.md(
                """
                ### Term grid with both scores

                Watch discount rows versus deductible rows. Discount is not in
                the hidden loss process; deductible is.
                """
            ),
            display_tradeoff.head(10),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, offer_tradeoff):
    display_offer_tradeoff = offer_tradeoff.copy()
    display_offer_tradeoff["tabfm_acceptance_probability"] = (
        display_offer_tradeoff["tabfm_acceptance_probability"].map(
            lambda value: f"{value:.1%}"
        )
    )
    display_offer_tradeoff["tabfm_expected_loss_cost"] = (
        display_offer_tradeoff["tabfm_expected_loss_cost"].map(
            lambda value: f"${value:,.0f}"
        )
    )
    mo.vstack(
        [
            mo.md(
                """
                ### Named offers with both scores

                Use this to discuss a package the customer might accept without
                treating expected loss as an approved price or an underwriting
                decision.
                """
            ),
            display_offer_tradeoff,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What the prototype establishes—and what it does not

    **If both evidence gates pass**, this run demonstrates that a pretrained
    tabular foundation model can extract held-out signal from a small context
    table on **two different targets** without dataset-specific weight
    training, then power the interaction pattern described in IBM's article.

    It does **not** establish:

    - that IBM SQL Data Insights uses TabFM or a transformer,
    - that synthetic holdout performance transfers to real insurance customers,
    - that TabFM beats tuned production models,
    - that displayed probabilities or costs are production-calibrated,
    - that what-if score changes are causal,
    - that named offers are approved products,
    - that similar-case retrieval is TabFM explaining itself,
    - or that the non-commercial TabFM weights can be deployed commercially.

    The next credible step is to replace `make_quote_history()` with a
    representative, governance-approved quote table that includes **both**
    labels, then rerun **both** evidence gates before making any business claim.
    """)
    return


if __name__ == "__main__":
    app.run()
