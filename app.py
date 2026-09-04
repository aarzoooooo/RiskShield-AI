import streamlit as st
import pandas as pd
import joblib

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "data/risk_model.pkl"
DATA_PATH = "data/model_data.csv"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
threshold = model_data["threshold"]
features = model_data["features"]

# ============================================================
# LOAD TRAINING DATA
# ============================================================

training_df = pd.read_csv(DATA_PATH)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ AI Risk Manager")

st.write(
    "AI-powered account takeover risk detection and security assessment."
)

st.divider()


# ============================================================
# TABS
# ============================================================

single_tab, batch_tab = st.tabs(
    ["🔎 Single Login", "📊 Batch Analysis"]
)


# ============================================================
# CATEGORY ENCODING
# ============================================================

def encode_category(value, column):

    categories = (
        training_df[column]
        .astype("category")
        .cat.categories
        .tolist()
    )

    if value in categories:
        return categories.index(value)

    return -1


# ============================================================
# BOOLEAN CONVERSION
# ============================================================

def convert_to_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in [
        "true",
        "1",
        "yes",
        "y"
    ]


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(
    round_trip_time,
    ip_address,
    country,
    login_successful,
    is_attack_ip,
    ml_score
):

    risk_points = 0
    reasons = []

    # --------------------------------------------------------
    # Attack IP
    # --------------------------------------------------------

    if is_attack_ip:

        risk_points += 60

        reasons.append(
            "🚨 IP address is flagged as an attack IP"
        )

    # --------------------------------------------------------
    # Failed login
    # --------------------------------------------------------

    if not login_successful:

        risk_points += 15

        reasons.append(
            "⚠️ Login attempt was unsuccessful"
        )

    # --------------------------------------------------------
    # High latency
    # --------------------------------------------------------

    if round_trip_time > 500:

        risk_points += 10

        reasons.append(
            "⚠️ High network round-trip time detected"
        )

    # --------------------------------------------------------
    # Unknown IP
    # --------------------------------------------------------

    if encode_category(
        ip_address,
        "IP Address"
    ) == -1:

        risk_points += 10

        reasons.append(
            "⚠️ IP address was not seen during training"
        )

    # --------------------------------------------------------
    # Unknown country
    # --------------------------------------------------------

    if encode_category(
        country,
        "Country"
    ) == -1:

        risk_points += 5

        reasons.append(
            "⚠️ Country was not seen during training"
        )

    # --------------------------------------------------------
    # Combine scores
    # --------------------------------------------------------

    rule_score = min(
        risk_points,
        100
    )

    final_score = max(
        ml_score,
        rule_score
    )

    # --------------------------------------------------------
    # Final risk level
    # --------------------------------------------------------

    if is_attack_ip:

        risk_level = "HIGH"

    elif final_score >= 50:

        risk_level = "HIGH"

    elif final_score >= 25:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return (
        rule_score,
        final_score,
        risk_level,
        reasons
    )


# ============================================================
# SINGLE LOGIN TAB
# ============================================================

with single_tab:

    st.subheader("Investigate a Login")

    st.write(
        "Enter login information to calculate account takeover risk."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        round_trip_time = st.number_input(
            "Round-Trip Time (ms)",
            min_value=0.0,
            value=100.0
        )

        ip_address = st.text_input(
            "IP Address",
            value="192.168.1.1"
        )

        country = st.text_input(
            "Country",
            value="US"
        )

        region = st.text_input(
            "Region",
            value="California"
        )

        city = st.text_input(
            "City",
            value="San Francisco"
        )

        asn = st.number_input(
            "ASN",
            min_value=0,
            value=60989
        )

    with col2:

        user_agent = st.text_input(
            "User Agent String",
            value="Mozilla/5.0"
        )

        browser = st.text_input(
            "Browser Name and Version",
            value="Chrome 120.0"
        )

        os_name = st.text_input(
            "OS Name and Version",
            value="Windows 10"
        )

        device_type = st.selectbox(
            "Device Type",
            [
                "desktop",
                "mobile",
                "tablet"
            ]
        )

        login_successful = st.selectbox(
            "Login Successful?",
            [False, True]
        )

        is_attack_ip = st.selectbox(
            "Is Attack IP?",
            [False, True]
        )

    st.write("")

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Login",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        input_values = {

            "Round-Trip Time [ms]":
                round_trip_time,

            "IP Address":
                encode_category(
                    ip_address,
                    "IP Address"
                ),

            "Country":
                encode_category(
                    country,
                    "Country"
                ),

            "Region":
                encode_category(
                    region,
                    "Region"
                ),

            "City":
                encode_category(
                    city,
                    "City"
                ),

            "ASN":
                asn,

            "User Agent String":
                encode_category(
                    user_agent,
                    "User Agent String"
                ),

            "Browser Name and Version":
                encode_category(
                    browser,
                    "Browser Name and Version"
                ),

            "OS Name and Version":
                encode_category(
                    os_name,
                    "OS Name and Version"
                ),

            "Device Type":
                encode_category(
                    device_type,
                    "Device Type"
                ),

            "Login Successful":
                int(login_successful),

            "Is Attack IP":
                int(is_attack_ip)
        }

        input_data = pd.DataFrame(
            [input_values]
        )

        input_data = input_data[
            features
        ]

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        probability = model.predict_proba(
            input_data
        )[0][1]

        ml_score = probability * 100

        # ----------------------------------------------------
        # Risk engine
        # ----------------------------------------------------

        (
            rule_score,
            final_score,
            risk_level,
            reasons
        ) = calculate_risk(

            round_trip_time,
            ip_address,
            country,
            login_successful,
            is_attack_ip,
            ml_score
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.divider()

        st.subheader("Risk Assessment")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "ML Risk Score",
                f"{ml_score:.2f}%"
            )

        with metric2:

            st.metric(
                "Rule Score",
                f"{rule_score:.2f}%"
            )

        with metric3:

            st.metric(
                "Final Risk Score",
                f"{final_score:.2f}%"
            )

        st.write(
            f"ML decision threshold: **{threshold:.4f}**"
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        st.subheader("Risk Level")

        if risk_level == "HIGH":

            st.error(
                "🚨 HIGH RISK — Suspicious login detected!"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "⚠️ MEDIUM RISK — Additional verification recommended."
            )

        else:

            st.success(
                "✅ LOW RISK — Login appears normal."
            )

        # ----------------------------------------------------
        # Risk indicators
        # ----------------------------------------------------

        st.subheader("Risk Indicators")

        if reasons:

            for reason in reasons:

                st.write(reason)

        else:

            st.write(
                "No major suspicious indicators detected."
            )

        # ----------------------------------------------------
        # Risk breakdown
        # ----------------------------------------------------

        st.subheader("Risk Score Breakdown")

        breakdown1, breakdown2 = st.columns(2)

        with breakdown1:

            st.metric(
                "Machine Learning",
                f"{ml_score:.2f}%"
            )

        with breakdown2:

            st.metric(
                "Security Rules",
                f"{rule_score:.2f}%"
            )

        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        st.subheader("Recommended Action")

        if risk_level == "HIGH":

            st.error(
                "🔒 Block or challenge the login "
                "and verify the user's identity."
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "🔐 Request additional authentication "
                "such as MFA."
            )

        else:

            st.success(
                "✅ Allow the login and continue monitoring."
            )


# ============================================================
# BATCH ANALYSIS TAB
# ============================================================
# ============================================================
# BATCH ANALYSIS TAB
# ============================================================

with batch_tab:

    st.subheader("📊 Batch Login Analysis")

    st.write(
        "Upload a CSV containing multiple login records "
        "to analyze account takeover risk at scale."
    )

    uploaded_file = st.file_uploader(
        "Upload Login CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(uploaded_file)

            required_columns = [
                "Round-Trip Time [ms]",
                "IP Address",
                "Country",
                "Region",
                "City",
                "ASN",
                "User Agent String",
                "Browser Name and Version",
                "OS Name and Version",
                "Device Type",
                "Login Successful",
                "Is Attack IP"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in batch_df.columns
            ]

            if missing_columns:

                st.error("Missing required columns:")
                st.write(missing_columns)

            else:

                st.success(
                    f"Successfully loaded {len(batch_df):,} login records."
                )

                # ====================================================
                # BATCH ML PREDICTION
                # ====================================================

                model_input = pd.DataFrame()

                for feature in features:

                    if feature == "Round-Trip Time [ms]":

                        model_input[feature] = pd.to_numeric(
                            batch_df[feature],
                            errors="coerce"
                        ).fillna(0)

                    elif feature == "ASN":

                        model_input[feature] = pd.to_numeric(
                            batch_df[feature],
                            errors="coerce"
                        ).fillna(0)

                    elif feature == "Login Successful":

                        model_input[feature] = batch_df[
                            feature
                        ].apply(convert_to_bool).astype(int)

                    elif feature == "Is Attack IP":

                        model_input[feature] = batch_df[
                            feature
                        ].apply(convert_to_bool).astype(int)

                    else:

                        categories = (
                            training_df[feature]
                            .astype("category")
                            .cat.categories
                        )

                        category_map = {
                            value: index
                            for index, value
                            in enumerate(categories)
                        }

                        model_input[feature] = (
                            batch_df[feature]
                            .astype(str)
                            .map(category_map)
                            .fillna(-1)
                            .astype(int)
                        )

                # Ensure exact feature order
                model_input = model_input[features]

                # ====================================================
                # ONE MODEL CALL FOR ENTIRE DATASET
                # ====================================================

                with st.spinner(
                    "Analyzing login activity with the ML model..."
                ):

                    probabilities = model.predict_proba(
                        model_input
                    )[:, 1]

                ml_scores = probabilities * 100

                # ====================================================
                # VECTORIZED SECURITY RULE ENGINE
                # ====================================================

                round_trip_series = pd.to_numeric(
                    batch_df["Round-Trip Time [ms]"],
                    errors="coerce"
                ).fillna(0)

                login_series = batch_df[
                    "Login Successful"
                ].apply(convert_to_bool)

                attack_series = batch_df[
                    "Is Attack IP"
                ].apply(convert_to_bool)

                ip_categories = (
                    training_df["IP Address"]
                    .astype("category")
                    .cat.categories
                )

                country_categories = (
                    training_df["Country"]
                    .astype("category")
                    .cat.categories
                )

                known_ips = set(ip_categories)
                known_countries = set(country_categories)

                unknown_ip = ~batch_df[
                    "IP Address"
                ].astype(str).isin(known_ips)

                unknown_country = ~batch_df[
                    "Country"
                ].astype(str).isin(known_countries)

                # ----------------------------------------------------
                # Rule scores
                # ----------------------------------------------------

                attack_points = attack_series.astype(int) * 60

                failed_points = (
                    (~login_series).astype(int) * 15
                )

                latency_points = (
                    (round_trip_series > 500)
                    .astype(int) * 10
                )

                unknown_ip_points = (
                    unknown_ip.astype(int) * 10
                )

                unknown_country_points = (
                    unknown_country.astype(int) * 5
                )

                rule_scores = (
                    attack_points
                    + failed_points
                    + latency_points
                    + unknown_ip_points
                    + unknown_country_points
                ).clip(upper=100)

                final_scores = pd.Series(
                    ml_scores
                ).combine(
                    rule_scores.reset_index(drop=True),
                    max
                )

                # ====================================================
                # RISK LEVEL
                # ====================================================

                risk_levels = pd.Series(
                    "LOW",
                    index=batch_df.index
                )

                risk_levels[
                    final_scores >= 25
                ] = "MEDIUM"

                risk_levels[
                    final_scores >= 50
                ] = "HIGH"

                risk_levels[
                    attack_series
                ] = "HIGH"

                # ====================================================
                # RISK INDICATORS
                # ====================================================

                def build_indicators(index):

                    reasons = []

                    if attack_series.iloc[index]:
                        reasons.append("🚨 Attack IP")

                    if not login_series.iloc[index]:
                        reasons.append("⚠️ Failed login")

                    if round_trip_series.iloc[index] > 500:
                        reasons.append("⚠️ High network latency")

                    if unknown_ip.iloc[index]:
                        reasons.append("⚠️ Unknown IP")

                    if unknown_country.iloc[index]:
                        reasons.append("⚠️ Unknown country")

                    if not reasons:
                        return "No major indicators"

                    return "; ".join(reasons)

                indicators = [
                    build_indicators(i)
                    for i in range(len(batch_df))
                ]

                # ====================================================
                # RESULTS DATAFRAME
                # ====================================================

                results_df = pd.DataFrame({

                    "Risk Score":
                        final_scores.round(2),

                    "ML Score":
                        pd.Series(ml_scores).round(2),

                    "Rule Score":
                        rule_scores.round(2),

                    "Risk Level":
                        risk_levels,

                    "IP Address":
                        batch_df["IP Address"].astype(str),

                    "Country":
                        batch_df["Country"].astype(str),

                    "Login Successful":
                        login_series,

                    "Attack IP":
                        attack_series,

                    "Risk Indicators":
                        indicators
                })

                # ====================================================
                # SECURITY OVERVIEW
                # ====================================================

                st.divider()

                st.subheader("🛡️ Security Overview")

                total = len(results_df)

                high_count = (
                    results_df["Risk Level"] == "HIGH"
                ).sum()

                medium_count = (
                    results_df["Risk Level"] == "MEDIUM"
                ).sum()

                low_count = (
                    results_df["Risk Level"] == "LOW"
                ).sum()

                average_score = (
                    results_df["Risk Score"].mean()
                )

                metric1, metric2, metric3, metric4 = st.columns(4)

                with metric1:
                    st.metric(
                        "Total Logins",
                        f"{total:,}"
                    )

                with metric2:
                    st.metric(
                        "🔴 High Risk",
                        f"{high_count:,}"
                    )

                with metric3:
                    st.metric(
                        "🟡 Medium Risk",
                        f"{medium_count:,}"
                    )

                with metric4:
                    st.metric(
                        "Average Risk",
                        f"{average_score:.2f}%"
                    )

                # ====================================================
                # RISK DISTRIBUTION
                # ====================================================

                st.divider()

                st.subheader("📊 Risk Distribution")

                distribution = pd.DataFrame({
                    "Risk Level": [
                        "HIGH",
                        "MEDIUM",
                        "LOW"
                    ],
                    "Number of Logins": [
                        high_count,
                        medium_count,
                        low_count
                    ]
                })

                st.bar_chart(
                    distribution.set_index(
                        "Risk Level"
                    )
                )
                # ========================================================
                # ATTACK INTELLIGENCE
                # ========================================================

                st.divider()

                st.subheader("🧠 Attack Intelligence")

                st.write(
                    "Identify major attack patterns and suspicious login behavior."
                )

                # --------------------------------------------------------
                # Attack statistics
                # --------------------------------------------------------

                attack_ip_count = int(
                    results_df["Attack IP"].sum()
                )

                failed_login_count = int(
                    (~results_df["Login Successful"]).sum()
                )

                high_risk_percentage = (
                    high_count / total * 100
                    if total > 0
                    else 0
                )

                attack_ip_percentage = (
                    attack_ip_count / total * 100
                    if total > 0
                    else 0
                )

                failed_login_percentage = (
                    failed_login_count / total * 100
                    if total > 0
                    else 0
                )

                # --------------------------------------------------------
                # Intelligence metrics
                # --------------------------------------------------------

                intel1, intel2, intel3, intel4 = st.columns(4)

                with intel1:

                    st.metric(
                        "🚨 Attack IP Attempts",
                        f"{attack_ip_count:,}"
                    )

                with intel2:

                    st.metric(
                        "❌ Failed Logins",
                        f"{failed_login_count:,}"
                    )

                with intel3:

                    st.metric(
                        "🔴 High-Risk %",
                        f"{high_risk_percentage:.2f}%"
                    )

                with intel4:

                    st.metric(
                        "🌐 Attack IP %",
                        f"{attack_ip_percentage:.2f}%"
                    )

                # --------------------------------------------------------
                # Country analysis
                # --------------------------------------------------------

                st.subheader("🌍 High-Risk Activity by Country")

                high_risk_countries = (
                    results_df[
                        results_df["Risk Level"] == "HIGH"
                    ]
                    .groupby("Country")
                    .size()
                    .sort_values(
                        ascending=False
                    )
                    .head(10)
                )

                if len(high_risk_countries) > 0:

                    st.bar_chart(
                        high_risk_countries
                    )

                else:

                    st.info(
                        "No high-risk country data available."
                    )

                # --------------------------------------------------------
                # Attack IP analysis
                # --------------------------------------------------------

                st.subheader("🚨 Attack IP Activity")

                attack_distribution = pd.DataFrame({

                    "IP Type": [
                        "Attack IP",
                        "Normal IP"
                    ],

                    "Login Attempts": [
                        attack_ip_count,
                        total - attack_ip_count
                    ]

                })

                st.bar_chart(
                    attack_distribution.set_index(
                        "IP Type"
                    )
                )

                # --------------------------------------------------------
                # Failed vs successful logins
                # --------------------------------------------------------

                st.subheader("🔐 Authentication Outcomes")

                authentication_distribution = pd.DataFrame({

                    "Authentication": [
                        "Successful",
                        "Failed"
                    ],

                    "Login Attempts": [
                        total - failed_login_count,
                        failed_login_count
                    ]

                })

                st.bar_chart(
                    authentication_distribution.set_index(
                        "Authentication"
                    )
                )

                # ====================================================
                # HIGH-RISK LOGINS
                # ====================================================

                st.divider()

                st.subheader("🚨 High-Risk Logins")

                high_risk = results_df[
                    results_df["Risk Level"] == "HIGH"
                ].sort_values(
                    "Risk Score",
                    ascending=False
                )

                st.write(
                    f"Detected **{len(high_risk):,} high-risk logins.**"
                )

                if len(high_risk) > 0:

                    st.dataframe(
                        high_risk.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.success(
                        "No high-risk logins detected."
                    )

                # ====================================================
                # MEDIUM-RISK LOGINS
                # ====================================================

                st.divider()

                st.subheader("⚠️ Medium-Risk Logins")

                medium_risk = results_df[
                    results_df["Risk Level"] == "MEDIUM"
                ].sort_values(
                    "Risk Score",
                    ascending=False
                )

                st.write(
                    f"Detected **{len(medium_risk):,} medium-risk logins.**"
                )

                if len(medium_risk) > 0:

                    st.dataframe(
                        medium_risk.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.success(
                        "No medium-risk logins detected."
                    )

                # ====================================================
                # EXPORT REPORT
                # ====================================================

                st.divider()

                st.subheader("📥 Export Security Report")

                csv_data = results_df.to_csv(
                    index=False
                ).encode(
                    "utf-8-sig"
                )
                st.download_button(
                    "⬇️ Download Complete Risk Report",
                    data=csv_data,
                    file_name="risk_analysis_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Unable to process the CSV: {error}"
            )
