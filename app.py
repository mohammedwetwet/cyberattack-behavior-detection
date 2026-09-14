import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cyberattack Behavior Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background-color: #0b0f14;
        color: #e8edf2;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }


    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #080c11;
        border-right: 1px solid #1d2733;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }


    /* -------------------------------------------------------
       TYPOGRAPHY
    ------------------------------------------------------- */

    h1 {
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
        color: #f4f7fa;
        margin-bottom: 0.4rem;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 600 !important;
        color: #f4f7fa;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #dce3ea;
    }

    p {
        color: #9ba8b5;
    }


    /* -------------------------------------------------------
       DESCRIPTION
    ------------------------------------------------------- */

    .app-description {
        color: #8e9aa7;
        font-size: 1rem;
        margin-bottom: 2rem;
        max-width: 760px;
        line-height: 1.6;
    }


    /* -------------------------------------------------------
       TOP HEADER
    ------------------------------------------------------- */

    .top-line {
        height: 1px;
        background: #1d2733;
        margin: 1.5rem 0 2rem 0;
    }


    /* -------------------------------------------------------
       SECTION CARD
    ------------------------------------------------------- */

    .section-card {
        background: #10161d;
        border: 1px solid #1d2733;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }


    /* -------------------------------------------------------
       BUTTONS
    ------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #2a3644;
        background: #151d26;
        color: #e9eef3;
        font-weight: 600;
        min-height: 44px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #526477;
        background: #1b2530;
        color: #ffffff;
    }


    /* Primary Detect Button */

    div[data-testid="stButton"] button[kind="primary"] {
        background: #e8edf2;
        color: #0b0f14;
        border: none;
        font-weight: 700;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #ffffff;
        color: #0b0f14;
    }


    /* -------------------------------------------------------
       INPUTS
    ------------------------------------------------------- */

    div[data-baseweb="input"] > div {
        background-color: #0e141b;
        border-color: #263341;
        border-radius: 7px;
    }

    div[data-baseweb="select"] > div {
        background-color: #0e141b;
        border-color: #263341;
        border-radius: 7px;
    }

    input {
        color: #edf2f6 !important;
    }


    /* -------------------------------------------------------
       METRICS
    ------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: #10161d;
        border: 1px solid #1d2733;
        border-radius: 10px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #8996a3;
    }

    div[data-testid="stMetricValue"] {
        color: #f1f5f8;
    }


    /* -------------------------------------------------------
       PROGRESS
    ------------------------------------------------------- */

    div[data-testid="stProgress"] > div {
        background-color: #1a232d;
    }


    /* -------------------------------------------------------
       EXPANDERS
    ------------------------------------------------------- */

    div[data-testid="stExpander"] {
        background: #10161d;
        border: 1px solid #1d2733;
        border-radius: 10px;
    }


    /* -------------------------------------------------------
       SIDEBAR INFO
    ------------------------------------------------------- */

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f0f4f7;
        margin-bottom: 1.2rem;
    }

    .sidebar-label {
        font-size: 0.75rem;
        color: #71808f;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1.4rem;
        margin-bottom: 0.35rem;
    }

    .sidebar-value {
        color: #d9e0e6;
        font-size: 0.9rem;
    }


    /* -------------------------------------------------------
       RESULT
    ------------------------------------------------------- */

    .result-normal {
        background: #101a17;
        border: 1px solid #29443a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .result-attack {
        background: #1b1113;
        border: 1px solid #4a292e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .result-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8996a3;
        margin-bottom: 0.4rem;
    }

    .result-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f4f7fa;
    }


    /* -------------------------------------------------------
       FEATURE COUNT
    ------------------------------------------------------- */

    .feature-count {
        color: #71808f;
        font-size: 0.85rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }


    /* -------------------------------------------------------
       HIDE STREAMLIT BRANDING
    ------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_cyberattack_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"
DEMO_PATH = BASE_DIR / "demo_samples.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    return joblib.load(FEATURES_PATH)


@st.cache_data
def load_metadata():

    if METADATA_PATH.exists():
        return joblib.load(METADATA_PATH)

    return {
        "model_name": "Cyberattack Detection Model",
        "target": "label",
        "classes": {
            0: "Normal",
            1: "Attack"
        }
    }


@st.cache_data
def load_demo_samples():

    if DEMO_PATH.exists():
        return joblib.load(DEMO_PATH)

    return None


# ============================================================
# SAFE LOADING
# ============================================================

try:

    model = load_model()
    feature_names = load_features()
    metadata = load_metadata()
    demo_samples = load_demo_samples()

except Exception as e:

    st.error("Failed to load the model files.")

    st.code(str(e))

    st.stop()


feature_names = [
    str(feature)
    for feature in feature_names
]


# ============================================================
# MODEL NAME
# ============================================================

model_name = metadata.get(
    "model_name",
    "Cyberattack Detection Model"
)


# ============================================================
# SESSION STATE
# ============================================================

if "current_example" not in st.session_state:
    st.session_state.current_example = None


if "example_counter" not in st.session_state:
    st.session_state.example_counter = 0


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


if "demo_sequence" not in st.session_state:

    sequence = (
        ["normal"] * 5 +
        ["attack"] * 5
    )

    rng = np.random.default_rng()

    rng.shuffle(sequence)

    st.session_state.demo_sequence = sequence


# ============================================================
# FEATURE GROUPS
# ============================================================

categorical_features = []
binary_features = []
numeric_features = []


for feature in feature_names:

    if (
        feature.startswith("proto_")
        or feature.startswith("service_")
        or feature.startswith("state_")
    ):
        categorical_features.append(feature)

    elif feature in [
        "is_ftp_login",
        "is_sm_ips_ports"
    ]:
        binary_features.append(feature)

    else:
        numeric_features.append(feature)


# ============================================================
# CATEGORICAL GROUPS
# ============================================================

protocol_features = [
    f for f in categorical_features
    if f.startswith("proto_")
]

service_features = [
    f for f in categorical_features
    if f.startswith("service_")
]

state_features = [
    f for f in categorical_features
    if f.startswith("state_")
]


# ============================================================
# HELPER: UPDATE WIDGET STATES
# ============================================================

def update_widget_states(example):

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for feature in numeric_features:

        value = example.get(
            feature,
            0.0
        )

        try:
            value = float(value)
        except Exception:
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        st.session_state[
            f"numeric_{feature}"
        ] = value


    # --------------------------------------------------------
    # Binary
    # --------------------------------------------------------

    for feature in binary_features:

        st.session_state[
            f"binary_{feature}"
        ] = bool(
            example.get(
                feature,
                0
            )
        )


    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    def selected_category(
        features,
        prefix
    ):

        selected = "None"

        for feature in features:

            value = example.get(
                feature,
                0
            )

            if bool(value):

                selected = feature.replace(
                    prefix,
                    "",
                    1
                )

                break

        return selected


    st.session_state["protocol_select"] = selected_category(
        protocol_features,
        "proto_"
    )

    st.session_state["service_select"] = selected_category(
        service_features,
        "service_"
    )

    st.session_state["state_select"] = selected_category(
        state_features,
        "state_"
    )


# ============================================================
# GENERATE NEW EXAMPLE
# ============================================================

def generate_new_example():

    # ========================================================
    # REAL TEST DATA
    # ========================================================

    if demo_samples is not None:

        # Start a new 10-example balanced sequence
        if st.session_state.example_counter >= 10:

            sequence = (
                ["normal"] * 5 +
                ["attack"] * 5
            )

            rng = np.random.default_rng()

            rng.shuffle(sequence)

            st.session_state.demo_sequence = sequence
            st.session_state.example_counter = 0


        sample_type = st.session_state.demo_sequence[
            st.session_state.example_counter
        ]


        samples = demo_samples.get(
            sample_type
        )


        if samples is None or len(samples) == 0:

            st.error(
                f"No {sample_type} samples are available."
            )

            return


        sample = samples.sample(
            n=1
        ).iloc[0]


        example = sample.to_dict()


        cleaned_example = {}

        for feature in feature_names:

            cleaned_example[feature] = example.get(
                feature,
                0
            )


        st.session_state.current_example = cleaned_example


        # Update actual widget states
        update_widget_states(
            cleaned_example
        )


        st.session_state.example_counter += 1

        # Clear previous result
        st.session_state.prediction_result = None

        return


    # ========================================================
    # FALLBACK
    # ========================================================

    example = {}

    for feature in feature_names:

        example[feature] = 0


    default_values = {

        "dur": 1.2,
        "spkts": 10,
        "dpkts": 8,
        "sbytes": 800,
        "dbytes": 1200,
        "rate": 50,
        "sttl": 64,
        "dttl": 64,
        "sload": 1000,
        "dload": 1500,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 10,
        "dinpkt": 10,
        "sjit": 0,
        "djit": 0,
        "swin": 8192,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 8192,
        "tcprtt": 0,
        "synack": 0,
        "ackdat": 0,
        "smean": 80,
        "dmean": 100,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 1,
        "ct_state_ttl": 1,
        "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 1,
        "ct_src_ltm": 1,
        "ct_srv_dst": 1
    }


    for feature, value in default_values.items():

        if feature in example:
            example[feature] = value


    st.session_state.current_example = example

    update_widget_states(
        example
    )

    st.session_state.prediction_result = None


# ============================================================
# FIRST EXAMPLE
# ============================================================

if st.session_state.current_example is None:

    generate_new_example()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Cyberattack Detection</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "Generate New Example",
        use_container_width=True
    ):

        generate_new_example()

        st.rerun()


    st.markdown(
        '<div class="sidebar-label">Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sidebar-value">{model_name}</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-label">Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sidebar-value">{len(feature_names)}</div>',
        unsafe_allow_html=True
    )


    if demo_samples is not None:

        st.markdown(
            '<div class="sidebar-label">Demo</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-value">UNSW-NB15 test samples</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "Cyberattack Behavior Detection"
)

st.markdown(
    """
    <div class="app-description">
    Analyze network traffic and classify its behavior as normal or malicious.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="top-line"></div>',
    unsafe_allow_html=True
)


# ============================================================
# TRAFFIC FEATURES
# ============================================================

st.header(
    "Traffic Features"
)

st.markdown(
    f'<div class="feature-count">{len(numeric_features)} numerical features</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# NUMERICAL FEATURES
# ------------------------------------------------------------

# Important common features shown first
priority_features = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "smean",
    "dmean"
]


priority_existing = [
    f for f in priority_features
    if f in numeric_features
]


remaining_numeric = [
    f for f in numeric_features
    if f not in priority_existing
]


# ============================================================
# PRIMARY FEATURES
# ============================================================

if priority_existing:

    cols = st.columns(3)

    for i, feature in enumerate(priority_existing):

        col = cols[i % 3]

        if f"numeric_{feature}" not in st.session_state:

            value = st.session_state.current_example.get(
                feature,
                0.0
            )

            try:
                value = float(value)
            except Exception:
                value = 0.0

            if not np.isfinite(value):
                value = 0.0

            st.session_state[
                f"numeric_{feature}"
            ] = value


        col.number_input(
            feature,
            key=f"numeric_{feature}"
        )


# ============================================================
# ADVANCED NUMERICAL FEATURES
# ============================================================

if remaining_numeric:

    with st.expander(
        "Advanced numerical features"
    ):

        cols = st.columns(3)

        for i, feature in enumerate(remaining_numeric):

            col = cols[i % 3]


            if f"numeric_{feature}" not in st.session_state:

                value = st.session_state.current_example.get(
                    feature,
                    0.0
                )

                try:
                    value = float(value)
                except Exception:
                    value = 0.0

                if not np.isfinite(value):
                    value = 0.0

                st.session_state[
                    f"numeric_{feature}"
                ] = value


            col.number_input(
                feature,
                key=f"numeric_{feature}"
            )


# ============================================================
# CONNECTION FEATURES
# ============================================================

if binary_features:

    with st.expander(
        "Connection features"
    ):

        cols = st.columns(3)

        for i, feature in enumerate(binary_features):

            col = cols[i % 3]


            if f"binary_{feature}" not in st.session_state:

                st.session_state[
                    f"binary_{feature}"
                ] = bool(
                    st.session_state.current_example.get(
                        feature,
                        0
                    )
                )


            col.checkbox(
                feature,
                key=f"binary_{feature}"
            )


# ============================================================
# PROTOCOL / SERVICE / STATE
# ============================================================

st.header(
    "Network Configuration"
)


def category_options(features, prefix):

    options = ["None"]

    for feature in features:

        name = feature.replace(
            prefix,
            "",
            1
        )

        options.append(name)

    return options


# ------------------------------------------------------------
# Protocol
# ------------------------------------------------------------

if protocol_features:

    protocol_options = category_options(
        protocol_features,
        "proto_"
    )

    if "protocol_select" not in st.session_state:

        st.session_state["protocol_select"] = "None"

    protocol_col, service_col, state_col = st.columns(3)


    with protocol_col:

        st.selectbox(
            "Protocol",
            protocol_options,
            key="protocol_select"
        )


# ------------------------------------------------------------
# Service
# ------------------------------------------------------------

if service_features:

    service_options = category_options(
        service_features,
        "service_"
    )

    if "service_select" not in st.session_state:

        st.session_state["service_select"] = "None"


    with service_col:

        st.selectbox(
            "Service",
            service_options,
            key="service_select"
        )


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

if state_features:

    state_options = category_options(
        state_features,
        "state_"
    )

    if "state_select" not in st.session_state:

        st.session_state["state_select"] = "None"


    with state_col:

        st.selectbox(
            "Connection State",
            state_options,
            key="state_select"
        )


# ============================================================
# BUILD INPUT
# ============================================================

def build_input():

    input_data = {}


    # --------------------------------------------------------
    # Numerical
    # --------------------------------------------------------

    for feature in numeric_features:

        value = st.session_state.get(
            f"numeric_{feature}",
            0.0
        )

        try:
            value = float(value)
        except Exception:
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        input_data[feature] = value


    # --------------------------------------------------------
    # Binary
    # --------------------------------------------------------

    for feature in binary_features:

        input_data[feature] = int(
            st.session_state.get(
                f"binary_{feature}",
                False
            )
        )


    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    for feature in categorical_features:

        input_data[feature] = 0


    # Protocol
    selected_protocol = st.session_state.get(
        "protocol_select",
        "None"
    )

    if selected_protocol != "None":

        feature = (
            "proto_" +
            selected_protocol
        )

        if feature in input_data:
            input_data[feature] = 1


    # Service
    selected_service = st.session_state.get(
        "service_select",
        "None"
    )

    if selected_service != "None":

        feature = (
            "service_" +
            selected_service
        )

        if feature in input_data:
            input_data[feature] = 1


    # State
    selected_state = st.session_state.get(
        "state_select",
        "None"
    )

    if selected_state != "None":

        feature = (
            "state_" +
            selected_state
        )

        if feature in input_data:
            input_data[feature] = 1


    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )


    # Exact model feature order
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )


    return input_df


# ============================================================
# DETECTION
# ============================================================

st.markdown(
    '<div class="top-line"></div>',
    unsafe_allow_html=True
)


detect = st.button(
    "Detect Network Behavior",
    type="primary",
    use_container_width=True
)


if detect:

    try:

        input_df = build_input()


        prediction = model.predict(
            input_df
        )[0]


        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_df
            )[0]


        st.session_state.prediction_result = {
            "prediction": int(prediction),
            "probabilities": probabilities,
            "input": input_df
        }


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(e)
        )


# ============================================================
# RESULT
# ============================================================

result = st.session_state.prediction_result


if result is not None:

    prediction = result["prediction"]

    probabilities = result["probabilities"]

    input_df = result["input"]


    # ========================================================
    # RESULT HEADER
    # ========================================================

    if prediction == 1:

        st.markdown(
            """
            <div class="result-attack">
                <div class="result-label">Classification</div>
                <div class="result-title">Cyberattack Detected</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-normal">
                <div class="result-label">Classification</div>
                <div class="result-title">Normal Network Behavior</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    if probabilities is not None:

        st.subheader(
            "Prediction Probability"
        )


        normal_probability = float(
            probabilities[0]
        )

        attack_probability = float(
            probabilities[1]
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Normal",
                f"{normal_probability * 100:.1f}%"
            )

            st.progress(
                normal_probability
            )


        with col2:

            st.metric(
                "Cyberattack",
                f"{attack_probability * 100:.1f}%"
            )

            st.progress(
                attack_probability
            )


    # ========================================================
    # INPUT DETAILS
    # ========================================================

    with st.expander(
        "View submitted features"
    ):

        display_df = input_df.T.reset_index()

        display_df.columns = [
            "Feature",
            "Value"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )