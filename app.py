"""Streamlit runtime for the Network Intrusion Detection System.

Loads pre-trained artifacts from models/. Does not train.
Run:  streamlit run app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_HELP,
    LABEL_NAMES,
    PROTO_OPTIONS,
    SERVICE_OPTIONS,
    STATE_OPTIONS,
)
from src.predict import ModelNotTrainedError, load_artifacts, predict_frame
from src.preprocessing import default_row, row_to_frame

st.set_page_config(
    page_title="NIDS | Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1c2541 0%, #0b132b 55%);
        color: #e0e1dd;
    }
    h1, h2, h3 { color: #ffffff !important; }
    .hero {
        border: 1px solid #3a506b;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(28,37,65,0.95) 0%, rgba(11,19,43,0.95) 100%);
        margin-bottom: 1rem;
    }
    .hero .eyebrow {
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #6fffe9;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .metric-note { color: #5bc0be; font-size: 0.85rem; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid #3a506b;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    .result-ok, .result-alert {
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        font-size: 1.15rem;
        font-weight: 650;
        margin: 0.6rem 0 1rem 0;
    }
    .result-ok {
        background: rgba(91, 192, 190, 0.12);
        border: 1px solid #5bc0be;
        color: #6fffe9;
    }
    .result-alert {
        background: rgba(255, 82, 82, 0.12);
        border: 1px solid #ff6b6b;
        color: #ffb4b4;
    }
</style>
"""


def _render_prediction_banner(label: int, probability: float) -> None:
    name = LABEL_NAMES.get(int(label), str(label))
    pct = probability * 100.0
    if int(label) == 1:
        st.markdown(
            f'<div class="result-alert">Attack detected — model confidence {pct:.1f}%</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-ok">Normal traffic — attack probability {pct:.1f}%</div>',
            unsafe_allow_html=True,
        )


def _option_with_current(options: tuple[str, ...], current: str) -> list[str]:
    values = list(options)
    if current not in values:
        values.insert(0, current)
    return values


@st.cache_resource
def _cached_artifacts():
    return load_artifacts()


def render_sidebar(metadata: dict) -> None:
    st.sidebar.markdown("### Model")
    st.sidebar.write(metadata.get("model", "GradientBoostingClassifier"))
    st.sidebar.caption(metadata.get("class_weight", "balanced"))
    source = metadata.get("data_source", "unknown")
    st.sidebar.markdown("### Training data")
    st.sidebar.write("UNSW-NB15" if source == "unsw-nb15" else "Synthetic stand-in (run train.py with official CSVs for production metrics)")
    st.sidebar.markdown("### Hold-out metrics")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Recall", f"{float(metadata.get('recall', 0)) * 100:.1f}%")
    c2.metric("Accuracy", f"{float(metadata.get('accuracy', 0)) * 100:.1f}%")
    c3, c4 = st.sidebar.columns(2)
    c3.metric("Precision", f"{float(metadata.get('precision', 0)) * 100:.1f}%")
    c4.metric("F1", f"{float(metadata.get('f1', 0)) * 100:.1f}%")
    st.sidebar.caption("Binary labels: 0 = Normal, 1 = Attack")
    st.sidebar.divider()
    st.sidebar.markdown(
        "Retrain with `python train.py` after placing "
        "`UNSW_NB15_training-set.csv` in the `data/` folder."
    )


def single_input_tab() -> None:
    st.subheader("Single flow prediction")
    st.caption("Enter one network flow. Unused services should be set to `-`.")
    defaults = default_row()

    with st.form("single_flow_form"):
        c1, c2, c3, c4 = st.columns(4)
        proto = c1.selectbox(
            "proto",
            _option_with_current(PROTO_OPTIONS, str(defaults["proto"])),
            help=FEATURE_HELP["proto"],
        )
        service = c2.selectbox(
            "service",
            _option_with_current(SERVICE_OPTIONS, str(defaults["service"])),
            help=FEATURE_HELP["service"],
        )
        state = c3.selectbox(
            "state",
            _option_with_current(STATE_OPTIONS, str(defaults["state"])),
            help=FEATURE_HELP["state"],
        )
        dur = c4.number_input("dur", min_value=0.0, value=float(defaults["dur"]), format="%.6f", help=FEATURE_HELP["dur"])

        values: dict[str, object] = {"proto": proto, "service": service, "state": state, "dur": dur}

        with st.expander("Packet, byte, and rate features", expanded=True):
            cols = st.columns(4)
            packet_fields = (
                "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
                "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
            )
            for i, name in enumerate(packet_fields):
                values[name] = cols[i % 4].number_input(
                    name,
                    value=float(defaults[name]),
                    help=FEATURE_HELP[name],
                )

        with st.expander("TCP window, RTT, and payload"):
            cols = st.columns(4)
            tcp_fields = (
                "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
                "smean", "dmean", "trans_depth", "response_body_len",
            )
            for i, name in enumerate(tcp_fields):
                values[name] = cols[i % 4].number_input(
                    name,
                    value=float(defaults[name]),
                    help=FEATURE_HELP[name],
                )

        with st.expander("Connection counters and flags"):
            cols = st.columns(4)
            count_fields = (
                "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
                "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
                "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
            )
            for i, name in enumerate(count_fields):
                values[name] = cols[i % 4].number_input(
                    name,
                    value=float(defaults[name]),
                    help=FEATURE_HELP[name],
                )

        submitted = st.form_submit_button("Classify flow", type="primary", use_container_width=True)

    if not submitted:
        return

    try:
        result = predict_frame(row_to_frame(values))
    except Exception as exc:
        st.error(str(exc))
        return

    row = result.iloc[0]
    _render_prediction_banner(int(row["prediction"]), float(row["attack_probability"]))
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", str(row["prediction_label"]))
    m2.metric("Attack probability", f"{float(row['attack_probability']) * 100:.1f}%")
    m3.metric("Encoded label", int(row["prediction"]))
    st.dataframe(result, use_container_width=True)


def csv_upload_tab() -> None:
    st.subheader("Batch prediction from CSV")
    st.caption(
        "Upload a CSV with UNSW-NB15 feature columns. Extra columns such as "
        "`id` and `attack_cat` are ignored. A `label` column is optional and is not used for scoring."
    )

    uploaded = st.file_uploader("Traffic CSV", type=["csv"])
    if uploaded is None:
        st.download_button(
            "Download sample CSV",
            data=_sample_csv_bytes(),
            file_name="sample_traffic.csv",
            mime="text/csv",
        )
        return

    try:
        frame = pd.read_csv(uploaded)
        result = predict_frame(frame)
    except Exception as exc:
        st.error(str(exc))
        return

    n = len(result)
    n_attack = int((result["prediction"] == 1).sum())
    n_normal = n - n_attack
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows", n)
    c2.metric("Attacks", n_attack)
    c3.metric("Normal", n_normal)
    c4.metric("Attack rate", f"{(n_attack / n * 100) if n else 0:.1f}%")

    st.dataframe(result, use_container_width=True, height=420)
    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions",
        data=csv_bytes,
        file_name="nids_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


@st.cache_data
def _sample_csv_bytes() -> bytes:
    from pathlib import Path

    sample_path = Path(__file__).resolve().parent / "data" / "sample_traffic.csv"
    if sample_path.exists():
        return sample_path.read_bytes()
    defaults = default_row()
    attack = dict(defaults)
    attack.update({"rate": 2200.0, "sttl": 64, "spkts": 30, "dbytes": 0, "service": "-", "state": "INT"})
    buf = io.StringIO()
    pd.DataFrame([defaults, attack])[FEATURE_COLUMNS].to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Cybersecurity &amp; machine learning</div>
            <h1>Network Intrusion Detection System</h1>
            <p>Classify a single flow or a batch of UNSW-NB15-style records as <b>Normal</b> or <b>Attack</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        _model, _preprocessor, metadata = _cached_artifacts()
    except ModelNotTrainedError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(metadata)
    tab_single, tab_batch = st.tabs(["Single input", "CSV upload"])
    with tab_single:
        single_input_tab()
    with tab_batch:
        csv_upload_tab()


if __name__ == "__main__":
    main()
