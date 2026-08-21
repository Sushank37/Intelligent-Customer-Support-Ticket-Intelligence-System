"""Professional Streamlit dashboard for Customer Support Ticket Intelligence."""

import streamlit as st

from src.prediction.predictor import TicketPredictor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ticket Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0b0f14;
    }

    /* Main content width */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #11161d;
        border-right: 1px solid #252c36;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    /* Header */
    .hero {
        padding: 10px 0 20px 0;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1.5px;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #9aa4b2;
        font-size: 17px;
        margin-bottom: 0;
    }

    /* Section headings */
    .section-title {
        font-size: 23px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Prediction cards */
    .prediction-card {
        background: #11161d;
        border: 1px solid #252c36;
        border-radius: 16px;
        padding: 22px;
        min-height: 180px;
    }

    .prediction-label {
        color: #8f9baa;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .prediction-value {
        font-size: 27px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 16px;
    }

    .confidence-label {
        color: #8f9baa;
        font-size: 13px;
    }

    /* Priority badges */
    .priority-high {
        color: #ff6b6b;
    }

    .priority-medium {
        color: #ffb454;
    }

    .priority-low {
        color: #53d88a;
    }

    /* Ticket preview */
    .ticket-box {
        background: #11161d;
        border: 1px solid #252c36;
        border-radius: 14px;
        padding: 18px;
        color: #c7ced8;
        line-height: 1.6;
        margin-top: 10px;
    }

    /* Status */
    .status-box {
        background: #10251b;
        border: 1px solid #1d5135;
        border-radius: 10px;
        padding: 10px 14px;
        color: #63e49a;
        font-size: 14px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #687383;
        font-size: 13px;
        padding-top: 25px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 48px;
    }

    /* Divider */
    hr {
        border-color: #252c36;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_predictor():
    return TicketPredictor()


with st.spinner("Loading AI models..."):
    predictor = load_predictor()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:24px;font-weight:700;">
        🎫 Ticket Intelligence
        </div>
        <div style="color:#8993a1;margin-top:5px;">
        AI Support Operations
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Ticket Context")

    language = st.selectbox(
        "Language",
        ["en", "de", "es", "fr", "pt"],
        index=0,
    )

    business_type = st.text_input(
        "Business Type",
        value="IT_Services",
    )

    st.markdown("### Ticket Tags")

    tags = {}

    for i in range(1, 10):
        tags[f"tag_{i}"] = st.text_input(
            f"Tag {i}",
            value="",
            key=f"tag_{i}",
        )

    st.divider()

    st.markdown(
        """
        <div class="status-box">
        ● AI Models Online
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Type • Queue • Priority")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Customer Support Ticket Intelligence
        </div>
        <div class="hero-subtitle">
            Automatically classify, prioritize and route customer
            support tickets using multilingual AI.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# TICKET INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Analyze a Support Ticket</div>',
    unsafe_allow_html=True,
)

ticket_text = st.text_area(
    "Customer message",
    height=190,
    placeholder=(
        "Example:\n"
        "Our server has been experiencing frequent overloads "
        "and slow performance. Please help us resolve this issue "
        "as soon as possible."
    ),
    label_visibility="collapsed",
)


analyze = st.button(
    "🔍  Analyze Ticket",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not ticket_text.strip():
        st.warning("Please enter a customer support ticket.")
        st.stop()

    ticket = {
        "ticket_text": ticket_text,
        "language": language,
        "business_type": business_type,
    }

    ticket.update(tags)

    with st.spinner("AI is analyzing the ticket..."):

        try:
            result = predictor.predict(ticket)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">AI Classification</div>',
        unsafe_allow_html=True,
    )

    type_result = result["type"]
    queue_result = result["queue"]
    priority_result = result["priority"]

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # TYPE
    # --------------------------------------------------------

    with col1:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Ticket Type
                </div>
                <div class="prediction-value">
                    {type_result["label"]}
                </div>
                <div class="confidence-label">
                    Confidence: {type_result["confidence"] * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            min(max(type_result["confidence"], 0.0), 1.0)
        )

    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    with col2:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Support Queue
                </div>
                <div class="prediction-value">
                    {queue_result["label"]}
                </div>
                <div class="confidence-label">
                    Confidence: {queue_result["confidence"] * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            min(max(queue_result["confidence"], 0.0), 1.0)
        )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    priority_label = priority_result["label"].lower()

    if priority_label == "high":
        priority_class = "priority-high"
        priority_icon = "🔴"

    elif priority_label == "medium":
        priority_class = "priority-medium"
        priority_icon = "🟠"

    else:
        priority_class = "priority-low"
        priority_icon = "🟢"

    with col3:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Priority
                </div>
                <div class="prediction-value {priority_class}">
                    {priority_icon} {priority_label.upper()}
                </div>
                <div class="confidence-label">
                    Confidence: {priority_result["confidence"] * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            min(max(priority_result["confidence"], 0.0), 1.0)
        )


    # ========================================================
    # TICKET SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Ticket Summary</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="ticket-box">
            {ticket_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # ROUTING SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Recommended Routing</div>',
        unsafe_allow_html=True,
    )

    routing_col1, routing_col2 = st.columns(2)

    with routing_col1:

        st.info(
            f"📥 Route this ticket to **{queue_result['label']}**"
        )

    with routing_col2:

        if priority_label == "high":
            st.error(
                "🚨 High-priority ticket — immediate attention recommended."
            )

        elif priority_label == "medium":
            st.warning(
                "⚠️ Medium-priority ticket — standard attention required."
            )

        else:
            st.success(
                "✅ Low-priority ticket — standard processing is appropriate."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer Support Ticket Intelligence
        &nbsp;•&nbsp;
        Multilingual AI Classification
        &nbsp;•&nbsp;
        Type + Queue + Priority
    </div>
    """,
    unsafe_allow_html=True,
)