"""Streamlit Web Dashboard for Customer Support Ticket Intelligence."""

import streamlit as st
import pandas as pd
from src.inference.predict import InferenceEngine

st.set_page_config(
    page_title="Ticket Intelligence Dashboard",
    page_icon="🎫",
    layout="wide"
)

st.title("🎫 Customer Support Ticket Intelligence Dashboard")
st.markdown("Analyze, classify, and intelligently route incoming customer support tickets in real-time.")

engine = InferenceEngine()

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Single Ticket Classification", "Batch Ticket Analytics"])

if page == "Single Ticket Classification":
    st.subheader("Real-Time Ticket Routing & Intelligence")
    
    col1, col2 = st.columns(2)
    with col1:
        ticket_id = st.text_input("Ticket ID", "TCK-8492")
        subject = st.text_input("Subject", "Payment failed during renewal")
    with col2:
        customer_tier = st.selectbox("Customer Tier", ["Standard", "Pro", "Enterprise"])
        
    description = st.text_area("Ticket Description", "I tried renewing my annual subscription but the credit card transaction failed with error code ERR_PAYMENT_DECLINED.")
    
    if st.button("Analyze Ticket"):
        full_text = f"{subject} {description}"
        res = engine.predict_ticket(full_text)
        
        st.success("Analysis Complete!")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Category", res["predicted_category"])
        m2.metric("Urgency Level", res["priority"])
        m3.metric("Confidence Score", f"{res['confidence_score'] * 100:.1f}%")
        
        st.json(res)

elif page == "Batch Ticket Analytics":
    st.subheader("Overview Metrics & Ticket Trends")
    st.info("Upload a batch CSV file to process multiple tickets simultaneously.")
    uploaded_file = st.file_uploader("Upload Ticket CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
