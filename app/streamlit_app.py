"""Streamlit Interactive Application for Hiver AI Support Agent."""
import os
import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.components import inject_custom_css
from src.agent.agent import SupportAgent
from src.utils.config import load_config
from src.utils.io import load_json

st.set_page_config(
    page_title="Hiver AI Support Agent - Apple Support",
    page_icon="🍏",
    layout="wide"
)

inject_custom_css()

@st.cache_resource
def get_agent():
    return SupportAgent.from_config()

agent = get_agent()

st.title("🍏 Apple Support AI Agent & Trustworthy RAG System")
st.markdown('<div class="brand-badge">Selected Brand: Apple Support (@AppleSupport)</div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Interactive Support Agent", "📊 Evaluation Benchmarks", "📖 System Architecture & Guardrails"])

with tabs[0]:
    st.markdown("### Test Customer Queries")
    
    SAMPLE_QUERIES = {
        "Custom Query": "",
        "[Battery Drain] iPhone 7 battery dropping 1% every 2 mins on iOS 11": "My iPhone 7 battery is draining 1% every two minutes since updating to iOS 11. Can you help?",
        "[Hardware Repair] Cracked back glass on iPhone 8 repair cost": "I dropped my iPhone 8 and the glass back shattered. How much does out-of-warranty repair cost?",
        "[High-Risk Security] Apple ID hacked & unauthorized Russian login": "I got an email saying my Apple ID was logged into from Moscow, Russia. I am in Ohio! Someone hacked my account!",
        "[Billing Issue] Unrecognized iTunes.com/bill charge of $9.99": "I was charged $9.99 on my credit card from 'ITUNES.COM/BILL' and don't know what it is for.",
        "[Network / SIM] 'No SIM Card Installed' error": "My iPhone suddenly says 'No SIM Card Installed' even though the SIM is inside."
    }

    selected_sample = st.selectbox("Choose a sample customer query or type your own below:", list(SAMPLE_QUERIES.keys()))
    
    default_text = SAMPLE_QUERIES[selected_sample]
    user_input = st.text_area("Customer Tweet / Message:", value=default_text, height=90)

    if st.button("Generate Grounded Support Reply", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a customer message.")
        else:
            with st.spinner("Classifying intent, retrieving historical precedent, and applying safety policies..."):
                output = agent.process_message(user_input)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🎯 Classification & Decision")
                st.markdown(f"**Predicted Intent**: `{output.intent}`")
                st.markdown(f"**Composite Confidence**: `{output.confidence:.2%}`")
                st.progress(float(output.confidence))

                if output.decision == "auto":
                    st.markdown(f'<div class="decision-auto">✅ ROUTING: AUTO-HANDLE<br><small>{output.reason}</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="decision-escalate">🚨 ROUTING: ESCALATE TO HUMAN<br><small>{output.reason}</small></div>', unsafe_allow_html=True)

                st.subheader("✍️ Grounded Support Reply")
                st.info(output.reply)

            with col2:
                st.subheader("📚 Grounding Evidence (Historical Precedents)")
                if output.evidence:
                    for i, doc in enumerate(output.evidence, 1):
                        with st.expander(f"Precedent #{i}: {doc.intent} (Score: {doc.relevance_score:.2f})", expanded=(i==1)):
                            st.markdown(f"**Customer Issue**: {doc.customer_issue}")
                            st.markdown(f"**Historical Brand Resolution**: {doc.brand_resolution}")
                            st.caption(f"Conversation ID: `{doc.conversation_id}`")
                else:
                    st.warning("No historical evidence retrieved.")

with tabs[1]:
    st.header("📊 Empirical Evaluation Results (200 Golden Examples)")
    metrics_path = ROOT_DIR / "evaluation" / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Intent Macro-F1", f"{metrics['summary']['intent_macro_f1']:.4f}")
        m_col2.metric("Auto-Handle Precision", f"{metrics['escalation_routing']['auto_precision']:.2%}")
        m_col3.metric("Escalate Recall", f"{metrics['escalation_routing']['escalate_recall']:.2%}")
        m_col4.metric("Avg Reply Quality", f"{metrics['summary']['avg_quality_score']:.2f} / 5.0")

        st.subheader("Model Benchmark Comparison")
        st.table({
            "Model": ["Baseline 1: Majority Class", "Baseline 2: TF-IDF + Logistic", "Final System: Semantic Classifier"],
            "Accuracy": [0.1000, 0.6050, 0.6000],
            "Macro Precision": [0.0100, 0.7011, 0.6211],
            "Macro Recall": [0.1000, 0.6050, 0.6000],
            "Macro F1": [0.0182, 0.6236, 0.6070]
        })

        st.subheader("Reply Quality Rubric (LLM-as-a-Judge 1-5)")
        st.json(metrics["reply_quality_rubric"])

        st.subheader("Human vs LLM Judge Agreement (50 Samples)")
        if "human_judge_agreement" in metrics and "overall_aggregate" in metrics["human_judge_agreement"]:
            st.json(metrics["human_judge_agreement"]["overall_aggregate"])
    else:
        st.info("Run `python scripts/run_evaluation.py` to generate real metrics.")

with tabs[2]:
    st.header("🛡️ Safety Policies & Architecture Guardrails")
    st.markdown("""
    - **Zero Irreversible Actions**: No automated refund disbursement, credit card changes, or account credential resets.
    - **PII Masking**: Emails, phone numbers, and case references are sanitized via regex gateway before LLM inference.
    - **Conservative Escalation**: Any query triggering security compromise, legal threats, hardware battery swelling, or low confidence ($<65\%$) is safely routed to human support teams.
    - **Grounded Retrieval**: All drafted responses cite historical precedents from 5,000 verified `@AppleSupport` resolutions.
    """)
