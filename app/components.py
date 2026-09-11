"""UI helper components and styling for Streamlit application."""
import streamlit as st

def inject_custom_css():
    """Injects custom modern styling for the support agent UI."""
    st.markdown("""
        <style>
        .brand-badge {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 15px;
        }
        .decision-auto {
            background-color: #d4edda;
            color: #155724;
            padding: 10px 18px;
            border-radius: 8px;
            font-weight: bold;
            border-left: 5px solid #28a745;
        }
        .decision-escalate {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px 18px;
            border-radius: 8px;
            font-weight: bold;
            border-left: 5px solid #dc3545;
        }
        .evidence-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }
        </style>
    """, unsafe_allow_html=True)
