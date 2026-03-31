"""Sidebar component: health check + stats dashboard."""

import streamlit as st

from chatbot_front.components.charts import hospital_review_bar_chart
from chatbot_front.services.api_client import get_stats, health_check


def render_sidebar():
    """Display the sidebar with the stats dashboard."""
    with st.sidebar:
        st.header("Dashboard")

        try:
            health = health_check()
            st.success(f"API connected — {health['vectordb_size']} indexed documents")
        except Exception:
            st.error("API unavailable. Start `chatbot_api` first.")
            st.stop()

        st.divider()

        try:
            stats = get_stats()
            st.metric("Total indexed reviews", stats["total_reviews"])
            st.subheader("Reviews per hospital")
            fig = hospital_review_bar_chart(stats["hospitals"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Unable to load statistics")
