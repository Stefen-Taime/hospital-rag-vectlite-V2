"""Chat component: history + input + response display."""

import streamlit as st

from chatbot_front.components.charts import source_relevance_chart
from chatbot_front.services.api_client import chat

TOOL_LABELS = {
    "reviews": ":mag: Search in patient reviews",
    "structured_data": ":bar_chart: Query on structured data",
}


def _tool_badge(tool_used: str) -> str:
    parts = [t.strip() for t in tool_used.split(",")]
    labels = [TOOL_LABELS.get(t, t) for t in parts]
    return " + ".join(labels)


def render_chat():
    """Display the RAG chat interface."""
    st.title("RAG — Montreal Hospital Reviews")
    st.caption("Multi-tool agent: semantic search in reviews + structured queries")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Your question about hospitals..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("The agent is analyzing your question..."):
                try:
                    result = chat(question)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    tool_used = result.get("tool_used", "")

                    if tool_used:
                        st.caption(_tool_badge(tool_used))

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources and relevance", expanded=False):
                            fig = source_relevance_chart(sources)
                            st.plotly_chart(fig, use_container_width=True)

                            for i, s in enumerate(sources, 1):
                                st.markdown(
                                    f"**{i}. {s['hospital_name']}** "
                                    f"(score: {s['score']:.4f})\n\n"
                                    f"> {s['review_excerpt']}..."
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                except Exception as exc:
                    st.error(f"Error: {exc}")
