"""Reusable Plotly chart components."""

import plotly.express as px
import plotly.graph_objects as go


def hospital_review_bar_chart(hospitals: list[dict]) -> go.Figure:
    """Horizontal bar chart: number of reviews per hospital."""
    names = [h["name"] for h in hospitals]
    counts = [h["review_count"] for h in hospitals]

    fig = px.bar(
        x=counts,
        y=names,
        orientation="h",
        labels={"x": "Number of reviews", "y": ""},
        color=counts,
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=max(250, len(names) * 50),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def source_relevance_chart(sources: list[dict]) -> go.Figure:
    """Bar chart: relevance scores of returned sources."""
    labels = [
        f"{s['hospital_name'][:30]} (v:{s['visit_id']})"
        for s in sources
    ]
    scores = [s["score"] for s in sources]

    fig = px.bar(
        x=scores,
        y=labels,
        orientation="h",
        labels={"x": "Relevance score", "y": ""},
        color=scores,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=max(200, len(labels) * 38),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig
