import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import build_dataset, load_data, get_available_categories, get_available_stores
from src.forecaster import train_forecast, summarise_forecast
from src.llm_explainer import generate_summary, answer_question
import shap
from src.forecaster import train_forecast, summarise_forecast, compute_shap, summarise_shap

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting Copilot",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Demand Forecasting Copilot")
st.caption("AI-powered demand forecasting for supply chain operations")

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    sales_raw, _, _ = load_data()
    categories = get_available_categories(sales_raw)
    stores = get_available_stores(sales_raw)

    category = st.selectbox("Product Category", categories, index=categories.index("FOODS"))
    store = st.selectbox("Store", stores, index=0)
    max_items = st.slider("Number of Items", min_value=3, max_value=20, value=5)
    horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=28, value=14)

    run = st.button("▶ Run Forecast", type="primary", use_container_width=True)

# ── Session state ─────────────────────────────────────────────
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "narrative" not in st.session_state:
    st.session_state.narrative = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "config" not in st.session_state:
    st.session_state.config = {}

# ── Run forecast ──────────────────────────────────────────────
if run:
    with st.spinner("Loading data and training model..."):
        df = build_dataset(category=category, store=store, max_items=max_items)
        forecast_df, mae, importance, model = train_forecast(df, horizon=horizon)
        summary = summarise_forecast(forecast_df)
        shap_df = compute_shap(model, df)
        shap_summary = summarise_shap(shap_df)

    with st.spinner("Generating AI summary..."):
            narrative = generate_summary(summary, category=category, store=store,
                                        shap_summary=shap_summary)

    st.session_state.forecast_df = forecast_df
    st.session_state.summary = summary
    st.session_state.narrative = narrative
    st.session_state.chat_history = []
    st.session_state.config = {
            "category": category, "store": store, "mae": mae,
            "importance": importance, "shap_summary": shap_summary
        }

# ── Main content ──────────────────────────────────────────────
if st.session_state.forecast_df is not None:
    forecast_df = st.session_state.forecast_df
    summary = st.session_state.summary
    narrative = st.session_state.narrative
    config = st.session_state.config

    # ── AI Summary ───────────────────────────────────────────
    st.subheader("🤖 AI Analysis")
    st.info(narrative)

    # ── Metrics row ──────────────────────────────────────────
    st.subheader("📊 Forecast Overview")
    cols = st.columns(len(summary))
    for col, (item_id, stats) in zip(cols, summary.items()):
        pct = stats["pct_change"]
        if pct > 200:
            delta = ">+200%"
            delta_color = "normal"
        elif pct < -200:
            delta = ">-200%"
            delta_color = "inverse"
        else:
            delta = f"{pct:+.1f}%"
            delta_color = "inverse"

        col.metric(
            label=item_id.replace("_", " "),
            value=f"{stats['predicted_avg_daily']:.2f} units/day",
            delta=delta,
            delta_color=delta_color
        )

    # ── Forecast chart ────────────────────────────────────────
    st.subheader("📈 Forecast vs Actual")
    items = forecast_df["item_id"].unique().tolist()
    selected_item = st.selectbox("Select item", items)

    item_df = forecast_df[forecast_df["item_id"] == selected_item]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=item_df["date"], y=item_df["sales"],
        name="Actual", line=dict(color="#636EFA")
    ))
    fig.add_trace(go.Scatter(
        x=item_df["date"], y=item_df["predicted"],
        name="Predicted", line=dict(color="#EF553B", dash="dash")
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Model info ────────────────────────────────────────────
    with st.expander("🔍 Model Details"):
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{config['mae']:.3f} units")
            col2.dataframe(
                config["importance"].head(8).reset_index()
                .rename(columns={"index": "Feature", 0: "Importance"}),
                hide_index=True,
                use_container_width=True
            )
            # SHAP summary per item
            col3.markdown("**Top SHAP drivers**")
            for item_id, drivers in config["shap_summary"].items():
                lines = ", ".join(
                    f"{f.replace('_', ' ')} {'+' if v > 0 else ''}{v}"
                    for f, v in drivers.items()
                )
                col3.caption(f"**{item_id}:** {lines}")
    # ── Q&A ───────────────────────────────────────────────────
    st.subheader("💬 Ask the Copilot")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("e.g. Which item should I restock first?"):
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Get and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass history excluding last user message
                history_for_api = st.session_state.chat_history[:-1]
                answer = answer_question(
                    question=question,
                    summary=summary,
                    category=config["category"],
                    store=config["store"],
                    history=history_for_api if history_for_api else None
                )
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Configure your forecast in the sidebar and click **Run Forecast** to begin.")


