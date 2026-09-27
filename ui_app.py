"""
ui_app.py
===============================================================================
Prototype Predictive Campus Intelligence System: Clean Executive Dashboard
Theme: Professional White / Light Executive Academic Theme
Target Audience: 2nd-Year Computer Science Undergraduates
Contributor: Student 3 (Frontend Architecture & Interactive Visualizations)

Modules Covered:
    - Module 6: Streamlit Web Development, Plotly 2D Spatial Blueprint Maps,
                Event-Driven Interaction, and Mitigation Recommendations
===============================================================================
"""

from datetime import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# 1. IMPORT BACKEND FUNCTIONS FROM core_logic.py
# =============================================================================
from core_logic import (
    CAMPUS_ZONES,
    ZONE_METADATA,
    CampusPredictor,
    enrich_zones_with_map,
    filter_congested_zones,
    generate_campus_dataset,
    get_rerouting_recommendations,
    load_model_from_disk,
    save_model_to_disk,
    validate_time_format,
)

# =============================================================================
# 2. STREAMLIT CACHING (PREVENTS SLOW RE-EXECUTION)
# =============================================================================

@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Loads and caches the 30-day simulated dataset from Student 1."""
    return generate_campus_dataset(num_days=30)


@st.cache_resource
def load_trained_ai_model(df: pd.DataFrame) -> CampusPredictor:
    """
    Checks for a serialized model on disk (Pickle - Module 5).
    If absent, trains a new model and serializes it to disk.
    """
    cached_model = load_model_from_disk()
    if cached_model is not None:
        return cached_model

    predictor = CampusPredictor()
    predictor.train(df)
    save_model_to_disk(predictor)
    return predictor


# =============================================================================
# 3. SPATIAL BLUEPRINT VISUALIZATION (WHITE THEME PLOTLY 2D MAP)
# =============================================================================

def build_campus_spatial_map(enriched_zones: list):
    """
    Renders an interactive spatial map of campus facilities on a clean white blueprint.
    Dots are color-coded (🟢 Normal, 🟡 Moderate, 🔴 High Congestion) with dynamic sizing.
    """
    map_rows = []
    for z in enriched_zones:
        meta = ZONE_METADATA[z["zone"]]
        coord_x, coord_y = meta["coords"]  # Module 1: Tuple unpacking
        
        map_rows.append({
            "Zone": z["zone"],
            "X": coord_x,
            "Y": coord_y,
            "Facility Type": meta["type"],
            "Capacity": f"{meta['capacity']} Seats",
            "Current Load (%)": z["current_occupancy"],
            "Forecast +30m (%)": z["predicted_occupancy"],
            "Status": z["risk_tier"],
            "Marker Size": 24 + (z["predicted_occupancy"] * 0.3)
        })

    df_map = pd.DataFrame(map_rows)

    fig = px.scatter(
        df_map,
        x="X",
        y="Y",
        text="Zone",
        size="Marker Size",
        color="Status",
        color_discrete_map={
            "NORMAL": "#10B981",          # Clean Emerald Green
            "MODERATE": "#F59E0B",        # Amber Warning
            "HIGH_CONGESTION": "#EF4444", # Crisp Alert Red
        },
        hover_data={
            "X": False,
            "Y": False,
            "Marker Size": False,
            "Facility Type": True,
            "Capacity": True,
            "Current Load (%)": ":.1f",
            "Forecast +30m (%)": ":.1f",
        },
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=12, color="#0F172A", family="Arial Black, Inter, sans-serif"),
        marker=dict(line=dict(width=2, color="#1E293B")),
    )

    fig.update_layout(
        title="<b>Campus Spatial Density Map (+30 Min Horizon)</b>",
        title_font=dict(size=15, color="#0F172A", family="Inter, sans-serif"),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        xaxis=dict(
            title="West-East Coordinate Axis (Meters)",
            range=[0, 100],
            showgrid=True,
            gridcolor="#E2E8F0",
            tickfont=dict(color="#475569"),
            title_font=dict(color="#334155"),
        ),
        yaxis=dict(
            title="South-North Coordinate Axis (Meters)",
            range=[0, 100],
            showgrid=True,
            gridcolor="#E2E8F0",
            tickfont=dict(color="#475569"),
            title_font=dict(color="#334155"),
        ),
        height=470,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#1E293B"),
        ),
    )
    return fig


# =============================================================================
# 4. MAIN STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Campus Occupancy Intelligence System",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Clean, Professional Light / White Theme Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
        
        /* Global Light Background */
        .stApp {
            background-color: #F8FAFC;
            color: #0F172A;
            font-family: 'Inter', sans-serif;
        }
        
        /* Clean White Header Banner */
        .white-header {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            padding: 22px 28px;
            border-radius: 10px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        .white-header h1 {
            color: #0F172A;
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .white-header p {
            color: #64748B;
            margin: 6px 0 0 0;
            font-size: 0.95rem;
        }
        .header-chip {
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            color: #1D4ED8;
            font-family: 'JetBrains Mono', monospace;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
            margin-bottom: 8px;
        }
        
        /* White Theme Cards */
        .white-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
            margin-bottom: 12px;
        }
        
        /* Mitigation Engine Alert Box */
        .mitigation-box {
            background-color: #FEF2F2;
            border-left: 4px solid #EF4444;
            border-top: 1px solid #FEE2E2;
            border-right: 1px solid #FEE2E2;
            border-bottom: 1px solid #FEE2E2;
            border-radius: 6px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Executive Header
    st.markdown(
        """
        <div class="white-header">
            <span class="header-chip">Academic Prototype &bull; Group of 3</span>
            <h1>🏛️ Predictive Campus Occupancy Intelligence System</h1>
            <p>Historical Foot-Traffic Modeling for Short-Term (+30 min) Congestion Forecasting & Proactive Mitigation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 1: Load Data & Model from Backend
    df_data = load_historical_data()
    model_pipeline = load_trained_ai_model(df_data)

    # Step 2: Sidebar Controls
    st.sidebar.markdown("### 🎛️ Simulation Parameters")
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_day = st.sidebar.selectbox("Day of the Week", day_names, index=1)
    day_index = day_names.index(selected_day)

    selected_time = st.sidebar.time_input("Simulation Clock (HH:MM)", value=time(13, 0))
    sim_hour = selected_time.hour
    sim_minute = selected_time.minute

    # Module 6: Regex format verification
    time_str = f"{sim_hour:02d}:{sim_minute:02d}"
    if not validate_time_format(time_str):
        st.sidebar.error("Invalid time format detected via Regex.")

    # Preset Scenarios
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Scenario Presets")
    scenario_choice = st.sidebar.radio(
        "Load Test Condition:",
        [
            "Custom Manual Sliders",
            "🥪 Lunch Hour Rush (13:00)",
            "🔔 Lecture Changeover (11:00)",
            "📚 Exam Study Peak (15:30)",
        ],
    )

    zone_inputs = {}

    if scenario_choice == "🥪 Lunch Hour Rush (13:00)":
        sim_hour, sim_minute = 13, 0
        zone_inputs = {
            "Canteen": 88.0, "Cafeteria B": 42.0, "Library": 79.0, "Labs": 61.0,
            "Lifts": 85.0, "Corridors": 68.0, "Auditorium": 35.0, "Parking": 54.0
        }
    elif scenario_choice == "🔔 Lecture Changeover (11:00)":
        sim_hour, sim_minute = 11, 0
        zone_inputs = {
            "Canteen": 32.0, "Cafeteria B": 20.0, "Library": 42.0, "Labs": 30.0,
            "Lifts": 88.0, "Corridors": 94.0, "Auditorium": 18.0, "Parking": 58.0
        }
    elif scenario_choice == "📚 Exam Study Peak (15:30)":
        sim_hour, sim_minute = 15, 30
        zone_inputs = {
            "Canteen": 35.0, "Cafeteria B": 28.0, "Library": 90.0, "Labs": 42.0,
            "Lifts": 38.0, "Corridors": 32.0, "Auditorium": 15.0, "Parking": 45.0
        }
    else:
        for z in CAMPUS_ZONES:
            zone_inputs[z] = 50.0

    # Fine-tuning sliders
    st.sidebar.markdown("---")
    with st.sidebar.expander("Fine-Tune Current Zone Loads (%)"):
        for z in CAMPUS_ZONES:
            zone_inputs[z] = st.slider(
                f"{z}",
                min_value=0.0,
                max_value=100.0,
                value=float(zone_inputs.get(z, 50.0)),
                step=1.0,
            )

    # Step 3: Run Model Inference on All Zones
    raw_predictions = []
    for zone in CAMPUS_ZONES:
        curr = zone_inputs[zone]
        pred = model_pipeline.predict(
            zone=zone,
            hour=sim_hour,
            minute=sim_minute,
            day_of_week=day_index,
            current_occ=curr,
        )
        raw_predictions.append({
            "zone": zone,
            "current_occupancy": curr,
            "predicted_occupancy": round(pred, 1),
        })

    # Step 4: Functional Python Transformations (Module 2)
    enriched_zones = enrich_zones_with_map(raw_predictions)
    congested_zones = filter_congested_zones(enriched_zones)
    reroute_plans = get_rerouting_recommendations(enriched_zones)

    # Step 5: Top KPI Metric Cards
    st.markdown("### 📊 Executive Campus Telemetry Overview")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    avg_current = np.mean([z["current_occupancy"] for z in enriched_zones])
    avg_pred = np.mean([z["predicted_occupancy"] for z in enriched_zones])
    delta_avg = avg_pred - avg_current

    kpi_col1.metric("Simulated Clock", f"{selected_day[:3]} {sim_hour:02d}:{sim_minute:02d}")
    kpi_col2.metric("Campus Mean Load", f"{avg_current:.1f}%")
    kpi_col3.metric("30-Min Forecast Mean", f"{avg_pred:.1f}%", f"{delta_avg:+.1f}% vs current", delta_color="inverse")
    kpi_col4.metric(
        "Critical Zones (≥75%)",
        f"{len(congested_zones)} / {len(CAMPUS_ZONES)}",
        delta="Action Needed" if congested_zones else "Nominal Flow",
        delta_color="inverse",
    )

    st.markdown("---")

    # Step 6: Map & Recommender Layout
    col_map, col_recom = st.columns([3, 2])

    with col_map:
        st.markdown("### 🗺️ Real-Time Spatial Heatmap")
        spatial_fig = build_campus_spatial_map(enriched_zones)
        st.plotly_chart(spatial_fig, use_container_width=True)

    with col_recom:
        st.markdown("### ⚡ Proactive Mitigation Engine")
        if reroute_plans:
            st.warning(f"🚨 **{len(reroute_plans)} Potential Bottleneck(s) Forecasted (+30 Mins)**")
            for plan in reroute_plans:
                st.markdown(
                    f"""
                    <div class="mitigation-box">
                        <strong style="color:#991B1B;">HOTSPOT: {plan['congested_zone']} ({plan['congested_pred']:.1f}% forecast)</strong><br>
                        <span style="color:#1E293B;">➡️ <b>Suggested Alternative:</b> <strong style="color:#047857;">{plan['suggested_zone']} ({plan['suggested_pred']:.1f}% load)</strong></span><br>
                        <span style="color:#1D4ED8;">💡 <b>Relief Headroom:</b> <strong>+{plan['relief_headroom']:.1f}% spare capacity</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ **Campus Traffic Flow Balanced**")
            st.info("No rerouting interventions or space diversions required for the next 30-minute window.")

    st.markdown("---")

    # Step 7: Zone Telemetry Cards
    st.markdown("### 🏢 Detailed Facility Telemetry Grid")
    card_cols = st.columns(4)

    for idx, zone_item in enumerate(enriched_zones):
        with card_cols[idx % 4]:
            with st.container(border=True):
                st.markdown(f"**{zone_item['zone']}**")
                st.caption(f"{zone_item['type']} &bull; Capacity: {zone_item['capacity']} seats")
                st.write(f"Status: **{zone_item['badge']}**")

                diff = round(zone_item["predicted_occupancy"] - zone_item["current_occupancy"], 1)
                st.metric(
                    label="+30 Min Projection",
                    value=f"{zone_item['predicted_occupancy']:.1f}%",
                    delta=f"{diff:+.1f}% vs current",
                )

                # Modern Visual Progress Bar & Capacity Telemetry (No odd ASCII characters)
                pred_pct = float(np.clip(zone_item["predicted_occupancy"], 0.0, 100.0))
                curr_pct = float(np.clip(zone_item["current_occupancy"], 0.0, 100.0))
                risk_color = "#EF4444" if zone_item["risk_tier"] == "HIGH_CONGESTION" else ("#F59E0B" if zone_item["risk_tier"] == "MODERATE" else "#10B981")
                est_seats = int(round((pred_pct / 100.0) * zone_item["capacity"]))

                st.markdown(
                    f"""
                    <div style="margin-top: 10px; margin-bottom: 4px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; font-weight: 600; margin-bottom: 5px;">
                            <span>Projected Load</span>
                            <span style="color: {risk_color}; font-weight: 700;">{pred_pct:.1f}%</span>
                        </div>
                        <div style="background-color: #E2E8F0; border-radius: 999px; height: 8px; width: 100%; overflow: hidden;">
                            <div style="background-color: {risk_color}; width: {pred_pct}%; height: 100%; border-radius: 999px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748B; margin-top: 7px; padding-top: 4px; border-top: 1px dashed #E2E8F0;">
                            <span>Current: <strong style="color: #1E293B;">{curr_pct:.1f}%</strong></span>
                            <span>Seats: <strong style="color: #1E293B;">{est_seats}/{zone_item['capacity']}</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Step 8: Academic Inspection Tabs
    st.markdown("---")
    st.markdown("### 🔬 Academic Verification & Diagnostics")
    tab_eval, tab_func, tab_data = st.tabs([
        "📈 Scikit-Learn Model Evaluation (Student 2)",
        "⚙️ Functional Python (`map`/`filter`) (Student 1)",
        "📋 Historical Training Dataset Sample (Student 1)",
    ])

    with tab_eval:
        st.markdown("#### Random Forest Evaluation per Zone")
        metrics_dict = {
            z: metrics.to_dict()
            for z, metrics in model_pipeline.metrics_container.items()
        }
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        st.dataframe(metrics_df.style.highlight_min(subset=["MAE (Error %)"], color="#D1FAE5"), use_container_width=True)
        st.caption("• **MAE (Mean Absolute Error)**: Average prediction error on unseen holdout test split.")
        st.caption("• **R2 Score**: Coefficient of determination measuring model accuracy.")

    with tab_func:
        st.markdown("#### Functional Programming Pipeline (`map()` & `filter()`)")
        st.caption("Demonstrating Module 2 functional transformations using anonymous `lambda` functions.")

        st.markdown("##### 1. `map()` Transformation Output (All 8 Zones)")
        st.markdown(
            """
            ```python
            # Code from core_logic.py:
            enrich_zones_with_map = list(map(lambda z: {**z, 'risk_tier': classify_risk_tier(z['predicted_occupancy'])}, zones_list))
            ```
            """
        )
        map_display_df = pd.DataFrame(enriched_zones)[["zone", "type", "capacity", "current_occupancy", "predicted_occupancy", "risk_tier"]]
        st.dataframe(map_display_df, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 2. `filter()` Interactive Filter Demonstration")
        st.markdown(
            """
            ```python
            # Code from core_logic.py:
            filtered_zones = list(filter(lambda z: z['predicted_occupancy'] >= threshold, zones_list))
            ```
            """
        )

        test_thresh = st.slider("Test Custom Filter Cutoff Threshold (%):", min_value=20.0, max_value=90.0, value=75.0, step=5.0)
        custom_filtered = list(filter(lambda z: z["predicted_occupancy"] >= test_thresh, enriched_zones))

        if custom_filtered:
            st.warning(f"🚨 **{len(custom_filtered)} zone(s) isolated with predicted load ≥ {test_thresh:.0f}%:**")
            st.json(custom_filtered)
        else:
            st.info(f"✅ **No zones currently exceed {test_thresh:.0f}% occupancy.** (All campus zones are operating within nominal flow at this simulated time).")
            st.caption("💡 *Tip for Presentation: Select 'Lunch Rush (13:00)' from the sidebar presets to demonstrate active bottleneck filtering.*")

    with tab_data:
        st.markdown("#### Historical Synthetic Dataset Preview")
        st.dataframe(df_data.head(10), use_container_width=True)
        st.caption(f"Total historical training records: {len(df_data):,} rows.")


if __name__ == "__main__":
    main()
