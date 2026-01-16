import textwrap
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH_DEFAULT = "C:\\Users\\Vijay Patidar\\Documents\\varahe\\assignment_JJM\\census-2011\\india-districts-census-2011.csv"


def _pretty(col: str) -> str:
    """Human-friendly label from a column name."""
    # Keep some common abbreviations readable
    col = col.replace("LPG_or_PNG", "LPG/PNG")
    col = col.replace("Housholds", "Households")
    col = col.replace("_", " ")
    col = col.replace("  ", " ").strip()
    # Title-case most words but preserve some acronyms
    keep_upper = {"SC", "ST", "LPG", "PNG", "TV", "RC", "OS", "NRH", "NSA", "AOM"}
    words = []
    for w in col.split():
        if w.upper() in keep_upper:
            words.append(w.upper())
        elif any(ch.isdigit() for ch in w):
            words.append(w)
        else:
            words.append(w[:1].upper() + w[1:].lower())
    return " ".join(words)


@st.cache_data(show_spinner=False)
def load_district_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names gently
    df.columns = [c.strip() for c in df.columns]

    # Ensure key columns exist
    required = {"State name", "District name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert numeric columns
    non_num = {"District code", "State name", "District name"}
    for c in df.columns:
        if c in non_num:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0)

    # Derived metrics (rates)
    def safe_rate(numer, denom):
        denom = denom.replace(0, np.nan)
        return (numer / denom * 100.0).fillna(0.0)

    if all(c in df.columns for c in ["Literate", "Population"]):
        df["Literacy_Rate_Total"] = safe_rate(df["Literate"], df["Population"])
    if all(c in df.columns for c in ["Male_Literate", "Male"]):
        df["Literacy_Rate_Male"] = safe_rate(df["Male_Literate"], df["Male"])
    if all(c in df.columns for c in ["Female_Literate", "Female"]):
        df["Literacy_Rate_Female"] = safe_rate(df["Female_Literate"], df["Female"])
    if all(c in df.columns for c in ["Female", "Male"]):
        df["Sex_Ratio_(F_per_1000_M)"] = safe_rate(df["Female"], df["Male"]) * 10

    if all(c in df.columns for c in ["Urban_Households", "Households"]):
        df["Urban_Households_Share_%"] = safe_rate(df["Urban_Households"], df["Households"])
    if all(c in df.columns for c in ["Rural_Households", "Households"]):
        df["Rural_Households_Share_%"] = safe_rate(df["Rural_Households"], df["Households"])

    return df


def aggregate_state(df_state: pd.DataFrame) -> pd.Series:
    """Aggregate all numeric columns across districts (state total)."""
    non_num = ["District code", "State name", "District name"]
    numeric_cols = [c for c in df_state.columns if c not in non_num and pd.api.types.is_numeric_dtype(df_state[c])]
    agg = df_state[numeric_cols].sum(axis=0)
    agg["District name"] = "(State total)"
    agg["State name"] = df_state["State name"].iloc[0] if len(df_state) else ""
    agg["District code"] = 0

    # Recompute derived rates on totals
    def safe_rate(numer, denom):
        return 0.0 if denom == 0 else float(numer) / float(denom) * 100.0

    if "Literate" in agg.index and "Population" in agg.index:
        agg["Literacy_Rate_Total"] = safe_rate(agg["Literate"], agg["Population"])
    if "Male_Literate" in agg.index and "Male" in agg.index:
        agg["Literacy_Rate_Male"] = safe_rate(agg["Male_Literate"], agg["Male"])
    if "Female_Literate" in agg.index and "Female" in agg.index:
        agg["Literacy_Rate_Female"] = safe_rate(agg["Female_Literate"], agg["Female"])
    if "Female" in agg.index and "Male" in agg.index:
        agg["Sex_Ratio_(F_per_1000_M)"] = safe_rate(agg["Female"], agg["Male"]) * 10
    if "Urban_Households" in agg.index and "Households" in agg.index:
        agg["Urban_Households_Share_%"] = safe_rate(agg["Urban_Households"], agg["Households"])
    if "Rural_Households" in agg.index and "Households" in agg.index:
        agg["Rural_Households_Share_%"] = safe_rate(agg["Rural_Households"], agg["Households"])

    return agg


def kpi_row(record: pd.Series) -> None:
    def fmt_int(x):
        try:
            return f"{int(round(float(x))):,}"
        except Exception:
            return "-"

    def fmt_pct(x):
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return "-"

    def fmt_ratio(x):
        try:
            return f"{float(x):.0f}"
        except Exception:
            return "-"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population", fmt_int(record.get("Population", 0)))
    c2.metric("Male", fmt_int(record.get("Male", 0)))
    c3.metric("Female", fmt_int(record.get("Female", 0)))
    c4.metric("Sex ratio (F per 1000 M)", fmt_ratio(record.get("Sex_Ratio_(F_per_1000_M)", 0)))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Literate", fmt_int(record.get("Literate", 0)))
    c6.metric("Literacy rate (total)", fmt_pct(record.get("Literacy_Rate_Total", 0)))
    c7.metric("Literacy rate (male)", fmt_pct(record.get("Literacy_Rate_Male", 0)))
    c8.metric("Literacy rate (female)", fmt_pct(record.get("Literacy_Rate_Female", 0)))


def bar_compare_one(record: pd.Series, cols: List[str], denom: Optional[str] = None, title: str = ""):
    vals = []
    labels = []
    for c in cols:
        if c not in record.index:
            continue
        v = float(record[c])
        if denom and denom in record.index and float(record[denom]) != 0:
            v = v / float(record[denom]) * 100.0
            labels.append(f"{_pretty(c)} (% of {_pretty(denom)})")
        else:
            labels.append(_pretty(c))
        vals.append(v)
    if not vals:
        st.info("No matching columns found for this chart.")
        return

    plot_df = pd.DataFrame({"Metric": labels, "Value": vals})
    fig = px.bar(plot_df, x="Metric", y="Value", title=title or None)
    fig.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def ranked_table(df_state: pd.DataFrame, metric: str, topn: int = 10) -> pd.DataFrame:
    out = df_state[["District name", metric]].copy()
    out = out.sort_values(metric, ascending=False).head(topn)
    out[metric] = out[metric].astype(float)
    return out


def main():
    st.set_page_config(page_title="Census 2011 Dashboard (India)", layout="wide")

    st.title("Census 2011 Dashboard (State → District)")
    st.caption(
        "Explore district-wise Census 2011 indicators. Use the sidebar to pick a State and District, then browse the tabs."
    )

    with st.sidebar:
        st.header("Filters")
        data_path = st.text_input("CSV path", value=DATA_PATH_DEFAULT)
        st.caption("Tip: keep the default if you uploaded the provided file set.")

    try:
        df = load_district_data(data_path)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    with st.sidebar:
        states = sorted(df["State name"].unique().tolist())
        state = st.selectbox("State", states, index=0)
        df_state = df[df["State name"] == state].copy()
        districts = sorted(df_state["District name"].unique().tolist())
        district = st.selectbox("District", ["All districts (state total)"] + districts, index=0)

        st.divider()
        st.subheader("Quick compare")
        metric_choices = [c for c in df.columns if c not in {"District code", "State name", "District name"}]
        default_metric = "Population" if "Population" in metric_choices else metric_choices[0]
        rank_metric = st.selectbox("Rank districts by", metric_choices, index=metric_choices.index(default_metric))
        topn = st.slider("Top N", min_value=5, max_value=25, value=10, step=1)

    if district == "All districts (state total)":
        record = aggregate_state(df_state)
    else:
        record = df_state[df_state["District name"] == district].iloc[0]

    # Header
    h1, h2 = st.columns([3, 2])
    with h1:
        st.subheader(f"{record.get('State name', state)} · {record.get('District name', district)}")
    with h2:
        # Show which dataset
        st.write("")
        st.caption("Data source: india-districts-census-2011.csv")

    kpi_row(record)

    # Ranking table in the main area (useful when district is selected too)
    st.markdown("#### District ranking (within selected state)")
    rank_df = ranked_table(df_state, rank_metric, topn=topn)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Demographics", "Literacy & Education", "Housing & Amenities", "Water & Sanitation", "Custom metrics", "Data & download"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            bar_compare_one(
                record,
                cols=["Male", "Female"],
                title="Male vs Female (counts)",
            )
        with c2:
            age_cols = [c for c in ["Age_Group_0_29", "Age_Group_30_49", "Age_Group_50", "Age not stated"] if c in record.index]
            bar_compare_one(record, cols=age_cols, denom="Population", title="Age groups (% of population)")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            bar_compare_one(record, cols=["Literate_Education", "Illiterate_Education"], denom=None, title="Literate vs Illiterate (education table)")
        with c2:
            edu_cols = [
                "Below_Primary_Education",
                "Primary_Education",
                "Middle_Education",
                "Secondary_Education",
                "Higher_Education",
                "Graduate_Education",
                "Other_Education",
            ]
            edu_cols = [c for c in edu_cols if c in record.index]
            bar_compare_one(record, cols=edu_cols, denom="Total_Education" if "Total_Education" in record.index else None, title="Education levels")

        st.markdown("---")
        st.subheader("Literacy rates")
        rate_df = pd.DataFrame(
            {
                "Category": ["Total", "Male", "Female"],
                "Rate (%)": [
                    float(record.get("Literacy_Rate_Total", 0)),
                    float(record.get("Literacy_Rate_Male", 0)),
                    float(record.get("Literacy_Rate_Female", 0)),
                ],
            }
        )
        fig = px.bar(rate_df, x="Category", y="Rate (%)")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            amen_cols = [
                "LPG_or_PNG_Households",
                "Housholds_with_Electric_Lighting",
                "Households_with_Internet",
                "Households_with_Computer",
            ]
            amen_cols = [c for c in amen_cols if c in record.index]
            bar_compare_one(record, cols=amen_cols, denom="Households" if "Households" in record.index else None, title="Amenities (% of households)")
        with c2:
            # Rural/Urban split
            ru_cols = [c for c in ["Rural_Households", "Urban_Households"] if c in record.index]
            bar_compare_one(record, cols=ru_cols, denom="Households" if "Households" in record.index else None, title="Rural vs Urban (% of households)")

        st.markdown("---")
        st.subheader("Assets")
        asset_cols = [
            "Households_with_Bicycle",
            "Households_with_Scooter_Motorcycle_Moped",
            "Households_with_Car_Jeep_Van",
            "Households_with_Television",
            "Households_with_Telephone_Mobile_Phone",
        ]
        asset_cols = [c for c in asset_cols if c in record.index]
        if asset_cols:
            bar_compare_one(record, cols=asset_cols, denom="Households" if "Households" in record.index else None, title="Assets (% of households)")
        else:
            st.info("Asset columns not present in this dataset.")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            san_cols = [
                "Having_bathing_facility_Total_Households",
                "Not_having_bathing_facility_within_the_premises_Total_Households",
                "Having_latrine_facility_within_the_premises_Total_Households",
                "Not_having_latrine_facility_within_the_premises_Alternative_source_Open_Households",
            ]
            san_cols = [c for c in san_cols if c in record.index]
            bar_compare_one(record, cols=san_cols, denom="Households" if "Households" in record.index else None, title="Sanitation basics (% of households)")

        with c2:
            water_cols = [
                "Main_source_of_drinking_water_Tapwater_Households",
                "Main_source_of_drinking_water_Handpump_Tubewell_Borewell_Households",
                "Main_source_of_drinking_water_Un_covered_well_Households",
                "Main_source_of_drinking_water_Tank_Pond_Lake_Households",
                "Main_source_of_drinking_water_River_Canal_Households",
                "Main_source_of_drinking_water_Spring_Households",
                "Main_source_of_drinking_water_Other_sources_Households",
            ]
            water_cols = [c for c in water_cols if c in record.index]
            bar_compare_one(record, cols=water_cols, denom="Households" if "Households" in record.index else None, title="Drinking water sources (% of households)")

        st.markdown("---")
        st.subheader("Latrine & bathing details")
        detail_cols = [
            "Type_of_bathing_facility_Enclosure_without_roof_Households",
            "Type_of_latrine_facility_Pit_latrine_Households",
            "Type_of_latrine_facility_Other_latrine_Households",
            "Type_of_latrine_facility_Night_soil_disposed_into_open_drain_Households",
            "Type_of_latrine_facility_Flush_pour_flush_latrine_connected_to_other_system_Households",
        ]
        detail_cols = [c for c in detail_cols if c in record.index]
        bar_compare_one(record, cols=detail_cols, denom="Households" if "Households" in record.index else None, title="Facilities (% of households)")

    with tab5:
        st.subheader("Pick any indicators to visualize")
        non_num = {"District code", "State name", "District name"}
        numeric_cols = [c for c in df.columns if c not in non_num and pd.api.types.is_numeric_dtype(df[c])]

        left, right = st.columns([2, 1])
        with left:
            chosen = st.multiselect(
                "Indicators",
                options=sorted(numeric_cols, key=lambda x: x.lower()),
                default=[c for c in ["Population", "LPG_or_PNG_Households", "Housholds_with_Electric_Lighting"] if c in numeric_cols],
            )
        with right:
            denom = st.selectbox(
                "Show as",
                options=["Raw counts", "% of population", "% of households"],
                index=0,
            )

        denom_col = None
        if denom == "% of population":
            denom_col = "Population" if "Population" in record.index else None
        elif denom == "% of households":
            denom_col = "Households" if "Households" in record.index else None

        if chosen:
            bar_compare_one(record, cols=chosen, denom=denom_col, title="Selected indicators")
        else:
            st.info("Select one or more indicators to see a chart.")

        st.markdown("---")
        st.subheader("State-wide view")
        st.caption("Plot the selected indicator across all districts in the chosen state.")
        metric_for_districts = st.selectbox("Indicator (district comparison)", options=sorted(numeric_cols), index=sorted(numeric_cols).index("Population") if "Population" in numeric_cols else 0)
        plot_state = df_state[["District name", metric_for_districts]].sort_values(metric_for_districts, ascending=False)
        fig = px.bar(plot_state, x="District name", y=metric_for_districts)
        fig.update_layout(xaxis_title=None, yaxis_title=_pretty(metric_for_districts), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("Raw row (selected district or state total)")
        row_df = pd.DataFrame(record).T
        st.dataframe(row_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Download filtered data")
        dl_choice = st.radio("Download", options=["All districts in selected state", "Only selected district/state total"], horizontal=True)
        if dl_choice == "All districts in selected state":
            dl_df = df_state.copy()
        else:
            dl_df = row_df.copy()

        csv_bytes = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name=f"census2011_{state.replace(' ','_')}.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Notes")
        st.write(
            textwrap.dedent(
                """
                - This dashboard sums district rows to create a **state total** view (when you choose “All districts”).
                - Rates (literacy, household shares) are computed from the displayed totals.
                """
            ).strip()
        )


if __name__ == "__main__":
    main()
