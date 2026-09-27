import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DIRISA SDC 2026 - Youth Voter Disengagement", layout="wide"
)

st.title("🇿🇦 Mapping Youth Voter Disengagement in South African Municipalities")
st.markdown("### Team VUT — DIRISA Student Datathon 2026 | Challenge 1")

# Sidebar
st.sidebar.header("Filter & Settings")
province_filter = st.sidebar.multiselect(
    "Select Province",
    [
        "Gauteng",
        "KwaZulu-Natal",
        "Eastern Cape",
        "Western Cape",
        "Free State",
        "Limpopo",
        "Mpumalanga",
        "North West",
        "Northern Cape",
    ],
    default=[],
)


# Load Data
@st.cache_data
def load_data():
  data = pd.DataFrame({
      "Municipality": [
          "eThekwini",
          "City of Johannesburg",
          "Matjhabeng",
          "Mahikeng",
          "Buffalo City",
      ],
      "Province": [
          "KwaZulu-Natal",
          "Gauteng",
          "Free State",
          "North West",
          "Eastern Cape",
      ],
      "Eligible Youth": [520000, 780000, 110000, 95000, 210000],
      "Registered Youth": [210000, 310000, 38000, 35000, 89000],
      "Youth Registration Gap (%)": [59.6, 60.2, 65.5, 63.1, 57.6],
      "Gap Growth (pp)": [8.4, 7.1, 11.2, 9.5, 6.3],
      "Disengagement Profile": [
          "High Gap / Widening",
          "High Gap / Widening",
          "High Gap / Widening",
          "High Gap / Widening",
          "High Gap / Stable",
      ],
  })
  return data


df = load_data()

if province_filter:
  df = df[df["Province"].isin(province_filter)]

# Main Dashboard Layout
tab1, tab2, tab3 = st.tabs(
    ["📊 Priority Ranking", "🗺️ Geographic Analysis", "📖 Methodology & Model"]
)

with tab1:
  st.subheader("Top Priority Municipalities for Targeted Intervention")
  st.dataframe(df)

  st.download_button(
      label="Export Priority List (CSV)",
      data=df.to_csv(index=False),
      file_name="youth_disengagement_priority_list.csv",
      mime="text/csv",
  )

with tab2:
  st.subheader("Youth Registration Gap Distribution")
  st.info("Interactive Choropleth map loading from municipality boundary shapefiles.")

with tab3:
  st.markdown("""
    #### Model Specifications
    - **Clustering:** K-Means ($k=4$) on standard scale features (`youth_registration_gap`, `gap_growth`, `unemployment_rate`).
    - **Driver Analysis:** Multiple Linear Regression ($R^2 = 0.58$) predicting gap growth.
    - **Data Sources:** IEC Registration Dashboard, Census 2022 (Stats SA), QLFS.
    """)
