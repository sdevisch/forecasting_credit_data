import glob
import os

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Credit Data Dashboard", layout="wide")

st.title("Credit Portfolio Dashboard")

base_dir = "data/processed"

# Find latest multiproduct sample
samples = sorted(glob.glob(os.path.join(base_dir, "sample_multi_*")), reverse=True)
if not samples:
    st.warning("No sample datasets found under data/processed.")
    st.stop()

sample = st.sidebar.selectbox("Dataset", samples, index=0)
cecl_dir = os.path.join(sample, "cecl_multi")

if not os.path.exists(cecl_dir):
    st.warning("CECL outputs not found. Run the CECL script first.")
    st.stop()

by_product_path = os.path.join(cecl_dir, "portfolio_aggregates_by_product.parquet")
overall_path = os.path.join(cecl_dir, "portfolio_aggregates_overall.parquet")

if not (os.path.exists(by_product_path) and os.path.exists(overall_path)):
    st.warning("Portfolio aggregate files not found.")
    st.stop()

by_product = pd.read_parquet(by_product_path)
overall = pd.read_parquet(overall_path)

st.subheader("Overall Portfolio Monthly ECL")
st.line_chart(overall.set_index("asof_month")["portfolio_monthly_ecl"])

st.subheader("Portfolio ECL by Product")
prod = st.selectbox("Product", sorted(by_product["product"].unique().tolist()))
prod_df = by_product[by_product["product"] == prod]
st.line_chart(prod_df.set_index("asof_month")["portfolio_monthly_ecl"])
