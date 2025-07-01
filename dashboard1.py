import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Load and process the dataset
with open("products.json", "r") as f:
    data = json.load(f)
    products = data["products"]["data"]["items"]

df = pd.DataFrame(products)
df["price"] = df["price"].astype(float)  # Convert price to float

# Dashboard title and layout
st.set_page_config(page_title="Sales Forecast Dashboard", page_icon=":bar_chart:", layout="wide")
st.title("📊 Sales Performance & Inventory Forecast Dashboard")

# Price range filter
min_price, max_price = df["price"].min(), df["price"].max()
price_range = st.slider("Select Price Range:", min_value=min_price, max_value=max_price, value=(min_price, max_price))
df_filtered = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]

# KPI Metrics
total_products = len(df_filtered)
avg_price = round(df_filtered["price"].mean(), 2)
st.subheader("🔹 Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Products", total_products)
col2.metric("Average Price ($)", avg_price)
st.markdown("---")

# Category-wise sales analysis
st.subheader("📈 Category-Wise Sales Trends")
category_sales = df_filtered.groupby("category")["price"].sum().reset_index()
fig_category = px.bar(category_sales, x="category", y="price", title="Total Sales by Category", color="category")
st.plotly_chart(fig_category, use_container_width=True)

# Subcategory trends
st.subheader("📊 Subcategory Trends")
subcategories = df_filtered["subcategory"].unique()
selected_subcat = st.selectbox("Select a Subcategory:", subcategories)
df_subcat = df_filtered[df_filtered["subcategory"] == selected_subcat]
fig_subcat = px.bar(df_subcat, x="name", y="price", title=f"Sales Trend in {selected_subcat}", color="price")
st.plotly_chart(fig_subcat, use_container_width=True)

# Forecast Inventory Recommendations
st.subheader("📌 Inventory Forecast Recommendations")
inventory_df = df_filtered.groupby("category")["price"].mean().reset_index()
inventory_df["Recommended Stock"] = (inventory_df["price"] * 10).astype(int)  # Simulated stock recommendation
st.dataframe(inventory_df)
st.markdown("⚡ Keep more stock for high-demand categories and adjust inventory for premium products.")

# Hide Streamlit UI elements
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
