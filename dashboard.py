import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Set up the page
st.set_page_config(page_title="Sales Performance Dashboard", page_icon=":bar_chart:", layout="wide")

# Load the dataset
all_df = pd.read_csv('all_data.csv')
all_df["order_purchase_timestamp"] = pd.to_datetime(all_df["order_purchase_timestamp"])

# Get min and max date
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

# Create constants
BAR_COLOR = "#E36414"

def filter_data_with_forecast(all_df):
    with st.sidebar:
        st.header("Filter here:")
        city = st.sidebar.multiselect(
            "Select the City:",
            options=all_df["customer_city"].unique(),
            default=[],  # Default to no selection (all cities)
            placeholder="Select a city",
        )
        # Set max_value to 20 years after the latest date in the data
        future_max_date = max_date + pd.DateOffset(years=20)
        date_range = st.sidebar.date_input(
            "Select the Date Range:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=future_max_date,
        )
        date_range = [pd.Timestamp(d) for d in date_range]
        if len(date_range) == 1:
            date_range = [date_range[0], date_range[0]]

    # Filter for selected city and date range (historical data)
    if city:
        df_selection = all_df[
            all_df["customer_city"].isin(city) &
            (all_df["order_purchase_timestamp"] >= date_range[0]) &
            (all_df["order_purchase_timestamp"] <= date_range[1])
        ].copy()
    else:
        df_selection = all_df[
            (all_df["order_purchase_timestamp"] >= date_range[0]) &
            (all_df["order_purchase_timestamp"] <= date_range[1])
        ].copy()

    # If user selected future dates, forecast for those months
    last_real_date = all_df["order_purchase_timestamp"].max()
    if date_range[1] > last_real_date:
        categories = all_df["product_category_name_english"].unique()
        forecast_rows = []
        forecast_cities = city if city else all_df["customer_city"].unique()
        for category in categories:
            for forecast_city in forecast_cities:
                # Try city+category, else fallback to category only
                df_cat_city = all_df[(all_df["product_category_name_english"] == category) & (all_df["customer_city"] == forecast_city)].copy()
                if not df_cat_city.empty:
                    df_cat = df_cat_city
                    use_city = True
                else:
                    df_cat = all_df[all_df["product_category_name_english"] == category].copy()
                    use_city = False
                if df_cat.empty:
                    continue
                df_cat["order_month"] = df_cat["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
                monthly_sales = df_cat.groupby("order_month")["total_price"].sum().reset_index().sort_values("order_month")
                if monthly_sales.empty:
                    continue
                monthly_sales = monthly_sales.sort_values("order_month")
                monthly_sales["t"] = np.arange(len(monthly_sales))
                # If only one month, use its value as a flat forecast
                if len(monthly_sales) == 1:
                    slope = 0
                    intercept = monthly_sales["total_price"].iloc[0]
                else:
                    slope, intercept = np.polyfit(monthly_sales["t"], monthly_sales["total_price"], 1)
                last_t = monthly_sales["t"].iloc[-1]
                last_month = monthly_sales["order_month"].iloc[-1]
                forecast_month = (last_month + pd.DateOffset(months=1))
                i = 1
                while forecast_month <= date_range[1]:
                    future_t = last_t + i
                    predicted_sales = slope * future_t + intercept
                    if predicted_sales < 0:
                        predicted_sales = 0
                    forecast_rows.append({
                        "order_purchase_timestamp": forecast_month,
                        "customer_city": forecast_city if use_city else "All Cities",
                        "product_category_name_english": category,
                        "total_price": predicted_sales,
                        "review_score": np.nan
                    })
                    forecast_month = forecast_month + pd.DateOffset(months=1)
                    i += 1
        if forecast_rows:
            df_forecast = pd.DataFrame(forecast_rows)
            df_selection = pd.concat([df_selection, df_forecast], ignore_index=True)
    return df_selection

def display_kpis(df_selection):
    total_sales = int(df_selection["total_price"].sum()) if "total_price" in df_selection else 0
    average_rating = round(df_selection["review_score"].mean(), 1) if "review_score" in df_selection else 0
    average_sales_per_order = round(df_selection["total_price"].mean(), 2) if "total_price" in df_selection else 0

    total_sales = 0 if np.isnan(total_sales) else total_sales
    average_rating = 0 if np.isnan(average_rating) else average_rating
    average_sales_per_order = 0 if np.isnan(average_sales_per_order) else average_sales_per_order

    star_rating = ":star:" * int(round(average_rating, 0))

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Sales")
        st.subheader(f"R$ {total_sales:,}")

    with middle_column:
        st.subheader("Average Rating")
        st.subheader(f"{star_rating} {average_rating}")

    with right_column:
        st.subheader("Average Sales per Order")
        st.subheader(f"R$ {average_sales_per_order:,}")

    st.markdown("""---""")

def plot_charts(df):
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Sales by Product Line
    sales_by_product_line = df.groupby("product_category_name_english")["total_price"].sum().fillna(0).sort_values()
    
    fig_product_sales = px.bar(
        sales_by_product_line,
        x=sales_by_product_line.values,
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>Sales by Product Line</b>",
        color_discrete_sequence=[BAR_COLOR] * len(sales_by_product_line),
        template="plotly_white",
    )
    
    # Daily Sales for all products: group by full date
    df["order_date"] = df["order_purchase_timestamp"].dt.date
    sales_by_date = df.groupby("order_date")["total_price"].sum().reset_index().fillna(0)
    
    fig_daily_sales = px.bar(
        sales_by_date,
        x="order_date",
        y="total_price",
        title="<b>Daily Sales (All Products)</b>",
        color_discrete_sequence=[BAR_COLOR] * len(sales_by_date),
        template="plotly_white",
    )
    fig_daily_sales.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickformat='%Y-%m-%d'
    )

    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig_daily_sales, use_container_width=True)
    right_column.plotly_chart(fig_product_sales, use_container_width=True)

def plot_category_sales(df):
    # Create a dropdown for the product categories present in the filtered data
    categories = sorted(df["product_category_name_english"].unique())
    selected_category = st.selectbox("Select a Product Category to view its Daily Sales", options=categories)
    
    # Filter the data for the selected category
    df_category = df[df["product_category_name_english"] == selected_category].copy()
    
    # Group by full date (not just day-of-month)
    df_category["order_date"] = df_category["order_purchase_timestamp"].dt.date
    sales_category_date = df_category.groupby("order_date")["total_price"].sum().reset_index().fillna(0)
    
    fig_category_sales = px.bar(
        sales_category_date,
        x="order_date",
        y="total_price",
        title=f"<b>Daily Sales for {selected_category}</b>",
        color_discrete_sequence=[BAR_COLOR],
        template="plotly_white",
    )
    fig_category_sales.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickformat='%Y-%m-%d'
    )
    
    st.plotly_chart(fig_category_sales, use_container_width=True)
    
    return selected_category

def forecast_inventory(df, selected_category):
    # Filter data for the selected category
    df_category = df[df["product_category_name_english"] == selected_category].copy()
    if df_category.empty:
        st.info("No sales data available for the selected category to forecast inventory.")
        return None

    # Group by month using order_purchase_timestamp
    df_category["order_month"] = df_category["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly_sales = df_category.groupby("order_month")["total_price"].sum().reset_index().sort_values("order_month")
    
    if monthly_sales.empty or len(monthly_sales) < 2:
        st.info("Insufficient monthly sales data to forecast inventory.")
        return None

    # Use a simple linear regression to capture the trend
    monthly_sales = monthly_sales.sort_values("order_month")
    monthly_sales["t"] = np.arange(len(monthly_sales))
    slope, intercept = np.polyfit(monthly_sales["t"], monthly_sales["total_price"], 1)

    # Let the user select the number of future months for forecast
    forecast_months = st.slider("Select number of future months to forecast inventory", min_value=1, max_value=12, value=3)
    
    forecast_data = []
    last_t = monthly_sales["t"].iloc[-1]
    last_month = monthly_sales["order_month"].iloc[-1]
    
    for i in range(1, forecast_months + 1):
        future_t = last_t + i
        # Predict using the linear model and set negative predictions to 0
        predicted_sales = slope * future_t + intercept
        if predicted_sales < 0:
            predicted_sales = 0
        forecast_month = (last_month + pd.DateOffset(months=i)).strftime("%Y-%m")
        forecast_data.append({
            "Month": forecast_month,
            "Predicted Monthly Sales": round(predicted_sales, 2)
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    return forecast_df

def main():
    st.title("Sales Performance Dashboard")
    st.markdown("##")
    
    df_selection = filter_data_with_forecast(all_df)
    display_kpis(df_selection)
    plot_charts(df_selection)
    
    # Category-specific graph and selection
    selected_category = plot_category_sales(df_selection)
    
    # Forecast inventory recommendation for the selected category
    st.markdown("### Inventory Recommendation Forecast")
    forecast_df = forecast_inventory(df_selection, selected_category)
    if forecast_df is not None:
        st.dataframe(forecast_df)
        
        # Check if any forecasted sales are 0
        if any(forecast_df["Predicted Monthly Sales"] == 0):
            st.markdown(
                """
                **Note:** It is suggested not to produce this item further and to consider removing it from inventory as its sales are decreasing currently.
                """
            )
        else:
            st.markdown(
                f"""
                Based on the historical monthly sales data for **{selected_category}**, we used a simple trend model to forecast future sales.
                Adjust your inventory orders based on these predicted monthly sales values.
                """
            )
    
    # Hide Streamlit UI elements
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
