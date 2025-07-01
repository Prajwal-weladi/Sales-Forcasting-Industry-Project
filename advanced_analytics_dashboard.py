import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series forecasting imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

# Set up the page
st.set_page_config(page_title="Advanced Sales Analytics Dashboard", page_icon="📊", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    all_df = pd.read_csv('all_data.csv')
    all_df["order_purchase_timestamp"] = pd.to_datetime(all_df["order_purchase_timestamp"])
    
    # Create subcategory based on product characteristics
    all_df['product_subcategory'] = all_df.apply(lambda row: create_subcategory(row), axis=1)
    
    return all_df

def create_subcategory(row):
    """Create subcategories based on product characteristics"""
    category = row['product_category_name_english']
    weight = row['product_weight_g']
    price = row['price']
    
    if category == 'health_beauty':
        if weight < 500:
            return 'Light Beauty Products'
        elif weight < 1000:
            return 'Medium Beauty Products'
        else:
            return 'Heavy Beauty Products'
    elif category == 'fashion_shoes':
        if price < 50:
            return 'Budget Shoes'
        elif price < 150:
            return 'Mid-range Shoes'
        else:
            return 'Premium Shoes'
    elif category == 'sports_leisure':
        if weight < 1000:
            return 'Light Sports Equipment'
        else:
            return 'Heavy Sports Equipment'
    elif category == 'baby':
        if price < 30:
            return 'Budget Baby Products'
        else:
            return 'Premium Baby Products'
    else:
        # For other categories, create subcategories based on price
        if price < 25:
            return 'Budget'
        elif price < 75:
            return 'Mid-range'
        else:
            return 'Premium'

# Load data
all_df = load_data()

# Get min and max date
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

# Create constants
BAR_COLOR = "#E36414"
SECONDARY_COLOR = "#0F4C75"

def filter_data_advanced(all_df):
    with st.sidebar:
        st.header("Advanced Filters")
        
        # Category and Subcategory filters
        categories = sorted(all_df["product_category_name_english"].unique())
        selected_categories = st.multiselect(
            "Select Categories:",
            options=categories,
            default=categories[:3] if len(categories) > 3 else categories,
            placeholder="Select categories"
        )
        
        if selected_categories:
            subcategories = sorted(all_df[all_df["product_category_name_english"].isin(selected_categories)]["product_subcategory"].unique())
            selected_subcategories = st.multiselect(
                "Select Subcategories:",
                options=subcategories,
                default=[],
                placeholder="Select subcategories (optional)"
            )
        else:
            selected_subcategories = []
        
        # City filter
        cities = sorted(all_df["customer_city"].unique())
        selected_cities = st.multiselect(
            "Select Cities:",
            options=cities,
            default=[],
            placeholder="Select cities (optional)"
        )
        
        # Future years slider
        future_years = st.slider("How many future years to allow for forecasting?", 1, 20, 5)
        future_max_date = max_date + pd.DateOffset(years=future_years)
        date_range = st.date_input(
            "Select Date Range:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=future_max_date,
        )
        date_range = [pd.Timestamp(d) for d in date_range]
        if len(date_range) == 1:
            date_range = [date_range[0], date_range[0]]

    # Apply filters
    mask = (all_df["order_purchase_timestamp"] >= date_range[0]) & \
           (all_df["order_purchase_timestamp"] <= date_range[1])
    
    if selected_categories:
        mask &= all_df["product_category_name_english"].isin(selected_categories)
    
    if selected_subcategories:
        mask &= all_df["product_subcategory"].isin(selected_subcategories)
    
    if selected_cities:
        mask &= all_df["customer_city"].isin(selected_cities)
    
    df_selection = all_df[mask].copy()

    # --- Forecast for future dates if needed ---
    last_real_date = all_df["order_purchase_timestamp"].max()
    if date_range[1] > last_real_date:
        # Use a simple linear trend forecast for total sales (can be replaced with ARIMA/SARIMA)
        df_hist = all_df.copy()
        df_hist["order_month"] = df_hist["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
        monthly_sales = df_hist.groupby("order_month")["total_price"].sum().reset_index().sort_values("order_month")
        monthly_sales = monthly_sales.sort_values("order_month")
        monthly_sales["t"] = np.arange(len(monthly_sales))
        if len(monthly_sales) > 1:
            slope, intercept = np.polyfit(monthly_sales["t"], monthly_sales["total_price"], 1)
        else:
            slope = 0
            intercept = monthly_sales["total_price"].iloc[0] if len(monthly_sales) == 1 else 0
        last_t = monthly_sales["t"].iloc[-1]
        last_month = monthly_sales["order_month"].iloc[-1]
        forecast_month = (last_month + pd.DateOffset(months=1))
        i = 1
        forecast_rows = []
        while forecast_month <= date_range[1]:
            future_t = last_t + i
            predicted_sales = slope * future_t + intercept
            if predicted_sales < 0:
                predicted_sales = 0
            forecast_rows.append({
                "order_purchase_timestamp": forecast_month,
                "customer_city": selected_cities[0] if selected_cities else "All Cities",
                "product_category_name_english": selected_categories[0] if selected_categories else "All Categories",
                "product_subcategory": selected_subcategories[0] if selected_subcategories else "All Subcategories",
                "total_price": predicted_sales,
                "review_score": np.nan
            })
            forecast_month = forecast_month + pd.DateOffset(months=1)
            i += 1
        if forecast_rows:
            df_forecast = pd.DataFrame(forecast_rows)
            df_selection = pd.concat([df_selection, df_forecast], ignore_index=True)

    return df_selection, selected_categories, selected_subcategories, selected_cities, date_range

def category_subcategory_analysis(df):
    """Category & Subcategory-wise Historical Sales Trend Analysis"""
    st.header("📈 Category & Subcategory Sales Trend Analysis")
    
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Monthly sales by category
    df_monthly = df.copy()
    df_monthly['year_month'] = df_monthly['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()
    
    # Category trend analysis
    category_monthly = df_monthly.groupby(['year_month', 'product_category_name_english'])['total_price'].sum().reset_index()
    
    fig_category_trend = px.line(
        category_monthly,
        x='year_month',
        y='total_price',
        color='product_category_name_english',
        title='Monthly Sales Trend by Category',
        template='plotly_white'
    )
    fig_category_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Sales (R$)",
        height=500
    )
    
    # Subcategory performance within categories
    subcategory_sales = df.groupby(['product_category_name_english', 'product_subcategory'])['total_price'].sum().reset_index()
    subcategory_sales = subcategory_sales.sort_values(['product_category_name_english', 'total_price'], ascending=[True, False])
    
    fig_subcategory = px.bar(
        subcategory_sales,
        x='product_category_name_english',
        y='total_price',
        color='product_subcategory',
        title='Sales by Category and Subcategory',
        template='plotly_white',
        barmode='group'
    )
    fig_subcategory.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Sales (R$)",
        height=500
    )
    
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_category_trend, use_container_width=True)
    col2.plotly_chart(fig_subcategory, use_container_width=True)
    
    # Top performing subcategories
    st.subheader("🏆 Top Performing Subcategories")
    top_subcategories = subcategory_sales.groupby('product_subcategory')['total_price'].sum().sort_values(ascending=False).head(10)
    
    fig_top_sub = px.bar(
        x=top_subcategories.values,
        y=top_subcategories.index,
        orientation='h',
        title='Top 10 Subcategories by Sales',
        color_discrete_sequence=[BAR_COLOR] * len(top_subcategories),
        template='plotly_white'
    )
    fig_top_sub.update_layout(
        xaxis_title="Total Sales (R$)",
        yaxis_title="Subcategory",
        height=400
    )
    
    st.plotly_chart(fig_top_sub, use_container_width=True)
    
    # Seasonal patterns by category
    st.subheader("📅 Seasonal Patterns by Category")
    df_monthly['month'] = df_monthly['order_purchase_timestamp'].dt.month
    df_monthly['month_name'] = df_monthly['order_purchase_timestamp'].dt.strftime('%B')
    
    seasonal_pattern = df_monthly.groupby(['month', 'month_name', 'product_category_name_english'])['total_price'].sum().reset_index()
    seasonal_pattern = seasonal_pattern.sort_values('month')
    
    fig_seasonal = px.line(
        seasonal_pattern,
        x='month_name',
        y='total_price',
        color='product_category_name_english',
        title='Seasonal Sales Patterns by Category',
        template='plotly_white'
    )
    fig_seasonal.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Sales (R$)",
        height=400
    )
    
    st.plotly_chart(fig_seasonal, use_container_width=True)

def seasonal_analysis(df):
    """Seasonal Analysis with Decomposition"""
    st.header("🌍 Seasonal Sales Analysis")
    
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Prepare time series data
    df_ts = df.copy()
    df_ts['date'] = df_ts['order_purchase_timestamp'].dt.date
    daily_sales = df_ts.groupby('date')['total_price'].sum().reset_index()
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales = daily_sales.set_index('date').sort_index()
    
    # Resample to monthly for better seasonal analysis
    monthly_sales = daily_sales.resample('M').sum()
    
    if len(monthly_sales) < 12:
        st.info("Insufficient data for seasonal decomposition (need at least 12 months)")
        return
    
    # Seasonal decomposition
    try:
        decomposition = seasonal_decompose(monthly_sales['total_price'], model='additive', period=12)
        
        # Plot decomposition
        fig_decomp = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.05
        )
        
        fig_decomp.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales['total_price'], name='Original'), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=monthly_sales.index, y=decomposition.trend, name='Trend'), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=monthly_sales.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=monthly_sales.index, y=decomposition.resid, name='Residual'), row=4, col=1)
        
        fig_decomp.update_layout(height=600, title_text="Seasonal Decomposition of Sales")
        st.plotly_chart(fig_decomp, use_container_width=True)
        
        # Seasonal strength analysis
        seasonal_strength = abs(decomposition.seasonal).mean() / abs(monthly_sales['total_price']).mean()
        st.metric("Seasonal Strength", f"{seasonal_strength:.2%}")
        
        if seasonal_strength > 0.1:
            st.success("Strong seasonal pattern detected")
        elif seasonal_strength > 0.05:
            st.info("Moderate seasonal pattern detected")
        else:
            st.warning("Weak seasonal pattern detected")
            
    except Exception as e:
        st.error(f"Error in seasonal decomposition: {str(e)}")
    
    # Monthly heatmap
    st.subheader("📊 Monthly Sales Heatmap")
    df_monthly = df.copy()
    df_monthly['year'] = df_monthly['order_purchase_timestamp'].dt.year
    df_monthly['month'] = df_monthly['order_purchase_timestamp'].dt.month
    
    monthly_heatmap = df_monthly.groupby(['year', 'month'])['total_price'].sum().reset_index()
    monthly_heatmap_pivot = monthly_heatmap.pivot(index='year', columns='month', values='total_price')
    
    fig_heatmap = px.imshow(
        monthly_heatmap_pivot,
        title='Monthly Sales Heatmap',
        color_continuous_scale='Viridis',
        aspect='auto'
    )
    fig_heatmap.update_layout(
        xaxis_title="Month",
        yaxis_title="Year",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

def city_month_analysis(df):
    """City-wise and Month-wise Sales Analysis"""
    st.header("🏙️ City-wise Sales Analysis")
    
    if df.empty:
        st.warning("No data available for the selected filters.")
        return None, None
    
    # Top cities by sales
    city_sales = df.groupby('customer_city')['total_price'].sum().sort_values(ascending=False).head(15)
    
    fig_city_sales = px.bar(
        x=city_sales.values,
        y=city_sales.index,
        orientation='h',
        title='Top 15 Cities by Sales',
        color_discrete_sequence=[BAR_COLOR] * len(city_sales),
        template='plotly_white'
    )
    fig_city_sales.update_layout(
        xaxis_title="Total Sales (R$)",
        yaxis_title="City",
        height=500
    )
    
    st.plotly_chart(fig_city_sales, use_container_width=True)
    
    # City-month analysis
    st.subheader("📅 City-Month Sales Patterns")
    df_city_month = df.copy()
    df_city_month['month'] = df_city_month['order_purchase_timestamp'].dt.strftime('%Y-%m')
    
    # Get top 5 cities for detailed analysis
    top_cities = city_sales.head(5).index.tolist()
    city_month_data = df_city_month[df_city_month['customer_city'].isin(top_cities)]
    
    if not city_month_data.empty:
        city_month_sales = city_month_data.groupby(['customer_city', 'month'])['total_price'].sum().reset_index()
        
        fig_city_month = px.line(
            city_month_sales,
            x='month',
            y='total_price',
            color='customer_city',
            title='Monthly Sales Trend by Top Cities',
            template='plotly_white'
        )
        fig_city_month.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Sales (R$)",
            height=400
        )
        
        st.plotly_chart(fig_city_month, use_container_width=True)
    
    # City-category preferences
    st.subheader("🏪 City-Category Preferences")
    city_category = df.groupby(['customer_city', 'product_category_name_english'])['total_price'].sum().reset_index()
    city_category_pivot = city_category.pivot(index='customer_city', columns='product_category_name_english', values='total_price').fillna(0)
    
    # Show top 10 cities
    top_10_cities = city_sales.head(10).index.tolist()
    city_category_top = city_category_pivot.loc[city_category_pivot.index.isin(top_10_cities)]
    
    fig_city_category = px.imshow(
        city_category_top,
        title='Category Preferences by City (Top 10 Cities)',
        color_continuous_scale='Viridis',
        aspect='auto'
    )
    fig_city_category.update_layout(
        xaxis_title="Category",
        yaxis_title="City",
        height=400
    )
    
    st.plotly_chart(fig_city_category, use_container_width=True)
    
    return city_sales, city_month_sales if 'city_month_sales' in locals() else None

def advanced_forecasting(df, selected_categories, selected_cities):
    """Advanced Forecasting with Multiple Models"""
    st.header("🔮 Advanced Sales Forecasting")
    
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Prepare time series data
    df_ts = df.copy()
    df_ts['date'] = df_ts['order_purchase_timestamp'].dt.date
    daily_sales = df_ts.groupby('date')['total_price'].sum().reset_index()
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales = daily_sales.set_index('date').sort_index()
    
    # Resample to monthly
    monthly_sales = daily_sales.resample('M').sum()
    
    if len(monthly_sales) < 6:
        st.info("Insufficient data for forecasting (need at least 6 months)")
        return
    
    # Split data for training and testing
    train_size = int(len(monthly_sales) * 0.8)
    train_data = monthly_sales[:train_size]
    test_data = monthly_sales[train_size:]
    
    st.subheader("📈 Model Performance Comparison")
    
    models = {}
    forecasts = {}
    
    # ARIMA Model
    try:
        # Find optimal ARIMA parameters
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        
        best_aic = float('inf')
        best_pdq = None
        
        for param in pdq:
            try:
                model = ARIMA(train_data['total_price'], order=param)
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_pdq = param
            except:
                continue
        
        if best_pdq:
            arima_model = ARIMA(train_data['total_price'], order=best_pdq)
            fitted_arima = arima_model.fit()
            models['ARIMA'] = fitted_arima
            
            # Forecast
            forecast_arima = fitted_arima.forecast(steps=len(test_data))
            forecasts['ARIMA'] = forecast_arima
            
            # Calculate metrics
            mae_arima = mean_absolute_error(test_data['total_price'], forecast_arima)
            rmse_arima = np.sqrt(mean_squared_error(test_data['total_price'], forecast_arima))
            
            st.metric("ARIMA MAE", f"R$ {mae_arima:,.2f}")
            st.metric("ARIMA RMSE", f"R$ {rmse_arima:,.2f}")
    except Exception as e:
        st.error(f"ARIMA model error: {str(e)}")
    
    # SARIMA Model (if we have enough data)
    if len(monthly_sales) >= 12:
        try:
            # Try SARIMA with seasonal component
            sarima_model = SARIMAX(train_data['total_price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fitted_sarima = sarima_model.fit(disp=False)
            models['SARIMA'] = fitted_sarima
            
            # Forecast
            forecast_sarima = fitted_sarima.forecast(steps=len(test_data))
            forecasts['SARIMA'] = forecast_sarima
            
            # Calculate metrics
            mae_sarima = mean_absolute_error(test_data['total_price'], forecast_sarima)
            rmse_sarima = np.sqrt(mean_squared_error(test_data['total_price'], forecast_sarima))
            
            st.metric("SARIMA MAE", f"R$ {mae_sarima:,.2f}")
            st.metric("SARIMA RMSE", f"R$ {rmse_sarima:,.2f}")
        except Exception as e:
            st.error(f"SARIMA model error: {str(e)}")
    
    # Plot forecasts
    if forecasts:
        st.subheader("📊 Forecast Comparison")
        
        fig_forecast = go.Figure()
        
        # Actual data
        fig_forecast.add_trace(go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales['total_price'],
            mode='lines',
            name='Actual',
            line=dict(color='black')
        ))
        
        # Forecasts
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            fig_forecast.add_trace(go.Scatter(
                x=test_data.index,
                y=forecast,
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))
        
        fig_forecast.update_layout(
            title='Sales Forecast Comparison',
            xaxis_title='Date',
            yaxis_title='Sales (R$)',
            height=500
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Future forecasting
    st.subheader("🔮 Future Sales Forecast")
    forecast_months = st.slider("Number of months to forecast", 1, 12, 6)
    
    if models:
        best_model_name = min(models.keys(), key=lambda x: models[x].aic if hasattr(models[x], 'aic') else float('inf'))
        best_model = models[best_model_name]
        
        # Generate future forecast
        future_forecast = best_model.forecast(steps=forecast_months)
        future_dates = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='M')
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Sales': future_forecast
        })
        
        st.dataframe(forecast_df)
        
        # Plot future forecast
        fig_future = go.Figure()
        
        # Historical data
        fig_future.add_trace(go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales['total_price'],
            mode='lines',
            name='Historical',
            line=dict(color='black')
        ))
        
        # Future forecast
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=future_forecast,
            mode='lines',
            name='Future Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig_future.update_layout(
            title=f'Future Sales Forecast ({best_model_name} Model)',
            xaxis_title='Date',
            yaxis_title='Sales (R$)',
            height=400
        )
        
        st.plotly_chart(fig_future, use_container_width=True)

def generate_recommendations(df, top_cities, selected_categories, selected_cities, date_range):
    """Generate City and Month-wise Recommendations"""
    st.header("💡 Strategic Recommendations")
    
    if df.empty:
        st.warning("No data available for recommendations.")
        return
    
    # City-wise recommendations
    st.subheader("🏙️ City-wise Recommendations")
    
    if top_cities is not None and not top_cities.empty:
        # Top performing cities
        top_5_cities = top_cities.head(5)
        
        for city in top_5_cities.index:
            city_data = df[df['customer_city'] == city]
            
            # Top categories for this city
            city_categories = city_data.groupby('product_category_name_english')['total_price'].sum().sort_values(ascending=False).head(3)
            
            # Top subcategories for this city
            city_subcategories = city_data.groupby('product_subcategory')['total_price'].sum().sort_values(ascending=False).head(3)
            
            # Seasonal analysis for this city
            city_data['month'] = city_data['order_purchase_timestamp'].dt.month
            city_monthly = city_data.groupby('month')['total_price'].sum()
            peak_month = city_monthly.idxmax()
            peak_month_name = pd.Timestamp(2020, peak_month, 1).strftime('%B')
            
            with st.expander(f"📊 {city} - Total Sales: R$ {top_5_cities[city]:,.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Categories:**")
                    for cat, sales in city_categories.items():
                        st.write(f"• {cat}: R$ {sales:,.2f}")
                    
                    st.write("**Peak Sales Month:**")
                    st.write(f"• {peak_month_name}")
                
                with col2:
                    st.write("**Top Subcategories:**")
                    for subcat, sales in city_subcategories.items():
                        st.write(f"• {subcat}: R$ {sales:,.2f}")
                    
                    st.write("**Recommendations:**")
                    st.write(f"• Focus on {city_categories.index[0]} category")
                    st.write(f"• Increase inventory in {peak_month_name}")
                    st.write(f"• Promote {city_subcategories.index[0]} subcategory")
    
    # Month-wise recommendations
    st.subheader("📅 Month-wise Recommendations")
    
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['order_purchase_timestamp'].dt.month
    df_monthly['month_name'] = df_monthly['order_purchase_timestamp'].dt.strftime('%B')
    
    # Overall monthly patterns
    monthly_pattern = df_monthly.groupby(['month', 'month_name'])['total_price'].sum().reset_index()
    monthly_pattern = monthly_pattern.sort_values('month')
    
    # Find peak and low months
    peak_month = monthly_pattern.loc[monthly_pattern['total_price'].idxmax()]
    low_month = monthly_pattern.loc[monthly_pattern['total_price'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Peak Sales Month", peak_month['month_name'], f"R$ {peak_month['total_price']:,.2f}")
        st.write("**Actions:**")
        st.write("• Increase inventory levels")
        st.write("• Launch promotional campaigns")
        st.write("• Extend marketing budget")
    
    with col2:
        st.metric("Low Sales Month", low_month['month_name'], f"R$ {low_month['total_price']:,.2f}")
        st.write("**Actions:**")
        st.write("• Reduce inventory costs")
        st.write("• Focus on high-margin items")
        st.write("• Plan clearance sales")
    
    # Category-specific monthly recommendations
    st.subheader("📊 Category-Month Optimization")
    
    category_monthly = df_monthly.groupby(['month_name', 'product_category_name_english'])['total_price'].sum().reset_index()
    
    # Find best performing category for each month
    best_category_by_month = category_monthly.loc[category_monthly.groupby('month_name')['total_price'].idxmax()]
    
    fig_category_month = px.bar(
        best_category_by_month,
        x='month_name',
        y='total_price',
        color='product_category_name_english',
        title='Best Performing Category by Month',
        template='plotly_white'
    )
    fig_category_month.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales (R$)",
        height=400
    )
    
    st.plotly_chart(fig_category_month, use_container_width=True)
    
    # Generate specific recommendations
    st.subheader("🎯 Actionable Recommendations")
    
    recommendations = []
    
    # Inventory recommendations
    for _, row in best_category_by_month.iterrows():
        recommendations.append({
            "Month": row['month_name'],
            "Category": row['product_category_name_english'],
            "Action": "Increase inventory",
            "Reason": f"Best performing category in {row['month_name']}"
        })
    
    # Marketing recommendations
    for city in top_cities.head(3).index if top_cities is not None else []:
        city_data = df[df['customer_city'] == city]
        top_cat = city_data.groupby('product_category_name_english')['total_price'].sum().idxmax()
        recommendations.append({
            "City": city,
            "Category": top_cat,
            "Action": "Target marketing",
            "Reason": f"Top category in {city}"
        })
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)

def retail_optimization_insights(df, selected_categories, selected_cities):
    """Retail Sales Optimization Insights"""
    st.header("🛒 Retail Optimization Insights")
    
    if df.empty:
        st.warning("No data available for optimization analysis.")
        return
    
    # Inventory optimization
    st.subheader("📦 Inventory Optimization")
    
    # Calculate inventory turnover metrics
    df_inventory = df.copy()
    df_inventory['month'] = df_inventory['order_purchase_timestamp'].dt.to_period('M')
    
    # Monthly sales by category
    monthly_category_sales = df_inventory.groupby(['month', 'product_category_name_english'])['total_price'].sum().reset_index()
    
    # Calculate coefficient of variation (CV) for demand variability
    cv_by_category = monthly_category_sales.groupby('product_category_name_english')['total_price'].agg(['mean', 'std']).reset_index()
    cv_by_category['cv'] = cv_by_category['std'] / cv_by_category['mean']
    cv_by_category = cv_by_category.sort_values('cv', ascending=False)
    
    # High CV = high variability = need safety stock
    high_variability = cv_by_category.head(5)
    low_variability = cv_by_category.tail(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**High Variability Categories (Need Safety Stock):**")
        for _, row in high_variability.iterrows():
            st.write(f"• {row['product_category_name_english']}: CV = {row['cv']:.2f}")
    
    with col2:
        st.write("**Low Variability Categories (Predictable Demand):**")
        for _, row in low_variability.iterrows():
            st.write(f"• {row['product_category_name_english']}: CV = {row['cv']:.2f}")
    
    # Stockout risk analysis
    st.subheader("⚠️ Stockout Risk Analysis")
    
    # Identify categories with increasing trends
    trend_analysis = []
    
    for category in df['product_category_name_english'].unique():
        cat_data = monthly_category_sales[monthly_category_sales['product_category_name_english'] == category]
        if len(cat_data) >= 3:
            # Calculate trend
            cat_data = cat_data.sort_values('month')
            x = np.arange(len(cat_data))
            y = cat_data['total_price'].values
            slope = np.polyfit(x, y, 1)[0]
            
            trend_analysis.append({
                'category': category,
                'trend': slope,
                'avg_sales': cat_data['total_price'].mean(),
                'recent_sales': cat_data['total_price'].iloc[-3:].mean()
            })
    
    trend_df = pd.DataFrame(trend_analysis)
    trend_df = trend_df.sort_values('trend', ascending=False)
    
    # High growth categories (risk of stockout)
    high_growth = trend_df.head(5)
    declining = trend_df.tail(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**High Growth Categories (Stockout Risk):**")
        for _, row in high_growth.iterrows():
            growth_rate = (row['recent_sales'] - row['avg_sales']) / row['avg_sales'] * 100
            st.write(f"• {row['category']}: {growth_rate:+.1f}% growth")
    
    with col2:
        st.write("**Declining Categories (Overstock Risk):**")
        for _, row in declining.iterrows():
            decline_rate = (row['recent_sales'] - row['avg_sales']) / row['avg_sales'] * 100
            st.write(f"• {row['category']}: {decline_rate:+.1f}% decline")
    
    # Promotional optimization
    st.subheader("🎯 Promotional Optimization")
    
    # Identify best promotional timing
    df_promo = df.copy()
    df_promo['day_of_week'] = df_promo['order_purchase_timestamp'].dt.dayofweek
    df_promo['day_name'] = df_promo['order_purchase_timestamp'].dt.strftime('%A')
    
    # Day of week analysis
    dow_sales = df_promo.groupby(['day_of_week', 'day_name'])['total_price'].sum().reset_index()
    dow_sales = dow_sales.sort_values('day_of_week')
    
    fig_dow = px.bar(
        dow_sales,
        x='day_name',
        y='total_price',
        title='Sales by Day of Week',
        template='plotly_white'
    )
    fig_dow.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Total Sales (R$)",
        height=400
    )
    
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # Best and worst days for promotions
    best_day = dow_sales.loc[dow_sales['total_price'].idxmax()]
    worst_day = dow_sales.loc[dow_sales['total_price'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Best Day for Promotions", best_day['day_name'], f"R$ {best_day['total_price']:,.2f}")
        st.write("**Strategy:**")
        st.write("• Launch new products")
        st.write("• Premium promotions")
        st.write("• High-margin items")
    
    with col2:
        st.metric("Day for Clearance", worst_day['day_name'], f"R$ {worst_day['total_price']:,.2f}")
        st.write("**Strategy:**")
        st.write("• Clearance sales")
        st.write("• Inventory reduction")
        st.write("• Low-margin promotions")
    
    # Price optimization insights
    st.subheader("💰 Price Optimization")
    
    # Price elasticity analysis
    df_price = df.copy()
    df_price['price_range'] = pd.cut(df_price['price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    price_sales = df_price.groupby('price_range')['total_price'].sum().reset_index()
    
    fig_price = px.bar(
        price_sales,
        x='price_range',
        y='total_price',
        title='Sales by Price Range',
        template='plotly_white'
    )
    fig_price.update_layout(
        xaxis_title="Price Range",
        yaxis_title="Total Sales (R$)",
        height=400
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Optimal price range
    optimal_range = price_sales.loc[price_sales['total_price'].idxmax()]
    st.success(f"**Optimal Price Range:** {optimal_range['price_range']} - Generates R$ {optimal_range['total_price']:,.2f} in sales")

def main():
    st.title("🚀 Advanced Sales Analytics & Forecasting Dashboard")
    st.markdown("### Comprehensive Retail Intelligence Platform")
    
    # Load and filter data
    df_selection, selected_categories, selected_subcategories, selected_cities, date_range = filter_data_advanced(all_df)
    
    # Display KPIs
    if not df_selection.empty:
        total_sales = int(df_selection["total_price"].sum())
        total_orders = len(df_selection["order_id"].unique())
        avg_order_value = round(df_selection["total_price"].mean(), 2)
        avg_rating = round(df_selection["review_score"].mean(), 1) if "review_score" in df_selection else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sales", f"R$ {total_sales:,}")
        with col2:
            st.metric("Total Orders", f"{total_orders:,}")
        with col3:
            st.metric("Avg Order Value", f"R$ {avg_order_value:,}")
        with col4:
            st.metric("Avg Rating", f"{avg_rating} ⭐")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Category Analysis", "🌍 Seasonal Analysis", "🏙️ City Analysis", 
        "🔮 Forecasting", "💡 Recommendations", "🛒 Optimization"
    ])
    
    with tab1:
        category_subcategory_analysis(df_selection)
    
    with tab2:
        seasonal_analysis(df_selection)
    
    with tab3:
        top_cities, city_month_sales = city_month_analysis(df_selection)
    
    with tab4:
        advanced_forecasting(df_selection, selected_categories, selected_cities)
    
    with tab5:
        generate_recommendations(df_selection, top_cities, selected_categories, selected_cities, date_range)
    
    with tab6:
        retail_optimization_insights(df_selection, selected_categories, selected_cities)
    
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