# Advanced Sales Analytics & Forecasting Dashboard

## 🚀 Overview

This comprehensive retail intelligence platform provides advanced analytics, forecasting, and optimization insights for e-commerce businesses. The dashboard combines historical data analysis with predictive modeling to deliver actionable business recommendations.

## 📊 Key Features

### 1. Category & Subcategory Analysis
- **Historical Sales Trends**: Analyze sales patterns over time by category and subcategory
- **Top Performing Subcategories**: Identify which subcategories contribute most to revenue within each category
- **Seasonal Patterns**: Discover category-specific seasonal effects and emerging trends
- **Subcategory Performance**: Compare subcategory performance across different categories

### 2. Seasonal Analysis
- **Seasonal Decomposition**: Break down sales data into trend, seasonal, and residual components
- **Seasonal Strength Measurement**: Quantify the strength of seasonal patterns
- **Monthly Heatmaps**: Visualize sales patterns across months and years
- **Seasonal Pattern Detection**: Automatically identify strong, moderate, or weak seasonal patterns

### 3. City-wise & Month-wise Analysis
- **Geographic Performance**: Analyze sales performance across different cities
- **City-Category Preferences**: Identify which categories perform best in each city
- **Monthly Sales Patterns**: Track sales trends by city over time
- **Local Demand Patterns**: Discover city-specific product preferences and seasonality

### 4. Advanced Forecasting Models
- **ARIMA Models**: Auto-regressive Integrated Moving Average models for time series forecasting
- **SARIMA Models**: Seasonal ARIMA models accounting for seasonal patterns
- **Model Performance Comparison**: Compare different forecasting models using MAE and RMSE metrics
- **Future Sales Prediction**: Generate forecasts for up to 12 months ahead
- **Category-wise Forecasting**: Predict demand for specific product categories

### 5. Strategic Recommendations
- **City-wise Recommendations**: Tailored recommendations for each city based on local performance
- **Month-wise Optimization**: Seasonal recommendations for inventory and marketing
- **Category Prioritization**: Identify which categories to focus on in each city and month
- **Actionable Insights**: Specific recommendations for inventory, marketing, and promotions

### 6. Retail Optimization
- **Inventory Optimization**: Identify categories with high/low demand variability
- **Stockout Risk Analysis**: Detect categories at risk of stockouts or overstock
- **Promotional Optimization**: Determine the best days for promotions and clearance sales
- **Price Optimization**: Analyze price elasticity and identify optimal price ranges

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements_advanced.txt)

### Installation Steps
1. Install dependencies:
```bash
pip install -r requirements_advanced.txt
```

2. Ensure your data file (`all_data.csv`) is in the backend directory

3. Run the advanced dashboard:
```bash
streamlit run advanced_analytics_dashboard.py
```

## 📈 Dashboard Sections

### 1. Category Analysis Tab
- **Monthly Sales Trends**: Line charts showing category performance over time
- **Category-Subcategory Breakdown**: Bar charts comparing subcategory performance
- **Top Subcategories**: Horizontal bar chart of top 10 subcategories by sales
- **Seasonal Patterns**: Monthly sales patterns for each category

### 2. Seasonal Analysis Tab
- **Seasonal Decomposition**: Four-panel chart showing original data, trend, seasonal, and residual components
- **Seasonal Strength Metrics**: Quantitative measurement of seasonal patterns
- **Monthly Heatmap**: Color-coded heatmap showing sales intensity by month and year

### 3. City Analysis Tab
- **Top Cities**: Bar chart of top 15 cities by total sales
- **City-Month Trends**: Line charts showing sales trends for top 5 cities
- **City-Category Preferences**: Heatmap showing category preferences by city

### 4. Forecasting Tab
- **Model Performance**: Comparison of ARIMA and SARIMA models with error metrics
- **Forecast Comparison**: Side-by-side comparison of different forecasting models
- **Future Predictions**: Interactive slider to select forecast horizon (1-12 months)
- **Forecast Visualization**: Charts showing historical data and future predictions

### 5. Recommendations Tab
- **City-wise Insights**: Expandable sections for each top city with specific recommendations
- **Month-wise Optimization**: Peak and low sales month analysis with action items
- **Category-Month Optimization**: Best performing categories by month
- **Actionable Recommendations**: Dataframe with specific recommendations

### 6. Optimization Tab
- **Inventory Optimization**: High/low variability category analysis
- **Stockout Risk Analysis**: Growth and declining category identification
- **Promotional Optimization**: Day-of-week analysis for best promotion timing
- **Price Optimization**: Price range analysis and optimal pricing insights

## 🎯 Business Applications

### Inventory Management
- **Safety Stock Planning**: Identify categories requiring safety stock based on demand variability
- **Stockout Prevention**: Detect categories with increasing trends that may need additional inventory
- **Overstock Prevention**: Identify declining categories to reduce inventory levels

### Marketing Strategy
- **Seasonal Campaigns**: Plan marketing campaigns based on seasonal patterns
- **Geographic Targeting**: Focus marketing efforts on high-performing cities
- **Category Promotion**: Promote categories that perform well in specific cities/months

### Pricing Strategy
- **Price Optimization**: Identify optimal price ranges for maximum revenue
- **Promotional Timing**: Determine the best days for promotions and clearance sales
- **Category Pricing**: Adjust pricing strategies based on category performance

### Supply Chain Optimization
- **Demand Forecasting**: Predict future demand for better supply chain planning
- **Seasonal Planning**: Prepare for seasonal demand fluctuations
- **Geographic Distribution**: Optimize inventory distribution across cities

## 🔧 Advanced Features

### Filtering Capabilities
- **Category Selection**: Filter by specific product categories
- **Subcategory Selection**: Filter by product subcategories
- **City Selection**: Filter by specific cities
- **Date Range Selection**: Analyze specific time periods

### Forecasting Models
- **ARIMA**: Best for non-seasonal time series data
- **SARIMA**: Best for seasonal time series data
- **Automatic Parameter Selection**: Automatically finds optimal model parameters
- **Model Validation**: Uses train-test split to validate model performance

### Interactive Visualizations
- **Plotly Charts**: Interactive charts with zoom, pan, and hover capabilities
- **Responsive Design**: Charts adapt to different screen sizes
- **Export Capabilities**: Charts can be exported as images

## 📊 Data Requirements

The dashboard expects a CSV file with the following columns:
- `order_purchase_timestamp`: Date and time of order
- `customer_city`: City where the customer is located
- `product_category_name_english`: Product category name
- `total_price`: Total price of the order
- `price`: Individual product price
- `product_weight_g`: Product weight in grams
- `review_score`: Customer review score (optional)
- `order_id`: Unique order identifier

## 🚀 Performance Optimization

- **Data Caching**: Uses Streamlit's caching to improve performance
- **Efficient Filtering**: Optimized data filtering for large datasets
- **Lazy Loading**: Charts are generated only when needed
- **Memory Management**: Efficient memory usage for large datasets

## 🔮 Future Enhancements

- **Machine Learning Models**: Integration of more advanced ML models
- **Real-time Data**: Support for real-time data feeds
- **External Factors**: Integration of external data (weather, holidays, etc.)
- **API Integration**: REST API for programmatic access
- **Mobile Optimization**: Mobile-friendly interface
- **Multi-language Support**: Support for multiple languages

## 📞 Support

For technical support or feature requests, please refer to the project documentation or contact the development team.

---

**Note**: This dashboard is designed for educational and business intelligence purposes. Always validate forecasts and recommendations with domain experts before making business decisions. 