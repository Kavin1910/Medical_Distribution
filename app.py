import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Bangalore Medical Distribution",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Bangalore Medical Distribution (Task by Lokesh)")
st.markdown("""
This dashboard provides insights into medical product distribution across Bangalore regions.
Analyze sales trends, regional performance, and product category distributions.
""")

# Generate or load data
@st.cache_data
def generate_medical_distribution_dataset(n_samples=5000):
    """
    Generate a synthetic dataset for medical product distribution in Bangalore
    """
    # Define Bangalore regions/zones
    regions = ['East Bangalore', 'West Bangalore', 'North Bangalore', 'South Bangalore', 'Central Bangalore']
    region_population = {
        'East Bangalore': 1800000, 
        'West Bangalore': 1500000, 
        'North Bangalore': 1200000, 
        'South Bangalore': 2000000, 
        'Central Bangalore': 1000000
    }
    region_healthcare_density = {
        'East Bangalore': 12, 
        'West Bangalore': 15, 
        'North Bangalore': 10, 
        'South Bangalore': 18, 
        'Central Bangalore': 25
    }
    
    # Define distributors
    distributors = [
        {'id': 1, 'name': 'MediSure Distributors', 'years_in_business': 12, 'partners': 150, 'regions': ['East Bangalore', 'South Bangalore']},
        {'id': 2, 'name': 'Bangalore Medical Supplies', 'years_in_business': 8, 'partners': 90, 'regions': ['West Bangalore', 'North Bangalore']},
        {'id': 3, 'name': 'HealthFirst Distribution', 'years_in_business': 15, 'partners': 200, 'regions': ['Central Bangalore', 'East Bangalore']},
        {'id': 4, 'name': 'CarePlus Medical', 'years_in_business': 5, 'partners': 75, 'regions': ['South Bangalore', 'West Bangalore']},
        {'id': 5, 'name': 'Lifeline Healthcare', 'years_in_business': 20, 'partners': 250, 'regions': ['North Bangalore', 'Central Bangalore', 'East Bangalore']}
    ]
    
    # Define product categories and products
    product_categories = ['Antibiotics', 'Pain Relievers', 'Cardiovascular', 'Vitamins', 'Medical Devices', 'Diabetes Care']
    
    products = [
        {'id': 101, 'name': 'AmoxiCure 500mg', 'category': 'Antibiotics', 'base_price': 120, 'manufacturer': 'PharmaCorp', 'launch_date': '2020-01-15', 'popularity': 0.85},
        {'id': 102, 'name': 'Paracetol Plus', 'category': 'Pain Relievers', 'base_price': 45, 'manufacturer': 'HealthGen', 'launch_date': '2019-07-22', 'popularity': 0.92},
        {'id': 103, 'name': 'CardioShield 10mg', 'category': 'Cardiovascular', 'base_price': 280, 'manufacturer': 'MediLife', 'launch_date': '2021-03-10', 'popularity': 0.78},
        {'id': 104, 'name': 'VitaBoost Complex', 'category': 'Vitamins', 'base_price': 180, 'manufacturer': 'NutriHealth', 'launch_date': '2020-11-05', 'popularity': 0.65},
        {'id': 105, 'name': 'GlucoCheck Meter', 'category': 'Medical Devices', 'base_price': 1200, 'manufacturer': 'MedTech', 'launch_date': '2021-09-30', 'popularity': 0.72},
        {'id': 106, 'name': 'InsulinEase 50IU', 'category': 'Diabetes Care', 'base_price': 450, 'manufacturer': 'DiaCare', 'launch_date': '2022-01-18', 'popularity': 0.81},
        {'id': 107, 'name': 'AzithroMax 250mg', 'category': 'Antibiotics', 'base_price': 160, 'manufacturer': 'PharmaCorp', 'launch_date': '2021-05-12', 'popularity': 0.79},
        {'id': 108, 'name': 'IbuRelief 400mg', 'category': 'Pain Relievers', 'base_price': 60, 'manufacturer': 'HealthGen', 'launch_date': '2020-03-25', 'popularity': 0.88},
        {'id': 109, 'name': 'BP Monitor Pro', 'category': 'Medical Devices', 'base_price': 1800, 'manufacturer': 'MedTech', 'launch_date': '2022-04-10', 'popularity': 0.70},
        {'id': 110, 'name': 'MultiVit Daily', 'category': 'Vitamins', 'base_price': 220, 'manufacturer': 'NutriHealth', 'launch_date': '2021-12-05', 'popularity': 0.75}
    ]
    
    # Generate sales data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 4, 30)
    date_range = (end_date - start_date).days
    
    data = []
    
    for _ in range(n_samples):
        # Select random date
        random_days = np.random.randint(0, date_range)
        sale_date = start_date + timedelta(days=random_days)
        
        # Seasonal effects (higher sales in winter months for antibiotics, etc.)
        month = sale_date.month
        is_winter = 1 if month in [11, 12, 1, 2] else 0
        is_monsoon = 1 if month in [6, 7, 8, 9] else 0
        
        # Select random distributor
        distributor = np.random.choice(distributors)
        distributor_id = distributor['id']
        distributor_name = distributor['name']
        distributor_years = distributor['years_in_business']
        distributor_partners = distributor['partners']
        
        # Select random region from distributor's coverage
        region = np.random.choice(distributor['regions'])
        population = region_population[region]
        healthcare_density = region_healthcare_density[region]
        
        # Select random product
        product = np.random.choice(products)
        product_id = product['id']
        product_name = product['name']
        product_category = product['category']
        base_price = product['base_price']
        manufacturer = product['manufacturer']
        popularity = product['popularity']
        
        # Adjust probability based on season and product category
        prob_adjustment = 0
        if is_winter and product_category == 'Antibiotics':
            prob_adjustment = 0.2
        elif is_monsoon and product_category == 'Antibiotics':
            prob_adjustment = 0.15
        elif product_category == 'Vitamins' and month in [3, 4, 5]:  # Spring months
            prob_adjustment = 0.1
            
        # Determine quantity based on product popularity, region population, and seasonal factors
        base_quantity = int(np.random.exponential(scale=5) + 1)
        quantity = int(base_quantity * (1 + popularity/2) * (population/1000000) * (1 + prob_adjustment))
        
        # Calculate price with small random variations
        price_variation = np.random.normal(loc=1.0, scale=0.05)  # 5% standard deviation
        price = base_price * price_variation
        
        # Calculate total sales value
        sales_value = price * quantity
        
        # Add promotion effects (10% of sales have promotions)
        has_promotion = np.random.choice([0, 1], p=[0.9, 0.1])
        if has_promotion:
            discount_rate = np.random.uniform(0.05, 0.25)  # 5% to 25% discount
            discounted_price = price * (1 - discount_rate)
            sales_value = discounted_price * quantity
        else:
            discount_rate = 0
            discounted_price = price
        
        # Create record
        record = {
            'date': sale_date.strftime('%Y-%m-%d'),
            'distributor_id': distributor_id,
            'distributor_name': distributor_name,
            'distributor_years': distributor_years,
            'distributor_partners': distributor_partners,
            'region': region,
            'region_population': population,
            'healthcare_density': healthcare_density,
            'product_id': product_id,
            'product_name': product_name,
            'product_category': product_category,
            'manufacturer': manufacturer,
            'quantity': quantity,
            'base_price': base_price,
            'actual_price': price,
            'discount_rate': discount_rate,
            'discounted_price': discounted_price,
            'sales_value': sales_value,
            'is_winter': is_winter,
            'is_monsoon': is_monsoon,
            'month': month,
            'year': sale_date.year,
            'day_of_week': sale_date.weekday(),
            'quarter': (month-1)//3 + 1
        }
        
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Create a sidebar for application navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Regional Analysis", "Time Series Analysis", "Product Analysis", "Predictive Models"])

# Generate data
np.random.seed(42)  # For reproducibility
df = generate_medical_distribution_dataset(10000)
df['date'] = pd.to_datetime(df['date'])

# Process data for analysis
@st.cache_data
def process_data(df):
    # Time series data
    time_series_data = df.groupby(['date', 'product_category'])['sales_value'].sum().reset_index()
    time_series_pivot = time_series_data.pivot(index='date', columns='product_category', values='sales_value').fillna(0)
    time_series_pivot['Total'] = time_series_pivot.sum(axis=1)
    
    # Monthly aggregation
    monthly_sales = time_series_pivot.resample('M').sum()
    
    # Regional data
    regional_data = df.groupby(['region', 'product_category'])['sales_value'].sum().reset_index()
    
    # Product data
    product_data = df.groupby(['product_id', 'product_name', 'product_category']).agg({
        'sales_value': 'sum',
        'quantity': 'sum',
        'base_price': 'first',
        'manufacturer': 'first'
    }).reset_index()
    
    # Distributor data
    distributor_data = df.groupby(['distributor_id', 'distributor_name']).agg({
        'sales_value': 'sum',
        'quantity': 'sum',
        'distributor_years': 'first',
        'distributor_partners': 'first'
    }).reset_index()
    
    return {
        'time_series_data': time_series_data,
        'time_series_pivot': time_series_pivot,
        'monthly_sales': monthly_sales,
        'regional_data': regional_data,
        'product_data': product_data,
        'distributor_data': distributor_data
    }

processed_data = process_data(df)
monthly_sales = processed_data['monthly_sales']
regional_data = processed_data['regional_data']
product_data = processed_data['product_data']
distributor_data = processed_data['distributor_data']

# ======= OVERVIEW PAGE =======
if page == "Overview":
    st.header("Dashboard Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"₹{df['sales_value'].sum():,.0f}")
    
    with col2:
        st.metric("Total Products Sold", f"{df['quantity'].sum():,.0f}")
    
    with col3:
        top_region = regional_data.groupby('region')['sales_value'].sum().idxmax()
        st.metric("Top Region", top_region)
    
    with col4:
        top_category = product_data.groupby('product_category')['sales_value'].sum().idxmax()
        st.metric("Top Category", top_category)
    
    # Sales trend chart
    st.subheader("Monthly Sales Trend")
    fig = px.line(monthly_sales.reset_index(), x='date', y='Total', 
                  title='Overall Monthly Sales Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional distribution
    st.subheader("Sales by Region")
    region_summary = df.groupby('region')['sales_value'].sum().reset_index()
    fig = px.pie(region_summary, values='sales_value', names='region', 
                 title='Sales Distribution by Region')
    st.plotly_chart(fig, use_container_width=True)
    
    # Product category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Product Category")
        cat_summary = df.groupby('product_category')['sales_value'].sum().reset_index()
        fig = px.bar(cat_summary, x='product_category', y='sales_value', 
                    title='Sales by Product Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 5 Products")
        top_products = product_data.sort_values('sales_value', ascending=False).head(5)
        fig = px.bar(top_products, x='product_name', y='sales_value', 
                    color='product_category', title='Top 5 Products by Sales')
        st.plotly_chart(fig, use_container_width=True)
    
    # Distributor performance
    st.subheader("Distributor Performance")
    fig = px.bar(distributor_data, x='distributor_name', y='sales_value',
                 title='Sales by Distributor', text_auto='.2s')
    fig.update_layout(xaxis_title="Distributor", yaxis_title="Total Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)

# ======= REGIONAL ANALYSIS PAGE =======
elif page == "Regional Analysis":
    st.header("Regional Sales Analysis")
    
    # Region selector
    region_filter = st.multiselect("Select Regions", 
                                  options=df['region'].unique(),
                                  default=df['region'].unique())
    
    filtered_df = df[df['region'].isin(region_filter)]
    
    # Regional sales map for Bangalore
    st.subheader("Bangalore Region Sales Heatmap")
    
    # Since we don't have exact geo coordinates, we'll use a simplified representation
    region_sales = filtered_df.groupby('region')['sales_value'].sum().reset_index()
    
    # Create a regional comparision chart
    fig = px.bar(region_sales, x='region', y='sales_value', 
                 title='Total Sales by Region', 
                 color='sales_value', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title="Region", yaxis_title="Total Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional performance by product category
    st.subheader("Product Category Performance by Region")
    
    region_cat_sales = filtered_df.groupby(['region', 'product_category'])['sales_value'].sum().reset_index()
    fig = px.bar(region_cat_sales, x='region', y='sales_value', color='product_category',
                 title='Sales by Region and Product Category',
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Healthcare density vs sales scatter plot
    st.subheader("Healthcare Density vs Sales")
    
    region_summary = filtered_df.groupby('region').agg({
        'sales_value': 'sum',
        'healthcare_density': 'first',
        'region_population': 'first'
    }).reset_index()
    
    fig = px.scatter(region_summary, x='healthcare_density', y='sales_value', 
                    size='region_population', color='region',
                    title='Healthcare Density vs Sales by Region',
                    size_max=60)
    fig.update_layout(xaxis_title="Healthcare Facilities per 10,000 People", 
                      yaxis_title="Total Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Population vs sales
    st.subheader("Population vs Sales")
    fig = px.scatter(region_summary, x='region_population', y='sales_value', 
                    color='region', size='healthcare_density', size_max=50,
                    title='Population vs Sales by Region')
    fig.update_layout(xaxis_title="Population", yaxis_title="Total Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)

# ======= TIME SERIES ANALYSIS PAGE =======
elif page == "Time Series Analysis":
    st.header("Time Series Analysis")
    
    # Category selector for time series
    categories = list(monthly_sales.columns)
    categories.remove('Total')  # Remove Total to show it separately
    selected_categories = st.multiselect("Select Product Categories", 
                                        options=categories,
                                        default=["Antibiotics"])
    
    # Time period selector
    date_range = st.date_input(
        "Select Date Range",
        [monthly_sales.index.min(), monthly_sales.index.max()],
        min_value=monthly_sales.index.min(),
        max_value=monthly_sales.index.max()
    )
    
    # Filter time series data
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_monthly = monthly_sales.loc[start_date:end_date]
    else:
        filtered_monthly = monthly_sales
    
    # Plot total sales trend
    st.subheader("Total Sales Time Series")
    fig = px.line(filtered_monthly.reset_index(), x='date', y='Total',
                 title='Total Monthly Sales Trend')
    fig.update_layout(xaxis_title="Date", yaxis_title="Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot selected categories
    st.subheader("Category Sales Trends")
    
    fig = go.Figure()
    for category in selected_categories:
        fig.add_trace(go.Scatter(
            x=filtered_monthly.index, 
            y=filtered_monthly[category],
            mode='lines',
            name=category
        ))
    
    fig.update_layout(
        title="Monthly Sales Trends by Product Category",
        xaxis_title="Date",
        yaxis_title="Sales (₹)",
        legend_title="Product Category"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    if len(selected_categories) > 0:
        st.subheader("Seasonal Patterns")
        
        # Group by month to see seasonal patterns
        selected_cat = selected_categories[0]  # Use first selected category
        monthly_pattern = df[df['product_category'] == selected_cat].groupby('month')['sales_value'].mean().reset_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pattern['month_name'] = monthly_pattern['month'].apply(lambda x: months[x-1])
        
        fig = px.line(monthly_pattern, x='month_name', y='sales_value',
                     title=f'Monthly Sales Pattern for {selected_cat}',
                     markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Average Sales (₹)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series forecast for selected category
        st.subheader(f"Sales Forecast for {selected_cat}")
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Get category data and prepare for forecasting
                category_data = monthly_sales[selected_cat].dropna()
                
                # Split data for training
                train_size = int(len(category_data) * 0.8)
                train_data = category_data[:train_size]
                test_data = category_data[train_size:]
                
                # Fit SARIMA model
                try:
                    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    results = model.fit(disp=False)
                    
                    # Forecast
                    forecast_steps = len(test_data)
                    future_steps = 3  # Additional months to forecast
                    total_steps = forecast_steps + future_steps
                    
                    forecast = results.get_forecast(steps=total_steps)
                    forecast_mean = forecast.predicted_mean
                    conf_int = forecast.conf_int()
                    
                    # Plot results
                    fig = go.Figure()
                    
                    # Training data
                    fig.add_trace(go.Scatter(
                        x=train_data.index,
                        y=train_data,
                        mode='lines',
                        name='Training Data'
                    ))
                    
                    # Test data
                    fig.add_trace(go.Scatter(
                        x=test_data.index,
                        y=test_data,
                        mode='lines',
                        name='Actual Sales'
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_mean.index[:forecast_steps],
                        y=forecast_mean[:forecast_steps],
                        mode='lines',
                        name='Forecasted Sales',
                        line=dict(color='red')
                    ))
                    
                    # Future forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_mean.index[forecast_steps:],
                        y=forecast_mean[forecast_steps:],
                        mode='lines',
                        name='Future Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Confidence intervals for future forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_mean.index[forecast_steps:].tolist() + forecast_mean.index[forecast_steps:][::-1].tolist(),
                        y=conf_int[f'upper {selected_cat}'][forecast_steps:].tolist() + 
                           conf_int[f'lower {selected_cat}'][forecast_steps:][::-1].tolist(),
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.1)',
                        line=dict(color='rgba(255,0,0,0)'),
                        name='95% Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f'Sales Forecast for {selected_cat}',
                        xaxis_title='Date',
                        yaxis_title='Sales (₹)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast metrics
                    mae = mean_absolute_error(test_data, forecast_mean[:forecast_steps])
                    rmse = np.sqrt(mean_squared_error(test_data, forecast_mean[:forecast_steps]))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Absolute Error", f"₹{mae:.2f}")
                    with col2:
                        st.metric("Root Mean Squared Error", f"₹{rmse:.2f}")
                        
                    # Future predictions
                    st.subheader("Future Sales Predictions")
                    future_dates = forecast_mean.index[forecast_steps:].strftime('%B %Y').tolist()
                    future_values = forecast_mean[forecast_steps:].tolist()
                    
                    future_df = pd.DataFrame({
                        'Month': future_dates,
                        'Predicted Sales': [f"₹{val:.2f}" for val in future_values]
                    })
                    
                    st.table(future_df)
                    
                except Exception as e:
                    st.error(f"Error in forecast: {e}")

# ======= PRODUCT ANALYSIS PAGE =======
elif page == "Product Analysis":
    st.header("Product Analysis")
    
    # Product category filter
    cat_filter = st.multiselect("Select Product Categories", 
                               options=df['product_category'].unique(),
                               default=df['product_category'].unique())
    
    filtered_df = df[df['product_category'].isin(cat_filter)]
    
    # Product performance overview
    st.subheader("Product Sales Overview")
    
    product_sales = filtered_df.groupby(['product_name', 'product_category', 'manufacturer'])['sales_value'].sum().reset_index()
    product_sales = product_sales.sort_values('sales_value', ascending=False)
    
    fig = px.bar(product_sales, x='product_name', y='sales_value', 
                color='product_category',
                title='Product Sales Ranking',
                hover_data=['manufacturer'])
    fig.update_layout(xaxis_title="Product", yaxis_title="Total Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Market share by category
    st.subheader("Market Share by Product Category")
    
    cat_sales = filtered_df.groupby('product_category')['sales_value'].sum().reset_index()
    total_sales = cat_sales['sales_value'].sum()
    cat_sales['market_share'] = cat_sales['sales_value'] / total_sales * 100
    
    fig = px.pie(cat_sales, values='market_share', names='product_category',
                title='Market Share by Product Category')
    st.plotly_chart(fig, use_container_width=True)
    
    # Manufacturer performance
    st.subheader("Manufacturer Performance")
    
    manuf_sales = filtered_df.groupby('manufacturer')['sales_value'].sum().reset_index()
    manuf_sales = manuf_sales.sort_values('sales_value', ascending=False)
    
    fig = px.bar(manuf_sales, x='manufacturer', y='sales_value',
                title='Sales by Manufacturer')
    fig.update_layout(xaxis_title="Manufacturer", yaxis_title="Total Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Price vs quantity analysis
    st.subheader("Price vs Quantity Analysis")
    
    product_price_qty = filtered_df.groupby(['product_name', 'product_category']).agg({
        'base_price': 'mean',
        'quantity': 'sum',
        'sales_value': 'sum'
    }).reset_index()
    
    fig = px.scatter(product_price_qty, x='base_price', y='quantity',
                    size='sales_value', color='product_category',
                    hover_name='product_name',
                    title='Price vs Quantity by Product',
                    size_max=50)
    fig.update_layout(xaxis_title="Base Price (₹)", yaxis_title="Total Quantity Sold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top product in each category
    st.subheader("Top Product in Each Category")
    
    top_by_cat = filtered_df.groupby('product_category').apply(
        lambda x: x.sort_values('sales_value', ascending=False).iloc[0]
    )[['product_name', 'sales_value', 'quantity']].reset_index()
    
    fig = px.bar(top_by_cat, x='product_category', y='sales_value',
                color='product_category',
                title='Top Selling Product by Category',
                text='product_name')
    fig.update_layout(xaxis_title="Category", yaxis_title="Total Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)

# ======= PREDICTIVE MODELS PAGE =======
elif page == "Predictive Models":
    st.header("Predictive Models & Insights")
    
    model_type = st.selectbox("Select Model Type", 
                             ["Regional Sales Prediction", "Product Sales Forecast", "What-If Analysis"])
    
    if model_type == "Regional Sales Prediction":
        st.subheader("Regional Sales Prediction Model")
        
        # Prepare data for regional sales prediction
        @st.cache_data
        def prepare_regional_model_data():
            region_product_features = df.groupby(['region', 'product_category']).agg({
                'sales_value': 'sum',
                'quantity': 'sum',
                'region_population': 'first',
                'healthcare_density': 'first',
                'distributor_years': 'mean',
                'distributor_partners': 'mean',
                'is_winter': 'mean',
                'is_monsoon': 'mean'
            }).reset_index()
            
            # Create dummy variables for categorical features
            region_dummies = pd.get_dummies(region_product_features['region'], prefix='region')
            product_dummies = pd.get_dummies(region_product_features['product_category'], prefix='category')
            
            # Combine with original features
            X = pd.concat([
                region_product_features.drop(['region', 'product_category', 'sales_value'], axis=1),
                region_dummies,
                product_dummies
            ], axis=1)
            
            y = region_product_features['sales_value']
            
            return X, y, region_product_features
        
        X, y, region_features = prepare_regional_model_data()
        
        # Train and evaluate model on click
        if st.button("Train Regional Sales Model"):
            with st.spinner("Training model..."):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = rf_model.predict(X_test)
                
                # Evaluate model
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"₹{mae:.2f}")
                with col2:
                    st.metric("Root Mean Squared Error", f"₹{rmse:.2f}")
                with col3:
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("Feature Importance")
                top_features = feature_importance.head(10)
                
                fig = px.bar(top_features, x='Importance', y='Feature', 
                            orientation='h', title='Top 10 Important Features')
                st.plotly_chart(fig, use_container_width=True)
                
                # Actual vs Predicted visualization
                prediction_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                
                fig = px.scatter(prediction_df, x='Actual', y='Predicted',
                                title='Actual vs Predicted Sales')
                fig.add_trace(
                    go.Scatter(x=[y_test.min(), y_test.max()], 
                              y=[y_test.min(), y_test.max()],
                              mode='lines', name='Perfect Prediction',
                              line=dict(color='red', dash='dash'))
                )
                fig.update_layout(xaxis_title="Actual Sales (₹)", 
                                 yaxis_title="Predicted Sales (₹)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Regional prediction tool
        st.subheader("Regional Sales Prediction Tool")
        st.write("Adjust the parameters below to predict sales for specific regions and product categories.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_region = st.selectbox("Select Region", df['region'].unique())
            selected_category = st.selectbox("Select Product Category", df['product_category'].unique())
        
        with col2:
            # Get region-specific data
            region_data = df[df['region'] == selected_region].iloc[0]
            region_population = region_data['region_population']
            healthcare_density = region_data['healthcare_density']
            
            population_factor = st.slider("Population Factor", 0.5, 1.5, 1.0, 0.1, 
                                         help="Adjust population impact (1.0 = current population)")
            healthcare_factor = st.slider("Healthcare Density Factor", 0.5, 1.5, 1.0, 0.1,
                                        help="Adjust healthcare facility impact")
        
        # Season selection
        is_winter = st.checkbox("Winter Season")
        is_monsoon = st.checkbox("Monsoon Season")
        
        if st.button("Predict Regional Sales"):
            with st.spinner("Generating prediction..."):
                # Train the model on full dataset
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                
                # Create input for prediction
                input_data = pd.DataFrame({
                    'quantity': [region_features[(region_features['region'] == selected_region) & 
                                               (region_features['product_category'] == selected_category)]['quantity'].values[0]],
                    'region_population': [region_population * population_factor],
                    'healthcare_density': [healthcare_density * healthcare_factor],
                    'distributor_years': [df[df['region'] == selected_region]['distributor_years'].mean()],
                    'distributor_partners': [df[df['region'] == selected_region]['distributor_partners'].mean()],
                    'is_winter': [1 if is_winter else 0],
                    'is_monsoon': [1 if is_monsoon else 0]
                })
                
                # Add dummy columns
                for region in df['region'].unique():
                    input_data[f'region_{region}'] = 1 if region == selected_region else 0
                
                for category in df['product_category'].unique():
                    input_data[f'category_{category}'] = 1 if category == selected_category else 0
                
                # Ensure columns match training data
                missing_cols = set(X.columns) - set(input_data.columns)
                for col in missing_cols:
                    input_data[col] = 0
                
                input_data = input_data[X.columns]
                
                # Make prediction
                prediction = rf_model.predict(input_data)[0]
                
                # Get baseline for comparison
                baseline = region_features[(region_features['region'] == selected_region) & 
                                         (region_features['product_category'] == selected_category)]['sales_value'].values[0]
                
                # Display prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Sales", f"₹{prediction:.2f}")
                with col2:
                    percent_change = ((prediction - baseline) / baseline) * 100
                    st.metric("Compared to Average", f"{percent_change:+.2f}%", 
                             delta=f"{percent_change:+.2f}%")
                
                # Context
                st.write(f"This prediction is for {selected_category} sales in {selected_region}.")
                if is_winter:
                    st.write("Winter season typically increases sales for Antibiotics and some other categories.")
                if is_monsoon:
                    st.write("Monsoon season often affects distribution patterns and can increase demand for certain medications.")
    
    elif model_type == "Product Sales Forecast":
        st.subheader("Product Sales Forecasting")
        
        # Select product for forecasting
        selected_product = st.selectbox("Select Product", 
                                       options=df['product_name'].unique())
        
        if st.button("Generate Product Forecast"):
            with st.spinner("Analyzing product trends..."):
                # Filter data for selected product
                product_data = df[df['product_name'] == selected_product]
                
                # Aggregate by month
                product_ts = product_data.groupby(pd.Grouper(key='date', freq='M'))['sales_value'].sum()
                
                # Fill any missing months with interpolation
                full_range = pd.date_range(start=product_ts.index.min(), end=product_ts.index.max(), freq='M')
                product_ts = product_ts.reindex(full_range).interpolate()
                
                # Train-test split for time series
                train_size = int(len(product_ts) * 0.8)
                train_data = product_ts[:train_size]
                test_data = product_ts[train_size:]
                
                # Create and plot forecast
                try:
                    # Simple forecasting with SARIMAX
                    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    results = model.fit(disp=False)
                    
                    # Generate forecasts
                    forecast_steps = len(test_data)
                    future_steps = 3  # Forecast 3 months into the future
                    total_steps = forecast_steps + future_steps
                    
                    forecast = results.get_forecast(steps=total_steps)
                    forecast_mean = forecast.predicted_mean
                    conf_int = forecast.conf_int()
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Training data
                    fig.add_trace(go.Scatter(
                        x=train_data.index,
                        y=train_data,
                        mode='lines',
                        name='Historical Data'
                    ))
                    
                    # Test data
                    fig.add_trace(go.Scatter(
                        x=test_data.index,
                        y=test_data,
                        mode='lines',
                        name='Actual Sales'
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_mean.index[:forecast_steps],
                        y=forecast_mean[:forecast_steps],
                        mode='lines',
                        name='Forecasted Sales',
                        line=dict(color='green')
                    ))
                    
                    # Future forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_mean.index[forecast_steps:],
                        y=forecast_mean[forecast_steps:],
                        mode='lines',
                        name='Future Forecast',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    # Confidence intervals
                    fig.add_trace(go.Scatter(
                        x=forecast_mean.index.tolist() + forecast_mean.index[::-1].tolist(),
                        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[::-1, 1].tolist(),
                        fill='toself',
                        fillcolor='rgba(0,176,0,0.1)',
                        line=dict(color='rgba(0,176,0,0)'),
                        name='95% Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f'Sales Forecast for {selected_product}',
                        xaxis_title='Month',
                        yaxis_title='Sales (₹)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    mae = mean_absolute_error(test_data, forecast_mean[:forecast_steps])
                    rmse = np.sqrt(mean_squared_error(test_data, forecast_mean[:forecast_steps]))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Absolute Error", f"₹{mae:.2f}")
                    with col2:
                        st.metric("Root Mean Squared Error", f"₹{rmse:.2f}")
                    
                    # Future prediction table
                    st.subheader("Future Sales Predictions")
                    future_dates = forecast_mean.index[forecast_steps:].strftime('%B %Y').tolist()
                    future_values = forecast_mean[forecast_steps:].tolist()
                    
                    future_df = pd.DataFrame({
                        'Month': future_dates,
                        'Predicted Sales': [f"₹{val:.2f}" for val in future_values]
                    })
                    
                    st.table(future_df)
                    
                    # Product statistics
                    st.subheader("Product Statistics")
                    product_stats = product_data.agg({
                        'sales_value': ['sum', 'mean', 'std'],
                        'quantity': ['sum', 'mean', 'std'],
                        'actual_price': ['mean', 'min', 'max'],
                        'discount_rate': 'mean'
                    })
                    
                    st.write(f"Total {selected_product} Sales: ₹{product_stats['sales_value']['sum']:.2f}")
                    st.write(f"Average Monthly Sales: ₹{product_stats['sales_value']['mean']:.2f}")
                    st.write(f"Total Quantity Sold: {product_stats['quantity']['sum']:.0f} units")
                    st.write(f"Average Price: ₹{product_stats['actual_price']['mean']:.2f}")
                    st.write(f"Average Discount Rate: {product_stats['discount_rate']['mean']*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error in forecast: {e}")
                    st.write("Not enough data for reliable forecasting. Try selecting a different product.")
    
    elif model_type == "What-If Analysis":
        st.subheader("What-If Analysis Tool")
        st.write("Explore how changes in various factors might affect sales performance.")
        
        # Parameters for what-if analysis
        analysis_type = st.radio("Analysis Type", ["Pricing Strategy", "Distributor Strategy", "Seasonal Effects"])
        
        if analysis_type == "Pricing Strategy":
            st.write("Analyze how price changes might impact sales volume and total revenue.")
            
            # Select product category
            selected_category = st.selectbox("Select Product Category", 
                                          options=df['product_category'].unique())
            
            # Get average price and elasticity estimate for this category
            category_data = df[df['product_category'] == selected_category]
            avg_price = category_data['actual_price'].mean()
            
            # Simplified price elasticity calculation
            elasticity = -1.5  # Default assumption for medical products
            
            # Price adjustment slider
            price_change = st.slider("Price Change Percentage", -50, 50, 0, 5)
            
            # Calculate expected impact
            new_price = avg_price * (1 + price_change/100)
            expected_quantity_change = price_change * elasticity / 100
            
            current_avg_quantity = category_data['quantity'].mean()
            expected_new_quantity = current_avg_quantity * (1 + expected_quantity_change)
            
            current_revenue = avg_price * current_avg_quantity
            expected_revenue = new_price * expected_new_quantity
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price Change", f"{price_change:+}%")
                st.write(f"New Avg Price: ₹{new_price:.2f}")
            
            with col2:
                quantity_change_pct = (expected_new_quantity / current_avg_quantity - 1) * 100
                st.metric("Est. Quantity Change", f"{quantity_change_pct:.1f}%", 
                         delta=f"{quantity_change_pct:.1f}%")
            
            with col3:
                revenue_change_pct = (expected_revenue / current_revenue - 1) * 100
                st.metric("Est. Revenue Change", f"{revenue_change_pct:.1f}%", 
                         delta=f"{revenue_change_pct:.1f}%")
            
            # Plot different scenarios
            price_changes = np.arange(-50, 51, 5)
            results = []
            for pc in price_changes:
                new_p = avg_price * (1 + pc/100)
                expected_q_change = pc * elasticity / 100
                new_q = current_avg_quantity * (1 + expected_q_change)
                new_rev = new_p * new_q
                rev_change = (new_rev / current_revenue - 1) * 100
                results.append({
                    'Price Change': pc,
                    'Revenue Change': rev_change
                })
            
            results_df = pd.DataFrame(results)
            
            fig = px.line(results_df, x='Price Change', y='Revenue Change',
                         title='Estimated Revenue Impact of Price Changes',
                         markers=True)
            fig.update_layout(xaxis_title="Price Change (%)", 
                             yaxis_title="Revenue Change (%)")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Analysis")
            if revenue_change_pct > 0:
                st.write(f"The analysis suggests a price {'increase' if price_change > 0 else 'decrease'} of {abs(price_change)}% could potentially increase revenue by {revenue_change_pct:.1f}%.")
            else:
                st.write(f"The price adjustment of {price_change}% may result in a revenue decrease of {abs(revenue_change_pct):.1f}%.")
            
            st.write("**Note:** This is based on a simplified price elasticity model. Actual results may vary based on market conditions, competition, and customer behavior.")
        
        elif analysis_type == "Distributor Strategy":
            st.write("Explore performance across different distributors and regions.")
            
            # Distributor comparison
            dist_data = df.groupby(['distributor_name', 'distributor_years', 'distributor_partners']).agg({
                'sales_value': 'sum',
                'quantity': 'sum'
            }).reset_index()
            
            fig = px.scatter(dist_data, x='distributor_years', y='distributor_partners',
                            size='sales_value', color='distributor_name',
                            title='Distributor Performance by Experience and Partner Network',
                            hover_data=['sales_value'])
            fig.update_layout(xaxis_title="Years in Business",
                             yaxis_title="Number of Partners")
            st.plotly_chart(fig, use_container_width=True)
            
            # Distributor-Region coverage analysis
            st.subheader("Distributor Coverage Analysis")
            
            # Create a coverage matrix
            coverage_data = df.groupby(['distributor_name', 'region'])['sales_value'].sum().reset_index()
            coverage_pivot = coverage_data.pivot(index='distributor_name', columns='region', values='sales_value')
            
            # Normalize for heatmap
            normalized_coverage = coverage_pivot.div(coverage_pivot.max(axis=1), axis=0)
            
            # Convert to long format for plotly
            coverage_long = normalized_coverage.reset_index().melt(id_vars=['distributor_name'], 
                                                                 var_name='region', 
                                                                 value_name='relative_sales')
            
            fig = px.density_heatmap(coverage_long, x='region', y='distributor_name', z='relative_sales',
                                   title='Distributor-Region Coverage Map',
                                   color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Coverage Gap Analysis")
            
            # Find potential coverage gaps
            low_coverage = coverage_long[coverage_long['relative_sales'] < 0.3]
            if not low_coverage.empty:
                st.write("Potential coverage gaps (regions with low relative sales performance):")
                for _, row in low_coverage.iterrows():
                    st.write(f"- {row['distributor_name']} in {row['region']}")
            else:
                st.write("No significant coverage gaps detected.")
            
            # Find potential partnership opportunities
            st.write("**Recommendation:** Consider strengthening distributor partnerships in underserved regions to improve coverage and sales performance.")
        
        elif analysis_type == "Seasonal Effects":
            st.write("Analyze how seasonal factors affect different product categories.")
            
            # Seasonal analysis
            seasonal_data = df.groupby(['month', 'product_category'])[['sales_value', 'quantity']].mean().reset_index()
            
            # Convert month numbers to names
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            seasonal_data['month_name'] = seasonal_data['month'].map(month_names)
            
            # Sort by month
            seasonal_data['month_order'] = seasonal_data['month']
            seasonal_data = seasonal_data.sort_values('month_order')
            
            # Plot seasonal patterns
            selected_categories = st.multiselect("Select Categories to Compare",
                                              options=df['product_category'].unique(),
                                              default=['Antibiotics', 'Vitamins'])
            
            if selected_categories:
                filtered_seasonal = seasonal_data[seasonal_data['product_category'].isin(selected_categories)]
                
                fig = px.line(filtered_seasonal, x='month_name', y='sales_value',
                             color='product_category', markers=True,
                             title='Seasonal Sales Patterns by Product Category')
                fig.update_layout(xaxis_title="Month",
                                 yaxis_title="Average Sales (₹)",
                                 xaxis={'categoryorder': 'array',
                                       'categoryarray': list(month_names.values())})
                st.plotly_chart(fig, use_container_width=True)
                
                # Quantity patterns
                fig = px.line(filtered_seasonal, x='month_name', y='quantity',
                             color='product_category', markers=True,
                             title='Seasonal Quantity Patterns by Product Category')
                fig.update_layout(xaxis_title="Month",
                                 yaxis_title="Average Quantity",
                                 xaxis={'categoryorder': 'array',
                                       'categoryarray': list(month_names.values())})
                st.plotly_chart(fig, use_container_width=True)
                
                # Peak analysis
                st.subheader("Seasonal Peak Analysis")
                
                for category in selected_categories:
                    cat_data = seasonal_data[seasonal_data['product_category'] == category]
                    peak_month_idx = cat_data['sales_value'].idxmax()
                    peak_month = cat_data.loc[peak_month_idx, 'month_name']
                    peak_value = cat_data.loc[peak_month_idx, 'sales_value']
                    
                    low_month_idx = cat_data['sales_value'].idxmin()
                    low_month = cat_data.loc[low_month_idx, 'month_name']
                    low_value = cat_data.loc[low_month_idx, 'sales_value']
                    
                    seasonality = (peak_value - low_value) / low_value * 100
                    
                    st.write(f"**{category}**")
                    st.write(f"- Peak month: {peak_month} (₹{peak_value:.2f})")
                    st.write(f"- Lowest month: {low_month} (₹{low_value:.2f})")
                    st.write(f"- Seasonality factor: {seasonality:.1f}% variation between peak and trough")
                
                # Recommendations
                st.subheader("Seasonal Strategy Recommendations")
                
                for category in selected_categories:
                    cat_data = seasonal_data[seasonal_data['product_category'] == category]
                    peak_month_idx = cat_data['sales_value'].idxmax()
                    peak_month = cat_data.loc[peak_month_idx, 'month_name']
                    peak_month_num = cat_data.loc[peak_month_idx, 'month']
                    
                    # Inventory planning months (3 months before peak)
                    planning_month_num = (peak_month_num - 3) % 12
                    if planning_month_num == 0:
                        planning_month_num = 12
                    planning_month = month_names[planning_month_num]
                    
                    st.write(f"**{category}**")
                    st.write(f"- Increase inventory planning by {planning_month} for {peak_month} peak season")
                    
                    # Pricing strategy
                    off_peak_months = cat_data.sort_values('sales_value').head(3)['month_name'].tolist()
                    st.write(f"- Consider promotional campaigns during off-peak months ({', '.join(off_peak_months)})")
                    
                    # Seasonal adjustments
                    if category == 'Antibiotics':
                        if 'Nov' in peak_month or 'Dec' in peak_month or 'Jan' in peak_month or 'Feb' in peak_month:
                            st.write("- Winter seasonality detected: Plan for increased demand during winter months")
                    
                    if category == 'Vitamins':
                        if 'Mar' in peak_month or 'Apr' in peak_month or 'May' in peak_month:
                            st.write("- Spring seasonality detected: Target health and wellness messaging in spring")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard analyzes medical product distribution data across Bangalore regions. "
    "It provides insights into regional performance, product trends, and predictive analytics."
)
st.sidebar.markdown("### Data Source")
st.sidebar.info(
    "Data represents synthetic medical product distribution across Bangalore regions "
    "from January 2022 to April 2024."
)

# App settings in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
color_theme = st.sidebar.selectbox("Color Theme", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])
