import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Procurement Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insights-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the NSN dataset"""
    try:
        # Try different methods to read the file
        
        # Method 1: Try Excel with openpyxl engine
        try:
            df = pd.read_excel('data/nsn_data.csv', engine='openpyxl')
            st.success("Loaded as Excel file with openpyxl engine")
        except:
            # Method 2: Try Excel with xlrd engine
            try:
                df = pd.read_excel('data/nsn_data.csv', engine='xlrd')
                st.success("Loaded as Excel file with xlrd engine")
            except:
                # Method 3: Try as CSV with different encodings
                try:
                    df = pd.read_csv('data/nsn_data.csv', encoding='utf-8')
                    st.success("Loaded as CSV file with UTF-8 encoding")
                except:
                    # Method 4: Try CSV with latin-1 encoding
                    df = pd.read_csv('data/nsn_data.csv', encoding='latin-1')
                    st.success("Loaded as CSV file with latin-1 encoding")
        
        # Clean and prepare data
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        df['Price_Log'] = np.log1p(df['Price'])  # Log transformation for better visualization
        
        st.info(f"Successfully loaded {len(df)} records")
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please check that the data file exists and is in the correct format")
        return pd.DataFrame()

def create_summary_metrics(df):
    """Create summary metrics for the dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Items",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        avg_price = df['Price'].mean()
        st.metric(
            label="Average Price",
            value=f"${avg_price:,.2f}",
            delta=None
        )
    
    with col3:
        total_value = df['Price'].sum()
        st.metric(
            label="Total Catalog Value",
            value=f"${total_value:,.2f}",
            delta=None
        )
    
    with col4:
        unique_categories = df['common_name'].nunique()
        st.metric(
            label="Product Categories",
            value=f"{unique_categories:,}",
            delta=None
        )

def create_price_distribution_chart(df):
    """Create price distribution visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Price Distribution (Log Scale)', 'Top 20 Most Expensive Items'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price distribution histogram
    fig.add_trace(
        go.Histogram(x=df['Price_Log'], nbinsx=50, name="Price Distribution"),
        row=1, col=1
    )
    
    # Top expensive items
    top_expensive = df.nlargest(20, 'Price')
    fig.add_trace(
        go.Bar(
            x=top_expensive['common_name'],
            y=top_expensive['Price'],
            name="Top Expensive Items",
            text=top_expensive['Price'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Log(Price + 1)", row=1, col=1)
    fig.update_xaxes(title_text="Product Category", tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=2)
    
    return fig

def create_category_analysis(df):
    """Create category-wise analysis"""
    # Category summary
    category_stats = df.groupby('common_name').agg({
        'Price': ['count', 'mean', 'sum'],
        'NSN': 'count'
    }).round(2)
    
    category_stats.columns = ['Item_Count', 'Avg_Price', 'Total_Value', 'NSN_Count']
    category_stats = category_stats.sort_values('Total_Value', ascending=False).head(15)
    
    # Create visualization
    fig = px.treemap(
        category_stats.reset_index(),
        path=['common_name'],
        values='Total_Value',
        color='Avg_Price',
        title="Product Categories by Total Value (Size) and Average Price (Color)",
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=600)
    return fig, category_stats

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">📊 Procurement Optimizer Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading procurement data..."):
        df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check the data file path.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=float(df['Price'].min()),
        max_value=float(df['Price'].max()),
        value=(float(df['Price'].min()), float(df['Price'].quantile(0.95))),
        format="$%.2f"
    )
    
    # Category filter
    categories = st.sidebar.multiselect(
        "Product Categories",
        options=sorted(df['common_name'].unique()),
        default=[]
    )
    
    # Unit filter
    units = st.sidebar.multiselect(
        "Unit of Issue",
        options=sorted(df['UI'].unique()),
        default=[]
    )
    
    # Apply filters
    filtered_df = df[
        (df['Price'] >= price_range[0]) & 
        (df['Price'] <= price_range[1])
    ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['common_name'].isin(categories)]
    
    if units:
        filtered_df = filtered_df[filtered_df['UI'].isin(units)]
    
    # Summary metrics
    st.markdown("## 📈 Key Metrics")
    create_summary_metrics(filtered_df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏷️ Categories", "🔍 Item Search", "💡 Insights"])
    
    with tab1:
        st.markdown("### Price Analysis")
        price_chart = create_price_distribution_chart(filtered_df)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Unit analysis
        st.markdown("### Analysis by Unit of Issue")
        unit_analysis = filtered_df.groupby('UI').agg({
            'Price': ['count', 'mean', 'sum']
        }).round(2)
        unit_analysis.columns = ['Item_Count', 'Avg_Price', 'Total_Value']
        unit_analysis = unit_analysis.sort_values('Total_Value', ascending=False)
        
        fig_units = px.bar(
            unit_analysis.reset_index(), 
            x='UI', 
            y='Total_Value',
            title="Total Value by Unit of Issue",
            labels={'Total_Value': 'Total Value ($)', 'UI': 'Unit of Issue'}
        )
        st.plotly_chart(fig_units, use_container_width=True)
    
    with tab2:
        st.markdown("### Category Performance Analysis")
        category_chart, category_stats = create_category_analysis(filtered_df)
        st.plotly_chart(category_chart, use_container_width=True)
        
        st.markdown("### Top Categories by Value")
        st.dataframe(category_stats, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Search Specific Items")
        
        # Search functionality
        search_term = st.text_input("Search by NSN, product name, or description:")
        
        if search_term:
            search_results = filtered_df[
                filtered_df['NSN'].str.contains(search_term, case=False, na=False) |
                filtered_df['common_name'].str.contains(search_term, case=False, na=False) |
                filtered_df['Description'].str.contains(search_term, case=False, na=False)
            ]
            
            if not search_results.empty:
                st.markdown(f"**Found {len(search_results)} items:**")
                st.dataframe(
                    search_results[['NSN', 'common_name', 'Price', 'UI', 'rep_office']],
                    use_container_width=True
                )
            else:
                st.info("No items found matching your search.")
        
        # Display random sample
        st.markdown("### Sample Items")
        sample_items = filtered_df.sample(min(10, len(filtered_df)))
        st.dataframe(
            sample_items[['NSN', 'common_name', 'Price', 'UI', 'Description']],
            use_container_width=True
        )
    
    with tab4:
        st.markdown("### 💡 Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insights-box">
            <h4>🎯 Cost Optimization Opportunities</h4>
            <ul>
            <li>Focus on high-value categories for maximum impact</li>
            <li>Investigate price variations within similar products</li>
            <li>Consider bulk purchasing for frequently ordered items</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insights-box">
            <h4>📊 Data Quality Observations</h4>
            <ul>
            <li>Price range varies significantly ($0.09 - $160K+)</li>
            <li>Multiple unit types suggest standardization opportunities</li>
            <li>Category consolidation could improve efficiency</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional insights based on filtered data
        if not filtered_df.empty:
            st.markdown("### 📈 Current Selection Insights")
            
            # Price insights
            q75 = filtered_df['Price'].quantile(0.75)
            high_value_items = len(filtered_df[filtered_df['Price'] > q75])
            
            st.info(f"""
            **Current Selection Summary:**
            - {len(filtered_df):,} items selected
            - {high_value_items:,} high-value items (top 25% by price)
            - Average price: ${filtered_df['Price'].mean():,.2f}
            - Most common unit: {filtered_df['UI'].mode().iloc[0] if not filtered_df.empty else 'N/A'}
            """)

if __name__ == "__main__":
    main()