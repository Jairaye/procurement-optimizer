import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Import our custom modules
try:
    from src.normalize_costs import normalize_procurement_costs, CostNormalizer
    from src.substitution_logic import find_product_substitutions, ProductSubstitutionEngine
    from src.scenario_simulator import create_scenario_simulator, ScenarioType
    from src.nsn_dependency_mapper import analyze_nsn_dependencies, create_bundle_recommendations
    modules_available = True
except ImportError as e:
    st.warning(f"Advanced modules not available: {e}")
    modules_available = False

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
    .feature-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .savings-highlight {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the NSN dataset"""
    try:
        # Try Excel format first
        try:
            df = pd.read_excel('data/nsn_data.csv')
        except:
            # Fallback to CSV if Excel fails
            df = pd.read_csv('data/nsn_data.csv')
        
        # Clean and prepare data
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        df['Price_Log'] = np.log1p(df['Price'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Make sure 'data/nsn_data.csv' exists in your repository")
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

def create_advanced_cost_analysis(df):
    """Create advanced cost analysis using normalization module"""
    if not modules_available:
        st.warning("Advanced cost analysis requires custom modules to be installed.")
        return
    
    with st.spinner("Performing advanced cost normalization..."):
        try:
            normalized_df, savings_analysis = normalize_procurement_costs(df)
            
            st.markdown("### 💰 Cost Optimization Analysis")
            
            # Display savings opportunities
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="savings-highlight">
                <h4>💡 Potential Savings Identified</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric(
                    "Potential Annual Savings",
                    f"${savings_analysis['potential_savings']:,.2f}",
                    f"{savings_analysis['savings_percentage']:.1f}% of total spend"
                )
                
                st.metric(
                    "High-Cost Items to Review",
                    f"{savings_analysis['high_cost_item_count']:,}",
                    f"{savings_analysis['optimization_summary']['categories_affected']} categories affected"
                )
            
            with col2:
                # Cost category distribution
                if 'cost_category' in normalized_df.columns:
                    cost_dist = normalized_df['cost_category'].value_counts()
                    fig_cost = px.pie(
                        values=cost_dist.values,
                        names=cost_dist.index,
                        title="Items by Cost Category"
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
            
            # Top optimization opportunities
            if not savings_analysis['top_optimization_categories'].empty:
                st.markdown("### 🎯 Top Optimization Opportunities")
                st.dataframe(
                    savings_analysis['top_optimization_categories'].head(10),
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"Error in cost analysis: {e}")

def create_substitution_analysis(df):
    """Create substitution analysis interface"""
    if not modules_available:
        st.warning("Substitution analysis requires custom modules to be installed.")
        return
    
    st.markdown("### 🔄 Product Substitution Analysis")
    
    # NSN input for substitution search
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_nsn = st.selectbox(
            "Select NSN for substitution analysis:",
            options=df['NSN'].unique()[:100],  # Limit for performance
            format_func=lambda x: f"{x} - {df[df['NSN']==x]['common_name'].iloc[0] if len(df[df['NSN']==x]) > 0 else 'Unknown'}"
        )
    
    with col2:
        if st.button("Find Substitutions", type="primary"):
            with st.spinner("Analyzing substitution opportunities..."):
                try:
                    substitutions = find_product_substitutions(df, target_nsn, max_results=10)
                    
                    if not substitutions.empty:
                        st.markdown("#### 🔍 Substitution Opportunities Found")
                        
                        # Display substitutions with savings
                        for idx, sub in substitutions.head(5).iterrows():
                            with st.expander(f"💰 Save ${sub['potential_savings']:.2f} ({sub['savings_percentage']:.1f}%)"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Alternative:** {sub['common_name']}")
                                    st.write(f"**NSN:** {sub['NSN']}")
                                    st.write(f"**Price:** ${sub['Price']:.2f}")
                                with col2:
                                    st.write(f"**Similarity Score:** {sub['similarity_score']:.2f}")
                                    st.write(f"**Unit:** {sub['UI']}")
                                    st.write(f"**Supplier:** {sub['rep_office']}")
                    else:
                        st.info("No suitable substitutions found for this item.")
                        
                except Exception as e:
                    st.error(f"Error in substitution analysis: {e}")

def create_scenario_simulation(df):
    """Create scenario simulation interface"""
    if not modules_available:
        st.warning("Scenario simulation requires custom modules to be installed.")
        return
    
    st.markdown("### 🎯 Procurement Scenario Simulation")
    
    # Scenario selection
    scenario_type = st.selectbox(
        "Select Scenario Type:",
        ["Volume Discount", "Supplier Consolidation", "Substitution Strategy", "Budget Constraint"]
    )
    
    try:
        simulator = create_scenario_simulator(df)
        
        if scenario_type == "Volume Discount":
            st.markdown("#### 📈 Volume Discount Simulation")
            
            col1, col2 = st.columns(2)
            with col1:
                discount_5_percent = st.slider("Discount for 5+ items (%)", 0, 20, 5)
                discount_10_percent = st.slider("Discount for 10+ items (%)", 0, 30, 10)
                discount_25_percent = st.slider("Discount for 25+ items (%)", 0, 40, 15)
            
            with col2:
                selected_categories = st.multiselect(
                    "Target Categories:",
                    options=df['common_name'].value_counts().head(20).index.tolist(),
                    default=df['common_name'].value_counts().head(5).index.tolist()
                )
            
            if st.button("Run Volume Discount Simulation"):
                with st.spinner("Simulating volume discount scenario..."):
                    discount_tiers = {
                        5: discount_5_percent / 100,
                        10: discount_10_percent / 100,
                        25: discount_25_percent / 100
                    }
                    
                    results = simulator.simulate_volume_discount_scenario(
                        discount_tiers, 
                        target_categories=selected_categories if selected_categories else None
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Savings", f"${results['total_savings']:,.2f}")
                    with col2:
                        st.metric("Savings Percentage", f"{results['savings_percentage']:.2f}%")
                    with col3:
                        st.metric("Items Affected", results['discount_details']['items_with_discount'])
        
        elif scenario_type == "Budget Constraint":
            st.markdown("#### 💰 Budget Constraint Simulation")
            
            current_total = df['Price'].sum()
            col1, col2 = st.columns(2)
            
            with col1:
                budget_limit = st.number_input(
                    "Budget Limit ($):",
                    min_value=1000,
                    max_value=int(current_total),
                    value=int(current_total * 0.8),
                    step=1000
                )
            
            with col2:
                optimization_strategy = st.selectbox(
                    "Optimization Strategy:",
                    ["value_priority", "necessity_priority", "cost_efficiency"]
                )
            
            if st.button("Run Budget Constraint Simulation"):
                with st.spinner("Optimizing for budget constraint..."):
                    results = simulator.simulate_budget_constraint_scenario(
                        budget_limit, 
                        optimization_strategy
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Budget Utilized", f"${results['budget_details']['budget_utilized']:,.2f}")
                    with col2:
                        st.metric("Utilization Rate", f"{results['budget_details']['budget_utilization_rate']:.1%}")
                    with col3:
                        st.metric("Items Selected", results['budget_details']['items_selected'])
    
    except Exception as e:
        st.error(f"Error in scenario simulation: {e}")

def create_dependency_analysis(df):
    """Create dependency and bundle analysis"""
    if not modules_available:
        st.warning("Dependency analysis requires custom modules to be installed.")
        return
    
    st.markdown("### 🔗 Item Dependencies & Bundle Opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Analyze Bundle Opportunities", type="primary"):
            with st.spinner("Analyzing bundle opportunities..."):
                try:
                    bundles = create_bundle_recommendations(df, min_bundle_size=2)
                    
                    if bundles:
                        st.markdown("#### 📦 Top Bundle Recommendations")
                        
                        for i, bundle in enumerate(bundles[:5]):
                            with st.expander(f"Bundle {i+1}: Save ${bundle['potential_savings']:.2f}"):
                                st.write(f"**Category:** {bundle['category']}")
                                st.write(f"**Bundle Size:** {bundle['bundle_size']} items")
                                st.write(f"**Original Total:** ${bundle['original_total']:.2f}")
                                st.write(f"**Discounted Total:** ${bundle['discounted_total']:.2f}")
                                
                                # Show items in bundle
                                st.write("**Items in Bundle:**")
                                for item in bundle['items']:
                                    st.write(f"- {item['NSN']}: {item['common_name']} - ${item['Price']:.2f}")
                    else:
                        st.info("No bundle opportunities found.")
                        
                except Exception as e:
                    st.error(f"Error in bundle analysis: {e}")
    
    with col2:
        # Individual item dependency lookup
        st.markdown("#### 🔍 Item Dependency Lookup")
        
        selected_nsn = st.selectbox(
            "Select NSN for dependency analysis:",
            options=df['NSN'].unique()[:50],  # Limit for performance
            format_func=lambda x: f"{x} - {df[df['NSN']==x]['common_name'].iloc[0] if len(df[df['NSN']==x]) > 0 else 'Unknown'}"
        )
        
        if st.button("Find Dependencies"):
            with st.spinner("Finding item dependencies..."):
                try:
                    from src.nsn_dependency_mapper import NSNDependencyMapper
                    mapper = NSNDependencyMapper()
                    dependencies = mapper.find_item_dependencies(df, selected_nsn)
                    
                    if dependencies:
                        target_item = dependencies['target_item']
                        st.write(f"**Target Item:** {target_item['name']}")
                        st.write(f"**Price:** ${target_item['price']:.2f}")
                        
                        # Show related items
                        if dependencies['similar_items']:
                            st.write("**Similar Items:**")
                            for item in dependencies['similar_items'][:3]:
                                st.write(f"- {item['name']}: ${item['price']:.2f}")
                        
                        if dependencies['bundle_candidates']:
                            st.write("**Bundle Candidates:**")
                            for item in dependencies['bundle_candidates']:
                                st.write(f"- {item['name']}: ${item['price']:.2f} (Save ${item['estimated_savings']:.2f})")
                    
                except Exception as e:
                    st.error(f"Error in dependency analysis: {e}")

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
    
    # Apply filters
    filtered_df = df[
        (df['Price'] >= price_range[0]) & 
        (df['Price'] <= price_range[1])
    ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['common_name'].isin(categories)]
    
    # Summary metrics
    st.markdown("## 📈 Key Metrics")
    create_summary_metrics(filtered_df)
    
    # Main content tabs with advanced features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", 
        "🏷️ Categories", 
        "🔍 Item Search", 
        "💰 Cost Analysis", 
        "🔄 Substitutions", 
        "🎯 Scenarios", 
        "💡 Insights"
    ])
    
    with tab1:
        st.markdown("### Price Analysis")
        
        # Create price distribution chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Price Distribution (Log Scale)', 'Top 20 Most Expensive Items'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Price distribution histogram
        fig.add_trace(
            go.Histogram(x=filtered_df['Price_Log'], nbinsx=50, name="Price Distribution"),
            row=1, col=1
        )
        
        # Top expensive items
        top_expensive = filtered_df.nlargest(20, 'Price')
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
        
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Category summary
        category_stats = filtered_df.groupby('common_name').agg({
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
        st.plotly_chart(fig, use_container_width=True)
        
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
        if len(filtered_df) > 0:
            sample_items = filtered_df.sample(min(10, len(filtered_df)))
            st.dataframe(
                sample_items[['NSN', 'common_name', 'Price', 'UI', 'Description']],
                use_container_width=True
            )
    
    with tab4:
        # Advanced cost analysis
        create_advanced_cost_analysis(filtered_df)
    
    with tab5:
        # Substitution analysis
        create_substitution_analysis(filtered_df)
    
    with tab6:
        # Scenario simulation
        create_scenario_simulation(filtered_df)
    
    with tab7:
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
            <li>Explore generic alternatives for branded items</li>
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
            <li>Supplier overlap indicates consolidation potential</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Bundle and dependency analysis
        create_dependency_analysis(filtered_df)
        
        # Additional insights based on filtered data
        if not filtered_df.empty:
            st.markdown("### 📈 Current Selection Insights")
            
            # Price insights
            q75 = filtered_df['Price'].quantile(0.75)
            high_value_items = len(filtered_df[filtered_df['Price'] > q75])
            
            # Supplier analysis
            supplier_concentration = filtered_df['rep_office'].value_counts()
            top_supplier_share = supplier_concentration.iloc[0] / len(filtered_df) if len(supplier_concentration) > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Selection Summary:**
                - {len(filtered_df):,} items selected
                - {high_value_items:,} high-value items (top 25% by price)
                - Average price: ${filtered_df['Price'].mean():,.2f}
                - Most common unit: {filtered_df['UI'].mode().iloc[0] if not filtered_df.empty else 'N/A'}
                """)
            
            with col2:
                st.info(f"""
                **Supplier Insights:**
                - {filtered_df['rep_office'].nunique()} unique suppliers
                - Top supplier: {supplier_concentration.index[0] if len(supplier_concentration) > 0 else 'N/A'}
                - Concentration: {top_supplier_share:.1%} of items
                - Categories: {filtered_df['common_name'].nunique()} different types
                """)
        
        # Advanced insights section
        if modules_available:
            st.markdown("### 🔍 Advanced Analytics Insights")
            
            try:
                # Quick cost analysis summary
                normalizer = CostNormalizer()
                normalized_sample = normalizer.normalize_dataframe(filtered_df.sample(min(1000, len(filtered_df))))
                
                optimization_flags = normalized_sample['optimization_flag'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Cost Optimization Flags:**")
                    for flag, count in optimization_flags.items():
                        percentage = count / len(normalized_sample) * 100
                        st.write(f"- {flag}: {count} items ({percentage:.1f}%)")
                
                with col2:
                    if 'High Cost' in optimization_flags:
                        high_cost_count = optimization_flags['High Cost']
                        potential_review_value = normalized_sample[
                            normalized_sample['optimization_flag'] == 'High Cost'
                        ]['Price'].sum()
                        
                        st.markdown("**Immediate Actions:**")
                        st.write(f"- Review {high_cost_count} high-cost items")
                        st.write(f"- Total value to review: ${potential_review_value:,.2f}")
                        st.write("- Consider substitution analysis")
                        st.write("- Negotiate with suppliers")
            
            except Exception as e:
                st.write("Advanced analytics processing...")

if __name__ == "__main__":
    main()