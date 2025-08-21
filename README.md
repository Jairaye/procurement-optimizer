# Procurement Optimizer 📊

A comprehensive data analytics tool for procurement analysts and managers to gain actionable insights from NSN (National Stock Number) catalog data. This interactive dashboard helps answer critical business questions about inventory costs, supplier relationships, and optimization opportunities.

## 🎯 Business Value

**For Procurement Managers:**
- Identify cost-saving opportunities across 12,400+ catalog items
- Analyze spending patterns and price trends by category
- Discover potential supplier consolidation opportunities
- Make data-driven decisions on inventory optimization

**For Analysts:**
- Interactive dashboards for deep-dive analysis
- Automated cost normalization and comparison tools
- Scenario modeling for procurement strategy planning
- Export capabilities for executive reporting

## 📈 Key Insights Available

This tool answers critical procurement questions such as:

- **Cost Analysis**: Which items/categories drive the highest costs?
- **Price Optimization**: Where are the biggest savings opportunities?
- **Supplier Performance**: How do representative offices compare on pricing?
- **Inventory Strategy**: What substitution opportunities exist for expensive items?
- **Budget Planning**: How do unit costs vary by packaging (EA, BX, DZ, etc.)?

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 2GB RAM minimum
- Web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/procurement-optimizer.git
cd procurement-optimizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run dashboard/streamlit_app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📁 Project Structure

```
procurement-optimizer/
│
├── 📊 dashboard/
│   └── streamlit_app.py          # Main interactive dashboard
│
├── 📁 data/
│   └── nsn_data.csv              # NSN catalog dataset (12,454 items)
│
├── 📓 notebooks/
│   └── exploratory_analysis.ipynb # Jupyter analysis & insights
│
├── 🔧 src/
│   ├── normalize_costs.py        # Cost standardization utilities
│   ├── nsn_dependency_mapper.py  # Item relationship analysis
│   ├── scenario_simulator.py     # What-if analysis tools
│   └── substitution_logic.py     # Alternative item recommendations
│
├── 📋 requirements.txt           # Python dependencies
└── 📖 README.md                 # This file
```

## 💡 Features

### Interactive Dashboard
- **Cost Analytics**: Real-time price analysis and trending
- **Category Insights**: Drill-down by product categories and suppliers
- **Search & Filter**: Find specific NSNs or product types instantly
- **Visualization**: Charts, heatmaps, and comparison tools
- **Export**: Download analysis results for presentations

### Advanced Analytics
- **Cost Normalization**: Standardize prices across different units of issue
- **Dependency Mapping**: Identify related items and bundle opportunities
- **Scenario Modeling**: Simulate procurement strategy changes
- **Substitution Engine**: Find alternative products for cost savings

### Business Intelligence
- **Executive Summary**: Key metrics and trends at-a-glance
- **Anomaly Detection**: Flag unusual pricing or inventory patterns
- **Benchmarking**: Compare costs against category averages
- **ROI Calculator**: Estimate savings from optimization strategies

## 📊 Sample Insights

**Dataset Overview:**
- 12,454 unique NSN items
- Price range: $0.09 - $160,169.07
- Average item cost: $488.19
- 40+ units of issue types
- 200+ product categories

**Top Cost Categories:**
1. Combination Wrench (Box and Open End) - 208 items
2. Socket (Regular Length) - 171 items  
3. Remanufactured Toner Cartridge - 170 items
4. Floor Machine Pad - 157 items
5. Drill Bit (High Speed Steel) - 122 items

## 🔍 Usage Examples

### For Procurement Managers
```python
# Find high-value optimization opportunities
python src/scenario_simulator.py --category "Toner Cartridge" --savings-target 15

# Analyze supplier performance
python src/normalize_costs.py --group-by rep_office --output executive_summary.xlsx
```

### For Analysts
```python
# Discover substitution opportunities
python src/substitution_logic.py --nsn "3510-00-222-1457" --max-price-diff 10

# Map item dependencies
python src/nsn_dependency_mapper.py --category "Wrench" --output dependency_map.json
```

## 📈 Dashboard Screenshots

*Add screenshots of your key dashboard views here when complete*

## 🛠️ Technical Details

**Built With:**
- **Frontend**: Streamlit (interactive web dashboard)
- **Backend**: Python, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Excel integration, CSV handling
- **Analysis**: Scikit-learn, SciPy (statistical analysis)

**Performance:**
- Loads 12K+ records in <2 seconds
- Real-time filtering and search
- Responsive design for desktop/tablet
- Memory optimized for large datasets

## 📚 Documentation

- **User Guide**: See `notebooks/exploratory_analysis.ipynb` for detailed analysis examples
- **API Reference**: Each module in `src/` contains detailed docstrings
- **Business Logic**: Comments explain procurement-specific calculations
- **Data Dictionary**: NSN field definitions and business rules

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new analysis'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create a Pull Request

## 📞 Contact

**Joshua [Your Last Name]**
- Portfolio: [Your Portfolio URL]
- LinkedIn: [Your LinkedIn Profile]
- Email: [Your Email]
- GitHub: [Your GitHub Profile]

## 📄 License

This project is part of a data analytics portfolio. The code is available under MIT License - see LICENSE file for details.

---

### 🎯 Portfolio Context

This project demonstrates:
- **Business Intelligence**: Translating data into actionable procurement insights
- **Full-Stack Development**: End-to-end analytics solution with interactive frontend
- **Data Engineering**: ETL processes, data cleaning, and optimization
- **Domain Expertise**: Understanding of procurement, supply chain, and inventory management
- **Technical Skills**: Python, Streamlit, data visualization, statistical analysis

*Part of a comprehensive data analytics portfolio showcasing real-world business problem solving.*