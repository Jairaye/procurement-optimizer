"""
Cost Normalization Module for Procurement Optimizer

This module standardizes costs across different units of issue (UI) to enable
fair price comparisons and identify cost optimization opportunities.

Author: Your Name
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostNormalizer:
    """
    Normalizes procurement costs across different units of issue for fair comparison.
    
    Business Logic:
    - Converts all prices to a standard "per unit" basis
    - Handles bulk packaging conversions (BX=12, DZ=12, etc.)
    - Identifies pricing anomalies and optimization opportunities
    """
    
    def __init__(self):
        """Initialize the cost normalizer with standard unit conversions."""
        
        # Standard unit conversion factors (to single units)
        self.unit_conversions = {
            'EA': 1,      # Each (base unit)
            'BX': 12,     # Box (typically 12 items)
            'DZ': 12,     # Dozen
            'PG': 1,      # Package (treat as single unit)
            'SE': 1,      # Set (treat as single unit)
            'KT': 1,      # Kit (treat as single unit)
            'PR': 2,      # Pair
            'RO': 500,    # Roll (estimated)
            'HD': 100,    # Hundred
            'BD': 1000,   # Thousand (BD = "bund" or bundle)
            'DR': 1,      # Drum (treat as single unit)
            'CN': 1,      # Can (treat as single unit)
            'CO': 1,      # Container (treat as single unit)
            'SH': 1,      # Sheet (treat as single unit)
            'BG': 1,      # Bag (treat as single unit)
            'MX': 1000,   # Mix/thousand
            'GL': 1,      # Gallon (treat as single unit)
            'CL': 1,      # Coil (treat as single unit)
            'BE': 1,      # Bottle (treat as single unit)
            'PD': 1,      # Pad (treat as single unit)
            'LG': 1,      # Log (treat as single unit)
            'RL': 1,      # Roll (treat as single unit)
            'BT': 1,      # Boot (treat as single unit)
            'LB': 1,      # Pound (treat as single unit)
            'AT': 1,      # Assembly (treat as single unit)
            'QT': 1,      # Quart (treat as single unit)
            'PT': 1,      # Pint (treat as single unit)
            'GR': 144,    # Gross (144 items)
            'BK': 1,      # Book (treat as single unit)
            'BO': 1,      # Bottle (treat as single unit)
            'SL': 1,      # Sleeve (treat as single unit)
            'AY': 1,      # Array (treat as single unit)
            'FT': 1,      # Foot (treat as single unit)
            'TU': 1,      # Tube (treat as single unit)
            'RM': 500,    # Ream (500 sheets)
            'OT': 1,      # Outfit (treat as single unit)
            'HK': 1,      # Hook (treat as single unit)
            'CE': 1,      # Case (treat as single unit)
            'BL': 1,      # Bale (treat as single unit)
        }
        
        logger.info(f"Initialized CostNormalizer with {len(self.unit_conversions)} unit types")
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize costs for an entire DataFrame.
        
        Args:
            df: DataFrame with 'Price', 'UI', and 'common_name' columns
            
        Returns:
            DataFrame with additional normalized cost columns
        """
        logger.info(f"Starting cost normalization for {len(df)} items")
        
        # Create a copy to avoid modifying original data
        normalized_df = df.copy()
        
        # Add normalized cost columns
        normalized_df['unit_conversion_factor'] = normalized_df['UI'].map(self.unit_conversions)
        normalized_df['price_per_unit'] = (
            normalized_df['Price'] / normalized_df['unit_conversion_factor'].fillna(1)
        )
        
        # Add cost category (for business insights)
        normalized_df['cost_category'] = pd.cut(
            normalized_df['price_per_unit'],
            bins=[0, 1, 10, 100, 1000, float('inf')],
            labels=['Low (<$1)', 'Medium ($1-$10)', 'High ($10-$100)', 
                   'Very High ($100-$1000)', 'Premium (>$1000)']
        )
        
        # Calculate category statistics
        normalized_df = self._add_category_benchmarks(normalized_df)
        
        logger.info("Cost normalization completed successfully")
        return normalized_df
    
    def _add_category_benchmarks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category-based price benchmarks for comparison."""
        
        # Calculate median price per unit by category
        category_medians = df.groupby('common_name')['price_per_unit'].median()
        df['category_median_price'] = df['common_name'].map(category_medians)
        
        # Calculate price ratio vs category median
        df['price_vs_category_median'] = df['price_per_unit'] / df['category_median_price']
        
        # Flag potential optimization opportunities
        df['optimization_flag'] = df['price_vs_category_median'].apply(
            lambda x: 'High Cost' if x > 1.5 else 'Low Cost' if x < 0.7 else 'Normal'
        )
        
        return df
    
    def find_cost_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify items with unusually high or low costs compared to similar items.
        
        Args:
            df: Normalized DataFrame
            threshold: Standard deviations from mean to flag as anomaly
            
        Returns:
            DataFrame of potential cost anomalies
        """
        normalized_df = self.normalize_dataframe(df) if 'price_per_unit' not in df.columns else df
        
        # Calculate z-scores within each category
        category_stats = normalized_df.groupby('common_name')['price_per_unit'].agg(['mean', 'std'])
        
        anomalies = []
        for category in category_stats.index:
            category_data = normalized_df[normalized_df['common_name'] == category]
            mean_price = category_stats.loc[category, 'mean']
            std_price = category_stats.loc[category, 'std']
            
            if std_price > 0:  # Avoid division by zero
                z_scores = (category_data['price_per_unit'] - mean_price) / std_price
                anomaly_mask = abs(z_scores) > threshold
                
                if anomaly_mask.any():
                    category_anomalies = category_data[anomaly_mask].copy()
                    category_anomalies['z_score'] = z_scores[anomaly_mask]
                    category_anomalies['anomaly_type'] = np.where(
                        z_scores[anomaly_mask] > 0, 'High Cost', 'Low Cost'
                    )
                    anomalies.append(category_anomalies)
        
        if anomalies:
            anomaly_df = pd.concat(anomalies, ignore_index=True)
            logger.info(f"Found {len(anomaly_df)} cost anomalies")
            return anomaly_df.sort_values('z_score', ascending=False)
        else:
            logger.info("No significant cost anomalies found")
            return pd.DataFrame()
    
    def generate_savings_opportunities(self, df: pd.DataFrame) -> Dict:
        """
        Identify potential cost savings opportunities.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Dictionary with savings analysis
        """
        normalized_df = self.normalize_dataframe(df) if 'price_per_unit' not in df.columns else df
        
        # Find high-cost items that could be substituted
        high_cost_items = normalized_df[normalized_df['optimization_flag'] == 'High Cost']
        
        # Calculate potential savings
        total_high_cost_value = high_cost_items['Price'].sum()
        
        # Estimate savings if high-cost items were reduced to category median
        potential_savings = high_cost_items.apply(
            lambda row: row['Price'] - (row['category_median_price'] * row['unit_conversion_factor']),
            axis=1
        ).sum()
        
        # Top categories for optimization
        category_opportunities = (
            high_cost_items.groupby('common_name')
            .agg({
                'Price': 'sum',
                'NSN': 'count',
                'price_vs_category_median': 'mean'
            })
            .sort_values('Price', ascending=False)
            .head(10)
        )
        
        savings_analysis = {
            'total_high_cost_value': total_high_cost_value,
            'potential_savings': max(0, potential_savings),  # Ensure non-negative
            'savings_percentage': (potential_savings / total_high_cost_value * 100) if total_high_cost_value > 0 else 0,
            'high_cost_item_count': len(high_cost_items),
            'top_optimization_categories': category_opportunities,
            'optimization_summary': {
                'items_to_review': len(high_cost_items),
                'categories_affected': high_cost_items['common_name'].nunique(),
                'average_cost_premium': high_cost_items['price_vs_category_median'].mean()
            }
        }
        
        logger.info(f"Identified ${potential_savings:,.2f} in potential savings ({savings_analysis['savings_percentage']:.1f}%)")
        return savings_analysis

def normalize_procurement_costs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to normalize costs and generate savings analysis.
    
    Args:
        df: DataFrame with procurement data
        
    Returns:
        Tuple of (normalized DataFrame, savings analysis)
    """
    normalizer = CostNormalizer()
    normalized_df = normalizer.normalize_dataframe(df)
    savings_analysis = normalizer.generate_savings_opportunities(normalized_df)
    
    return normalized_df, savings_analysis

# Example usage and testing
if __name__ == "__main__":
    # This would typically be called from your Streamlit app
    print("Cost Normalization Module - Ready for import")
    print("Usage: from src.normalize_costs import normalize_procurement_costs")