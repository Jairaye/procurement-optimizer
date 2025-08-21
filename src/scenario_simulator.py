"""
Scenario Simulator Module for Procurement Optimizer

This module enables "what-if" analysis for procurement strategies, allowing managers
to simulate the impact of different procurement decisions on costs and efficiency.

Author: Your Name
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of procurement scenarios that can be simulated."""
    VOLUME_DISCOUNT = "volume_discount"
    SUPPLIER_CONSOLIDATION = "supplier_consolidation"
    SUBSTITUTION_STRATEGY = "substitution_strategy"
    INVENTORY_OPTIMIZATION = "inventory_optimization"
    BUDGET_CONSTRAINT = "budget_constraint"

@dataclass
class ScenarioParameters:
    """Parameters for different scenario types."""
    scenario_type: ScenarioType
    target_categories: List[str] = None
    discount_percentage: float = 0.0
    volume_threshold: int = 0
    budget_limit: float = 0.0
    substitution_rate: float = 0.0
    consolidation_savings: float = 0.0

class ProcurementScenarioSimulator:
    """
    Simulates various procurement scenarios and their financial impact.
    
    Capabilities:
    - Volume discount negotiations
    - Supplier consolidation benefits
    - Product substitution strategies
    - Budget constraint optimization
    - Inventory level adjustments
    """
    
    def __init__(self, base_data: pd.DataFrame):
        """
        Initialize the scenario simulator with base procurement data.
        
        Args:
            base_data: DataFrame with current procurement data
        """
        self.base_data = base_data.copy()
        self.scenarios = {}
        
        # Calculate baseline metrics
        self.baseline_metrics = self._calculate_baseline_metrics()
        
        logger.info(f"Initialized scenario simulator with {len(self.base_data)} items")
        logger.info(f"Baseline total value: ${self.baseline_metrics['total_value']:,.2f}")
    
    def simulate_volume_discount_scenario(self, 
                                        discount_tiers: Dict[int, float],
                                        target_categories: List[str] = None) -> Dict:
        """
        Simulate volume discount negotiations.
        
        Args:
            discount_tiers: Dictionary of {quantity_threshold: discount_percentage}
            target_categories: List of categories to apply discounts to
            
        Returns:
            Dictionary with scenario results
        """
        logger.info("Simulating volume discount scenario")
        
        scenario_data = self.base_data.copy()
        
        # Filter to target categories if specified
        if target_categories:
            mask = scenario_data['common_name'].isin(target_categories)
            scenario_data.loc[~mask, 'discount_applied'] = 0.0
        else:
            mask = pd.Series([True] * len(scenario_data))
        
        # Apply volume discounts
        scenario_data['original_price'] = scenario_data['Price']
        scenario_data['discount_applied'] = 0.0
        
        # Group by category to calculate volumes
        category_volumes = scenario_data.groupby('common_name').size()
        
        for category, volume in category_volumes.items():
            # Find applicable discount tier
            applicable_discount = 0.0
            for threshold, discount in sorted(discount_tiers.items()):
                if volume >= threshold:
                    applicable_discount = discount
            
            # Apply discount to category items
            category_mask = (scenario_data['common_name'] == category) & mask
            scenario_data.loc[category_mask, 'discount_applied'] = applicable_discount
            scenario_data.loc[category_mask, 'Price'] = (
                scenario_data.loc[category_mask, 'original_price'] * (1 - applicable_discount)
            )
        
        # Calculate results
        results = self._calculate_scenario_results(scenario_data, "Volume Discount")
        results['discount_details'] = {
            'discount_tiers': discount_tiers,
            'categories_affected': scenario_data[scenario_data['discount_applied'] > 0]['common_name'].nunique(),
            'average_discount': scenario_data['discount_applied'].mean(),
            'items_with_discount': len(scenario_data[scenario_data['discount_applied'] > 0])
        }
        
        return results
    
    def simulate_supplier_consolidation_scenario(self,
                                               consolidation_savings: float = 0.05,
                                               target_suppliers: List[str] = None) -> Dict:
        """
        Simulate supplier consolidation benefits.
        
        Args:
            consolidation_savings: Expected savings percentage from consolidation
            target_suppliers: List of suppliers to consolidate (rep_office)
            
        Returns:
            Dictionary with scenario results
        """
        logger.info("Simulating supplier consolidation scenario")
        
        scenario_data = self.base_data.copy()
        
        # Identify consolidation opportunities
        if target_suppliers:
            consolidation_mask = scenario_data['rep_office'].isin(target_suppliers)
        else:
            # Identify suppliers with overlapping product categories
            supplier_categories = scenario_data.groupby('rep_office')['common_name'].apply(set)
            
            # Find suppliers with >50% category overlap
            consolidation_candidates = []
            suppliers = list(supplier_categories.index)
            
            for i, supplier1 in enumerate(suppliers):
                for supplier2 in suppliers[i+1:]:
                    overlap = len(supplier_categories[supplier1] & supplier_categories[supplier2])
                    total_unique = len(supplier_categories[supplier1] | supplier_categories[supplier2])
                    
                    if overlap / total_unique > 0.5:  # 50% overlap threshold
                        consolidation_candidates.extend([supplier1, supplier2])
            
            consolidation_mask = scenario_data['rep_office'].isin(consolidation_candidates)
        
        # Apply consolidation savings
        scenario_data['original_price'] = scenario_data['Price']
        scenario_data['consolidation_savings'] = 0.0
        
        scenario_data.loc[consolidation_mask, 'consolidation_savings'] = consolidation_savings
        scenario_data.loc[consolidation_mask, 'Price'] = (
            scenario_data.loc[consolidation_mask, 'original_price'] * (1 - consolidation_savings)
        )
        
        # Calculate results
        results = self._calculate_scenario_results(scenario_data, "Supplier Consolidation")
        results['consolidation_details'] = {
            'consolidation_savings_rate': consolidation_savings,
            'suppliers_affected': scenario_data[consolidation_mask]['rep_office'].nunique(),
            'items_affected': consolidation_mask.sum(),
            'categories_affected': scenario_data[consolidation_mask]['common_name'].nunique()
        }
        
        return results
    
    def simulate_substitution_strategy_scenario(self,
                                              substitution_targets: Dict[str, float],
                                              average_substitution_savings: float = 0.15) -> Dict:
        """
        Simulate product substitution strategy.
        
        Args:
            substitution_targets: Dict of {category: substitution_rate}
            average_substitution_savings: Expected average savings from substitutions
            
        Returns:
            Dictionary with scenario results
        """
        logger.info("Simulating substitution strategy scenario")
        
        scenario_data = self.base_data.copy()
        scenario_data['original_price'] = scenario_data['Price']
        scenario_data['substitution_applied'] = False
        scenario_data['substitution_savings'] = 0.0
        
        total_substitutions = 0
        
        for category, substitution_rate in substitution_targets.items():
            category_items = scenario_data[scenario_data['common_name'] == category]
            
            if len(category_items) > 0:
                # Select items for substitution (highest priced items first)
                sorted_items = category_items.sort_values('Price', ascending=False)
                num_to_substitute = int(len(sorted_items) * substitution_rate)
                
                if num_to_substitute > 0:
                    items_to_substitute = sorted_items.head(num_to_substitute).index
                    
                    # Apply substitution savings with some randomness
                    np.random.seed(42)  # For reproducibility
                    savings_variation = np.random.normal(average_substitution_savings, 0.03, num_to_substitute)
                    savings_variation = np.clip(savings_variation, 0.05, 0.30)  # 5-30% savings range
                    
                    scenario_data.loc[items_to_substitute, 'substitution_applied'] = True
                    scenario_data.loc[items_to_substitute, 'substitution_savings'] = savings_variation
                    scenario_data.loc[items_to_substitute, 'Price'] = (
                        scenario_data.loc[items_to_substitute, 'original_price'] * (1 - savings_variation)
                    )
                    
                    total_substitutions += num_to_substitute
        
        # Calculate results
        results = self._calculate_scenario_results(scenario_data, "Substitution Strategy")
        results['substitution_details'] = {
            'substitution_targets': substitution_targets,
            'total_substitutions': total_substitutions,
            'average_savings_per_substitution': scenario_data[scenario_data['substitution_applied']]['substitution_savings'].mean(),
            'categories_with_substitutions': len(substitution_targets)
        }
        
        return results
    
    def simulate_budget_constraint_scenario(self,
                                          budget_limit: float,
                                          optimization_strategy: str = "value_priority") -> Dict:
        """
        Simulate procurement under budget constraints.
        
        Args:
            budget_limit: Maximum budget available
            optimization_strategy: "value_priority", "necessity_priority", or "cost_efficiency"
            
        Returns:
            Dictionary with scenario results
        """
        logger.info(f"Simulating budget constraint scenario with ${budget_limit:,.2f} limit")
        
        scenario_data = self.base_data.copy()
        
        # Calculate priority scores based on strategy
        if optimization_strategy == "value_priority":
            # Prioritize high-value items
            scenario_data['priority_score'] = scenario_data['Price']
        elif optimization_strategy == "necessity_priority":
            # Prioritize based on category importance (simplified)
            category_importance = {
                'Safety': 10, 'Critical': 9, 'Essential': 8, 'Important': 7,
                'Standard': 6, 'Optional': 5, 'Nice-to-have': 4
            }
            scenario_data['priority_score'] = scenario_data['common_name'].map(
                lambda x: category_importance.get(x, 6)  # Default to standard
            )
        else:  # cost_efficiency
            # Prioritize cost-efficient items (value per dollar)
            scenario_data['priority_score'] = 1 / scenario_data['Price']
        
        # Sort by priority and select items within budget
        sorted_data = scenario_data.sort_values('priority_score', ascending=False)
        
        selected_items = []
        cumulative_cost = 0
        
        for idx, item in sorted_data.iterrows():
            if cumulative_cost + item['Price'] <= budget_limit:
                selected_items.append(idx)
                cumulative_cost += item['Price']
            else:
                break
        
        # Create scenario dataset with only selected items
        scenario_data = scenario_data.loc[selected_items].copy()
        
        # Calculate results
        results = self._calculate_scenario_results(scenario_data, "Budget Constraint")
        results['budget_details'] = {
            'budget_limit': budget_limit,
            'budget_utilized': cumulative_cost,
            'budget_utilization_rate': cumulative_cost / budget_limit,
            'items_selected': len(selected_items),
            'items_excluded': len(self.base_data) - len(selected_items),
            'optimization_strategy': optimization_strategy
        }
        
        return results
    
    def simulate_inventory_optimization_scenario(self,
                                               reorder_frequency_change: Dict[str, float],
                                               carrying_cost_rate: float = 0.25) -> Dict:
        """
        Simulate inventory optimization strategies.
        
        Args:
            reorder_frequency_change: Dict of {category: frequency_multiplier}
            carrying_cost_rate: Annual carrying cost as percentage of item value
            
        Returns:
            Dictionary with scenario results
        """
        logger.info("Simulating inventory optimization scenario")
        
        scenario_data = self.base_data.copy()
        scenario_data['original_price'] = scenario_data['Price']
        scenario_data['carrying_cost_savings'] = 0.0
        
        for category, frequency_multiplier in reorder_frequency_change.items():
            category_mask = scenario_data['common_name'] == category
            
            # Calculate carrying cost impact
            # More frequent orders = lower inventory = lower carrying costs
            carrying_cost_reduction = (1 - 1/frequency_multiplier) * carrying_cost_rate
            carrying_cost_reduction = max(0, min(carrying_cost_reduction, carrying_cost_rate))
            
            scenario_data.loc[category_mask, 'carrying_cost_savings'] = carrying_cost_reduction
            scenario_data.loc[category_mask, 'Price'] = (
                scenario_data.loc[category_mask, 'original_price'] * (1 - carrying_cost_reduction)
            )
        
        # Calculate results
        results = self._calculate_scenario_results(scenario_data, "Inventory Optimization")
        results['inventory_details'] = {
            'reorder_frequency_changes': reorder_frequency_change,
            'carrying_cost_rate': carrying_cost_rate,
            'categories_optimized': len(reorder_frequency_change),
            'average_carrying_cost_savings': scenario_data['carrying_cost_savings'].mean()
        }
        
        return results
    
    def compare_scenarios(self, scenarios: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple scenarios side by side.
        
        Args:
            scenarios: List of scenario result dictionaries
            
        Returns:
            DataFrame comparing scenario metrics
        """
        comparison_data = []
        
        # Add baseline for comparison
        baseline = {
            'scenario_name': 'Current State (Baseline)',
            'total_value': self.baseline_metrics['total_value'],
            'total_savings': 0,
            'savings_percentage': 0,
            'items_affected': 0,
            'avg_price': self.baseline_metrics['avg_price'],
            'item_count': self.baseline_metrics['item_count']
        }
        comparison_data.append(baseline)
        
        # Add each scenario
        for scenario in scenarios:
            comparison_data.append({
                'scenario_name': scenario['scenario_name'],
                'total_value': scenario['total_value'],
                'total_savings': scenario['total_savings'],
                'savings_percentage': scenario['savings_percentage'],
                'items_affected': scenario.get('items_affected', 0),
                'avg_price': scenario['avg_price'],
                'item_count': scenario['item_count']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate ROI and efficiency metrics
        comparison_df['savings_per_item'] = (
            comparison_df['total_savings'] / comparison_df['items_affected'].replace(0, 1)
        )
        comparison_df['efficiency_score'] = (
            comparison_df['savings_percentage'] * comparison_df['items_affected'] / 100
        )
        
        return comparison_df
    
    def _calculate_baseline_metrics(self) -> Dict:
        """Calculate baseline metrics for comparison."""
        return {
            'total_value': self.base_data['Price'].sum(),
            'avg_price': self.base_data['Price'].mean(),
            'item_count': len(self.base_data),
            'category_count': self.base_data['common_name'].nunique(),
            'supplier_count': self.base_data['rep_office'].nunique()
        }
    
    def _calculate_scenario_results(self, scenario_data: pd.DataFrame, scenario_name: str) -> Dict:
        """Calculate standard metrics for any scenario."""
        
        # Calculate savings
        if 'original_price' in scenario_data.columns:
            total_original_value = scenario_data['original_price'].sum()
            total_new_value = scenario_data['Price'].sum()
            total_savings = total_original_value - total_new_value
            savings_percentage = (total_savings / total_original_value) * 100 if total_original_value > 0 else 0
        else:
            total_original_value = self.baseline_metrics['total_value']
            total_new_value = scenario_data['Price'].sum()
            total_savings = total_original_value - total_new_value
            savings_percentage = (total_savings / total_original_value) * 100 if total_original_value > 0 else 0
        
        return {
            'scenario_name': scenario_name,
            'total_value': total_new_value,
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'avg_price': scenario_data['Price'].mean(),
            'item_count': len(scenario_data),
            'scenario_data': scenario_data,
            'baseline_comparison': {
                'baseline_value': self.baseline_metrics['total_value'],
                'new_value': total_new_value,
                'absolute_savings': total_savings,
                'percentage_savings': savings_percentage
            }
        }

def create_scenario_simulator(df: pd.DataFrame) -> ProcurementScenarioSimulator:
    """
    Convenience function to create a scenario simulator.
    
    Args:
        df: DataFrame with procurement data
        
    Returns:
        Configured ProcurementScenarioSimulator instance
    """
    return ProcurementScenarioSimulator(df)

# Example usage
if __name__ == "__main__":
    print("Procurement Scenario Simulator - Ready for import")
    print("Usage: from src.scenario_simulator import create_scenario_simulator")