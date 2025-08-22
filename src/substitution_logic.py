"""
Substitution Logic Module for Procurement Optimizer

This module identifies alternative products that could substitute for expensive items,
helping procurement analysts find cost-saving opportunities while maintaining functionality.

Author: Your Name
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductSubstitutionEngine:
    """
    Identifies potential product substitutions based on:
    - Similar product names and descriptions
    - Same functional category
    - Compatible specifications
    - Significant price differences
    """
    
    def __init__(self, similarity_threshold: float = 0.6, max_price_ratio: float = 0.8):
        """
        Initialize the substitution engine.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) for substitution candidates
            max_price_ratio: Maximum price ratio for substitution (0.8 = 20% savings minimum)
        """
        self.similarity_threshold = similarity_threshold
        self.max_price_ratio = max_price_ratio
        
        # Common substitution patterns
        self.substitution_patterns = {
            'brand_variations': [
                (r'\b(HP|Hewlett.?Packard)\b', 'printer_supplies'),
                (r'\b(Canon|Lexmark|Brother|Epson)\b', 'printer_supplies'),
                (r'\b(3M|Scotch)\b', 'office_supplies'),
                (r'\b(Pelican|Storm)\b', 'cases_containers'),
            ],
            'generic_equivalents': [
                (r'\b(OEM|Original|Genuine)\b', 'branded'),
                (r'\b(Compatible|Generic|Remanufactured)\b', 'aftermarket'),
            ],
            'size_variations': [
                (r'\b(\d+(?:\.\d+)?)\s?(inch|in|mm|cm)\b', 'dimensional'),
                (r'\b(Small|Medium|Large|XL|XXL)\b', 'size_category'),
            ]
        }
        
        logger.info("Initialized ProductSubstitutionEngine")
    
    def find_substitutions(self, df: pd.DataFrame, target_nsn: str) -> pd.DataFrame:
        """
        Find potential substitutions for a specific NSN.
        
        Args:
            df: DataFrame with procurement data
            target_nsn: NSN to find substitutions for
            
        Returns:
            DataFrame of potential substitutions ranked by savings potential
        """
        # Get target item
        target_item = df[df['NSN'] == target_nsn]
        if target_item.empty:
            logger.warning(f"NSN {target_nsn} not found in dataset")
            return pd.DataFrame()
        
        target_row = target_item.iloc[0]
        logger.info(f"Finding substitutions for {target_nsn}: {target_row['common_name']}")
        
        # Find potential candidates
        candidates = self._find_substitution_candidates(df, target_row)
        
        # Score and rank candidates
        if not candidates.empty:
            scored_candidates = self._score_substitutions(candidates, target_row)
            return scored_candidates.sort_values('substitution_score', ascending=False)
        else:
            logger.info(f"No substitution candidates found for {target_nsn}")
            return pd.DataFrame()
    
    def find_category_substitutions(self, df: pd.DataFrame, category: str, top_n: int = 20) -> pd.DataFrame:
        """
        Find substitution opportunities within a product category.
        
        Args:
            df: DataFrame with procurement data
            category: Product category to analyze
            top_n: Number of top opportunities to return
            
        Returns:
            DataFrame of substitution opportunities ranked by potential savings
        """
        category_items = df[df['common_name'] == category].copy()
        
        if len(category_items) < 2:
            logger.info(f"Not enough items in category '{category}' for substitution analysis")
            return pd.DataFrame()
        
        logger.info(f"Analyzing substitution opportunities in category: {category}")
        
        # Sort by price (highest first)
        category_items = category_items.sort_values('Price', ascending=False)
        
        substitution_opportunities = []
        
        # Compare expensive items with cheaper alternatives
        for i, (idx, expensive_item) in enumerate(category_items.head(10).iterrows()):
            # Find cheaper alternatives
            cheaper_alternatives = category_items[
                (category_items['Price'] < expensive_item['Price'] * self.max_price_ratio) &
                (category_items.index != idx)
            ]
            
            if not cheaper_alternatives.empty:
                for alt_idx, alternative in cheaper_alternatives.iterrows():
                    similarity = self._calculate_similarity(expensive_item, alternative)
                    
                    if similarity >= self.similarity_threshold:
                        potential_savings = expensive_item['Price'] - alternative['Price']
                        
                        substitution_opportunities.append({
                            'expensive_nsn': expensive_item['NSN'],
                            'expensive_name': expensive_item['common_name'],
                            'expensive_price': expensive_item['Price'],
                            'alternative_nsn': alternative['NSN'],
                            'alternative_name': alternative['common_name'],
                            'alternative_price': alternative['Price'],
                            'potential_savings': potential_savings,
                            'savings_percentage': (potential_savings / expensive_item['Price']) * 100,
                            'similarity_score': similarity,
                            'substitution_score': self._calculate_substitution_score(
                                similarity, potential_savings, expensive_item['Price']
                            )
                        })
        
        if substitution_opportunities:
            opportunities_df = pd.DataFrame(substitution_opportunities)
            return opportunities_df.sort_values('substitution_score', ascending=False).head(top_n)
        else:
            return pd.DataFrame()
    
    def analyze_brand_substitutions(self, df: pd.DataFrame) -> Dict:
        """
        Analyze potential savings from brand substitutions (OEM vs Generic).
        
        Args:
            df: DataFrame with procurement data
            
        Returns:
            Dictionary with brand substitution analysis
        """
        logger.info("Analyzing brand substitution opportunities")
        
        # Identify branded vs generic items
        df['brand_type'] = df.apply(self._classify_brand_type, axis=1)
        
        # Group by functional similarity
        brand_analysis = {}
        
        # Find OEM items that have generic equivalents
        oem_items = df[df['brand_type'] == 'branded']
        generic_items = df[df['brand_type'] == 'aftermarket']
        
        substitution_pairs = []
        
        for _, oem_item in oem_items.iterrows():
            # Find similar generic items
            similar_generics = generic_items[
                generic_items['common_name'] == oem_item['common_name']
            ]
            
            if not similar_generics.empty:
                best_generic = similar_generics.loc[similar_generics['Price'].idxmin()]
                
                if best_generic['Price'] < oem_item['Price'] * self.max_price_ratio:
                    potential_savings = oem_item['Price'] - best_generic['Price']
                    
                    substitution_pairs.append({
                        'oem_nsn': oem_item['NSN'],
                        'oem_price': oem_item['Price'],
                        'generic_nsn': best_generic['NSN'],
                        'generic_price': best_generic['Price'],
                        'category': oem_item['common_name'],
                        'potential_savings': potential_savings,
                        'savings_percentage': (potential_savings / oem_item['Price']) * 100
                    })
        
        if substitution_pairs:
            pairs_df = pd.DataFrame(substitution_pairs)
            total_savings = pairs_df['potential_savings'].sum()
            
            brand_analysis = {
                'total_potential_savings': total_savings,
                'number_of_opportunities': len(pairs_df),
                'average_savings_percentage': pairs_df['savings_percentage'].mean(),
                'top_categories': pairs_df.groupby('category')['potential_savings'].sum().sort_values(ascending=False).head(5),
                'substitution_pairs': pairs_df.sort_values('potential_savings', ascending=False)
            }
        else:
            brand_analysis = {
                'total_potential_savings': 0,
                'number_of_opportunities': 0,
                'average_savings_percentage': 0,
                'top_categories': pd.Series(),
                'substitution_pairs': pd.DataFrame()
            }
        
        logger.info(f"Found {brand_analysis['number_of_opportunities']} brand substitution opportunities")
        return brand_analysis
    
    def _find_substitution_candidates(self, df: pd.DataFrame, target_item: pd.Series) -> pd.DataFrame:
        """Find potential substitution candidates for a target item."""
        
        # Filter candidates by category and price
        candidates = df[
            (df['common_name'] == target_item['common_name']) &  # Same category
            (df['Price'] < target_item['Price'] * self.max_price_ratio) &  # Significantly cheaper
            (df['NSN'] != target_item['NSN'])  # Different item
        ].copy()
        
        return candidates
    
    def _calculate_similarity(self, item1: pd.Series, item2: pd.Series) -> float:
        """Calculate similarity between two items based on name and description."""
        
        # Text similarity
        name_similarity = SequenceMatcher(None, 
            item1['common_name'].lower(), 
            item2['common_name'].lower()
        ).ratio()
        
        desc_similarity = 0
        if pd.notna(item1.get('Description')) and pd.notna(item2.get('Description')):
            desc_similarity = SequenceMatcher(None,
                item1['Description'].lower()[:200],  # First 200 chars
                item2['Description'].lower()[:200]
            ).ratio()
        
        # Unit compatibility
        unit_similarity = 1.0 if item1.get('UI') == item2.get('UI') else 0.7
        
        # Weighted average
        overall_similarity = (
            name_similarity * 0.4 +
            desc_similarity * 0.4 +
            unit_similarity * 0.2
        )
        
        return overall_similarity
    
    def _calculate_substitution_score(self, similarity: float, savings: float, original_price: float) -> float:
        """Calculate overall substitution score combining similarity and savings."""
        
        savings_score = min(savings / original_price, 0.5) * 2  # Normalize to 0-1
        
        # Weighted combination: 60% similarity, 40% savings
        substitution_score = (similarity * 0.6) + (savings_score * 0.4)
        
        return substitution_score
    
    def _score_substitutions(self, candidates: pd.DataFrame, target_item: pd.Series) -> pd.DataFrame:
        """Score substitution candidates."""
        
        scored_candidates = candidates.copy()
        
        # Calculate similarities and scores
        scored_candidates['similarity_score'] = scored_candidates.apply(
            lambda row: self._calculate_similarity(target_item, row), axis=1
        )
        
        scored_candidates['potential_savings'] = target_item['Price'] - scored_candidates['Price']
        scored_candidates['savings_percentage'] = (
            scored_candidates['potential_savings'] / target_item['Price'] * 100
        )
        
        scored_candidates['substitution_score'] = scored_candidates.apply(
            lambda row: self._calculate_substitution_score(
                row['similarity_score'], 
                row['potential_savings'], 
                target_item['Price']
            ), axis=1
        )
        
        # Filter by minimum similarity threshold
        scored_candidates = scored_candidates[
            scored_candidates['similarity_score'] >= self.similarity_threshold
        ]
        
        return scored_candidates
    
    def _classify_brand_type(self, item: pd.Series) -> str:
        """Classify item as branded, aftermarket, or unknown."""
        
        text_to_check = f"{item['common_name']} {item.get('Description', '')}".lower()
        
        # Check for aftermarket indicators
        aftermarket_keywords = ['remanufactured', 'compatible', 'generic', 'aftermarket']
        if any(keyword in text_to_check for keyword in aftermarket_keywords):
            return 'aftermarket'
        
        # Check for branded indicators
        branded_keywords = ['oem', 'original', 'genuine', 'hp ', 'canon', 'lexmark', '3m']
        if any(keyword in text_to_check for keyword in branded_keywords):
            return 'branded'
        
        return 'unknown'

def find_product_substitutions(df: pd.DataFrame, nsn: str, max_results: int = 10) -> pd.DataFrame:
    """
    Convenience function to find substitutions for a specific NSN.
    
    Args:
        df: DataFrame with procurement data
        nsn: NSN to find substitutions for
        max_results: Maximum number of results to return
        
    Returns:
        DataFrame of potential substitutions
    """
    engine = ProductSubstitutionEngine()
    substitutions = engine.find_substitutions(df, nsn)
    return substitutions.head(max_results)

# Example usage
if __name__ == "__main__":
    print("Product Substitution Engine - Ready for import")
    print("Usage: from src.substitution_logic import find_product_substitutions")