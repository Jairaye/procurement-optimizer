"""
NSN Dependency Mapper Module for Procurement Optimizer

This module identifies relationships and dependencies between NSN items,
helping procurement analysts understand which items are commonly used together
and optimize bundle purchasing strategies.

Author: Your Name
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import re
from collections import defaultdict, Counter
import logging
from itertools import combinations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSNDependencyMapper:
    """
    Maps dependencies and relationships between NSN items.
    
    Identifies:
    - Items commonly used together (bundle opportunities)
    - Complementary products (accessories, consumables)
    - Maintenance item relationships
    - Category dependencies
    """
    
    def __init__(self, similarity_threshold: float = 0.3):
        """
        Initialize the dependency mapper.
        
        Args:
            similarity_threshold: Minimum similarity score for relationship detection
        """
        self.similarity_threshold = similarity_threshold
        self.dependency_patterns = self._initialize_dependency_patterns()
        
        logger.info("Initialized NSN Dependency Mapper")
    
    def _initialize_dependency_patterns(self) -> Dict:
        """Initialize patterns for detecting item relationships."""
        return {
            'consumables': {
                'keywords': ['cartridge', 'toner', 'ink', 'paper', 'refill', 'replacement'],
                'relationship_type': 'consumable_for'
            },
            'accessories': {
                'keywords': ['case', 'cover', 'stand', 'mount', 'adapter', 'cable'],
                'relationship_type': 'accessory_for'
            },
            'maintenance': {
                'keywords': ['filter', 'belt', 'maintenance', 'service', 'repair', 'spare'],
                'relationship_type': 'maintenance_for'
            },
            'tools': {
                'keywords': ['wrench', 'socket', 'screwdriver', 'bit', 'tool'],
                'relationship_type': 'tool_for'
            },
            'sets_kits': {
                'keywords': ['set', 'kit', 'collection', 'assortment', 'package'],
                'relationship_type': 'bundle'
            }
        }
    
    def analyze_item_dependencies(self, df: pd.DataFrame) -> Dict:
        """
        Analyze dependencies across all items in the dataset.
        
        Args:
            df: DataFrame with NSN procurement data
            
        Returns:
            Dictionary with comprehensive dependency analysis
        """
        logger.info(f"Analyzing dependencies for {len(df)} items")
        
        # Find various types of relationships
        semantic_relationships = self._find_semantic_relationships(df)
        category_relationships = self._find_category_relationships(df)
        bundle_opportunities = self._find_bundle_opportunities(df)
        accessory_relationships = self._find_accessory_relationships(df)
        
        # Combine all relationships
        all_relationships = {
            'semantic_relationships': semantic_relationships,
            'category_relationships': category_relationships,
            'bundle_opportunities': bundle_opportunities,
            'accessory_relationships': accessory_relationships,
            'summary': {
                'total_relationships': (
                    len(semantic_relationships) + 
                    len(category_relationships) + 
                    len(bundle_opportunities) + 
                    len(accessory_relationships)
                ),
                'items_with_dependencies': len(set(
                    [rel['primary_nsn'] for rel in semantic_relationships] +
                    [rel['related_nsn'] for rel in semantic_relationships] +
                    [rel['primary_nsn'] for rel in category_relationships] +
                    [rel['related_nsn'] for rel in category_relationships]
                )),
                'potential_bundles': len(bundle_opportunities)
            }
        }
        
        logger.info(f"Found {all_relationships['summary']['total_relationships']} total relationships")
        return all_relationships
    
    def find_item_dependencies(self, df: pd.DataFrame, target_nsn: str) -> Dict:
        """
        Find dependencies for a specific NSN item.
        
        Args:
            df: DataFrame with NSN procurement data
            target_nsn: NSN to analyze dependencies for
            
        Returns:
            Dictionary with item-specific dependency information
        """
        target_item = df[df['NSN'] == target_nsn]
        if target_item.empty:
            logger.warning(f"NSN {target_nsn} not found in dataset")
            return {}
        
        target_row = target_item.iloc[0]
        logger.info(f"Finding dependencies for {target_nsn}: {target_row['common_name']}")
        
        dependencies = {
            'target_item': {
                'nsn': target_nsn,
                'name': target_row['common_name'],
                'price': target_row['Price'],
                'description': target_row.get('Description', '')
            },
            'consumables': self._find_consumables_for_item(df, target_row),
            'accessories': self._find_accessories_for_item(df, target_row),
            'maintenance_items': self._find_maintenance_items_for_item(df, target_row),
            'similar_items': self._find_similar_items(df, target_row),
            'bundle_candidates': self._find_bundle_candidates_for_item(df, target_row)
        }
        
        return dependencies
    
    def create_bundle_recommendations(self, df: pd.DataFrame, min_bundle_size: int = 2, max_bundle_size: int = 5) -> List[Dict]:
        """
        Create bundle purchasing recommendations.
        
        Args:
            df: DataFrame with NSN procurement data
            min_bundle_size: Minimum number of items in a bundle
            max_bundle_size: Maximum number of items in a bundle
            
        Returns:
            List of bundle recommendations with cost analysis
        """
        logger.info("Creating bundle recommendations")
        
        bundle_recommendations = []
        
        # Group items by functional categories
        category_groups = df.groupby('common_name')
        
        for category, group_items in category_groups:
            if len(group_items) >= min_bundle_size:
                # Find complementary items within category
                bundles = self._identify_category_bundles(group_items, min_bundle_size, max_bundle_size)
                bundle_recommendations.extend(bundles)
        
        # Find cross-category bundles (tools + consumables, etc.)
        cross_category_bundles = self._find_cross_category_bundles(df, min_bundle_size, max_bundle_size)
        bundle_recommendations.extend(cross_category_bundles)
        
        # Sort by potential savings
        bundle_recommendations.sort(key=lambda x: x['potential_savings'], reverse=True)
        
        logger.info(f"Created {len(bundle_recommendations)} bundle recommendations")
        return bundle_recommendations[:20]  # Return top 20 bundles
    
    def analyze_supplier_dependencies(self, df: pd.DataFrame) -> Dict:
        """
        Analyze dependencies between suppliers and product categories.
        
        Args:
            df: DataFrame with NSN procurement data
            
        Returns:
            Dictionary with supplier dependency analysis
        """
        logger.info("Analyzing supplier dependencies")
        
        # Supplier-category relationships
        supplier_categories = df.groupby('rep_office')['common_name'].apply(set).to_dict()
        
        # Find suppliers with overlapping capabilities
        supplier_overlaps = {}
        suppliers = list(supplier_categories.keys())
        
        for i, supplier1 in enumerate(suppliers):
            overlaps = []
            for supplier2 in suppliers[i+1:]:
                overlap = supplier_categories[supplier1] & supplier_categories[supplier2]
                if len(overlap) > 0:
                    overlaps.append({
                        'supplier': supplier2,
                        'common_categories': list(overlap),
                        'overlap_count': len(overlap)
                    })
            
            if overlaps:
                supplier_overlaps[supplier1] = sorted(overlaps, key=lambda x: x['overlap_count'], reverse=True)
        
        # Supplier specialization analysis
        supplier_specialization = {}
        for supplier, categories in supplier_categories.items():
            supplier_items = df[df['rep_office'] == supplier]
            specialization = {
                'category_count': len(categories),
                'item_count': len(supplier_items),
                'avg_price': supplier_items['Price'].mean(),
                'total_value': supplier_items['Price'].sum(),
                'specialization_score': len(supplier_items) / len(categories) if len(categories) > 0 else 0
            }
            supplier_specialization[supplier] = specialization
        
        return {
            'supplier_overlaps': supplier_overlaps,
            'supplier_specialization': supplier_specialization,
            'consolidation_opportunities': self._identify_consolidation_opportunities(supplier_overlaps, df)
        }
    
    def _find_semantic_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Find relationships based on item names and descriptions."""
        relationships = []
        
        # Check each item against dependency patterns
        for idx, item in df.iterrows():
            item_text = f"{item['common_name']} {item.get('Description', '')}".lower()
            
            for pattern_name, pattern_info in self.dependency_patterns.items():
                if any(keyword in item_text for keyword in pattern_info['keywords']):
                    # Find potential related items
                    related_items = self._find_related_items_by_pattern(df, item, pattern_info)
                    
                    for related_item in related_items:
                        relationships.append({
                            'primary_nsn': item['NSN'],
                            'primary_name': item['common_name'],
                            'related_nsn': related_item['NSN'],
                            'related_name': related_item['common_name'],
                            'relationship_type': pattern_info['relationship_type'],
                            'confidence_score': self._calculate_relationship_confidence(item, related_item)
                        })
        
        return relationships
    
    def _find_category_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Find relationships between different product categories."""
        relationships = []
        
        # Common category relationships
        category_pairs = [
            ('Toner Cartridge', 'Printer'),
            ('Socket', 'Wrench'),
            ('Drill Bit', 'Drill'),
            ('Case', 'Equipment'),
            ('Filter', 'Machine')
        ]
        
        for primary_cat, related_cat in category_pairs:
            primary_items = df[df['common_name'].str.contains(primary_cat, case=False, na=False)]
            related_items = df[df['common_name'].str.contains(related_cat, case=False, na=False)]
            
            for _, primary_item in primary_items.iterrows():
                for _, related_item in related_items.iterrows():
                    if primary_item['NSN'] != related_item['NSN']:
                        relationships.append({
                            'primary_nsn': primary_item['NSN'],
                            'primary_name': primary_item['common_name'],
                            'related_nsn': related_item['NSN'],
                            'related_name': related_item['common_name'],
                            'relationship_type': 'category_dependency',
                            'confidence_score': 0.7
                        })
        
        return relationships
    
    def _find_bundle_opportunities(self, df: pd.DataFrame) -> List[Dict]:
        """Find opportunities for bundling items together."""
        bundle_opportunities = []
        
        # Group by category and find items that could be bundled
        for category in df['common_name'].unique():
            category_items = df[df['common_name'] == category]
            
            if len(category_items) >= 2:
                # Sort by price to create value bundles
                sorted_items = category_items.sort_values('Price')
                
                # Create bundles of 2-4 items
                for bundle_size in [2, 3, 4]:
                    if len(sorted_items) >= bundle_size:
                        # Take lowest priced items for bundle
                        bundle_items = sorted_items.head(bundle_size)
                        
                        bundle_value = bundle_items['Price'].sum()
                        estimated_bundle_discount = 0.05 + (bundle_size - 2) * 0.02  # 5-9% discount
                        discounted_price = bundle_value * (1 - estimated_bundle_discount)
                        
                        bundle_opportunities.append({
                            'bundle_id': f"{category}_{bundle_size}_item_bundle",
                            'category': category,
                            'items': bundle_items[['NSN', 'common_name', 'Price']].to_dict('records'),
                            'original_total': bundle_value,
                            'discounted_total': discounted_price,
                            'potential_savings': bundle_value - discounted_price,
                            'bundle_size': bundle_size,
                            'estimated_discount': estimated_bundle_discount
                        })
        
        return bundle_opportunities
    
    def _find_accessory_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Find accessory relationships between items."""
        relationships = []
        
        # Common accessory patterns
        accessory_keywords = ['case', 'cover', 'stand', 'mount', 'adapter', 'cable', 'strap']
        
        for idx, item in df.iterrows():
            item_text = item['common_name'].lower()
            
            if any(keyword in item_text for keyword in accessory_keywords):
                # This item might be an accessory
                # Find potential main items it could be an accessory for
                potential_mains = df[
                    (df.index != idx) &  # Different item
                    (~df['common_name'].str.contains('|'.join(accessory_keywords), case=False, na=False))  # Not another accessory
                ]
                
                # Simple matching based on similar words
                item_words = set(item_text.split())
                
                for _, potential_main in potential_mains.head(5).iterrows():  # Limit to avoid too many relationships
                    main_words = set(potential_main['common_name'].lower().split())
                    
                    # Check for word overlap
                    word_overlap = len(item_words & main_words)
                    if word_overlap >= 1:
                        relationships.append({
                            'primary_nsn': potential_main['NSN'],
                            'primary_name': potential_main['common_name'],
                            'related_nsn': item['NSN'],
                            'related_name': item['common_name'],
                            'relationship_type': 'accessory',
                            'confidence_score': min(word_overlap * 0.3, 0.9)
                        })
        
        return relationships
    
    def _find_consumables_for_item(self, df: pd.DataFrame, target_item: pd.Series) -> List[Dict]:
        """Find consumable items for a target item."""
        consumables = []
        
        consumable_keywords = ['cartridge', 'toner', 'ink', 'paper', 'refill', 'replacement']
        target_words = set(target_item['common_name'].lower().split())
        
        for idx, item in df.iterrows():
            if item['NSN'] != target_item['NSN']:
                item_text = item['common_name'].lower()
                
                if any(keyword in item_text for keyword in consumable_keywords):
                    item_words = set(item_text.split())
                    word_overlap = len(target_words & item_words)
                    
                    if word_overlap >= 1:
                        consumables.append({
                            'nsn': item['NSN'],
                            'name': item['common_name'],
                            'price': item['Price'],
                            'confidence': min(word_overlap * 0.4, 0.9)
                        })
        
        return sorted(consumables, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _find_accessories_for_item(self, df: pd.DataFrame, target_item: pd.Series) -> List[Dict]:
        """Find accessory items for a target item."""
        accessories = []
        
        accessory_keywords = ['case', 'cover', 'stand', 'mount', 'adapter', 'cable']
        target_words = set(target_item['common_name'].lower().split())
        
        for idx, item in df.iterrows():
            if item['NSN'] != target_item['NSN']:
                item_text = item['common_name'].lower()
                
                if any(keyword in item_text for keyword in accessory_keywords):
                    item_words = set(item_text.split())
                    word_overlap = len(target_words & item_words)
                    
                    if word_overlap >= 1:
                        accessories.append({
                            'nsn': item['NSN'],
                            'name': item['common_name'],
                            'price': item['Price'],
                            'confidence': min(word_overlap * 0.4, 0.9)
                        })
        
    def _find_maintenance_items_for_item(self, df: pd.DataFrame, target_item: pd.Series) -> List[Dict]:
        """Find maintenance items for a target item."""
        maintenance_items = []
        
        maintenance_keywords = ['filter', 'belt', 'maintenance', 'service', 'repair', 'spare', 'part']
        target_words = set(target_item['common_name'].lower().split())
        
        for idx, item in df.iterrows():
            if item['NSN'] != target_item['NSN']:
                item_text = item['common_name'].lower()
                
                if any(keyword in item_text for keyword in maintenance_keywords):
                    item_words = set(item_text.split())
                    word_overlap = len(target_words & item_words)
                    
                    if word_overlap >= 1:
                        maintenance_items.append({
                            'nsn': item['NSN'],
                            'name': item['common_name'],
                            'price': item['Price'],
                            'confidence': min(word_overlap * 0.4, 0.9)
                        })
        
        return sorted(maintenance_items, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _find_similar_items(self, df: pd.DataFrame, target_item: pd.Series) -> List[Dict]:
        """Find similar items to the target item."""
        similar_items = []
        
        # Items in same category but different NSNs
        category_items = df[
            (df['common_name'] == target_item['common_name']) &
            (df['NSN'] != target_item['NSN'])
        ]
        
        for idx, item in category_items.iterrows():
            price_difference = abs(item['Price'] - target_item['Price'])
            price_similarity = 1 - min(price_difference / max(item['Price'], target_item['Price']), 1)
            
            similar_items.append({
                'nsn': item['NSN'],
                'name': item['common_name'],
                'price': item['Price'],
                'price_difference': item['Price'] - target_item['Price'],
                'similarity': price_similarity
            })
        
        return sorted(similar_items, key=lambda x: x['similarity'], reverse=True)[:5]
    
    def _find_bundle_candidates_for_item(self, df: pd.DataFrame, target_item: pd.Series) -> List[Dict]:
        """Find items that could be bundled with the target item."""
        bundle_candidates = []
        
        # Find items in same category with complementary pricing
        category_items = df[
            (df['common_name'] == target_item['common_name']) &
            (df['NSN'] != target_item['NSN'])
        ]
        
        # Sort by price to find good bundle combinations
        sorted_items = category_items.sort_values('Price')
        
        # Select items with prices in a reasonable range
        price_range = target_item['Price'] * 0.5  # Within 50% of target price
        suitable_items = sorted_items[
            abs(sorted_items['Price'] - target_item['Price']) <= price_range
        ]
        
        for idx, item in suitable_items.head(3).iterrows():
            bundle_value = target_item['Price'] + item['Price']
            estimated_discount = 0.05  # 5% bundle discount
            
            bundle_candidates.append({
                'nsn': item['NSN'],
                'name': item['common_name'],
                'price': item['Price'],
                'bundle_total': bundle_value,
                'estimated_savings': bundle_value * estimated_discount,
                'bundle_score': 0.8  # High score for same category bundles
            })
        
        return bundle_candidates
    
    def _identify_category_bundles(self, group_items: pd.DataFrame, min_size: int, max_size: int) -> List[Dict]:
        """Identify bundle opportunities within a category."""
        bundles = []
        
        if len(group_items) < min_size:
            return bundles
        
        # Sort by price for logical bundle creation
        sorted_items = group_items.sort_values('Price')
        
        # Create bundles of different sizes
        for bundle_size in range(min_size, min(max_size + 1, len(sorted_items) + 1)):
            # Take consecutive items to create price-coherent bundles
            for start_idx in range(len(sorted_items) - bundle_size + 1):
                bundle_items = sorted_items.iloc[start_idx:start_idx + bundle_size]
                
                bundle_value = bundle_items['Price'].sum()
                estimated_discount = 0.03 + (bundle_size - min_size) * 0.01  # 3-6% discount
                
                bundles.append({
                    'bundle_id': f"category_bundle_{start_idx}_{bundle_size}",
                    'category': bundle_items.iloc[0]['common_name'],
                    'items': bundle_items[['NSN', 'common_name', 'Price']].to_dict('records'),
                    'original_total': bundle_value,
                    'discounted_total': bundle_value * (1 - estimated_discount),
                    'potential_savings': bundle_value * estimated_discount,
                    'bundle_size': bundle_size,
                    'bundle_type': 'category_bundle'
                })
        
        return bundles
    
    def _find_cross_category_bundles(self, df: pd.DataFrame, min_size: int, max_size: int) -> List[Dict]:
        """Find bundle opportunities across different categories."""
        cross_bundles = []
        
        # Define logical cross-category relationships
        cross_category_pairs = [
            (['Toner Cartridge', 'HP Toner Cartridge', 'Lexmark Toner'], ['Paper', 'Copy Paper']),
            (['Socket', 'Wrench'], ['Tool Case', 'Case']),
            (['Drill Bit', 'Bit'], ['Drill', 'Power Tool']),
            (['Filter'], ['Maintenance', 'Service Kit'])
        ]
        
        for primary_categories, secondary_categories in cross_category_pairs:
            primary_items = df[df['common_name'].str.contains('|'.join(primary_categories), case=False, na=False)]
            secondary_items = df[df['common_name'].str.contains('|'.join(secondary_categories), case=False, na=False)]
            
            # Create small cross-category bundles
            for _, primary_item in primary_items.head(3).iterrows():
                for _, secondary_item in secondary_items.head(2).iterrows():
                    bundle_value = primary_item['Price'] + secondary_item['Price']
                    estimated_discount = 0.04  # 4% cross-category discount
                    
                    cross_bundles.append({
                        'bundle_id': f"cross_bundle_{primary_item['NSN']}_{secondary_item['NSN']}",
                        'category': 'Cross-Category',
                        'items': [
                            {'NSN': primary_item['NSN'], 'common_name': primary_item['common_name'], 'Price': primary_item['Price']},
                            {'NSN': secondary_item['NSN'], 'common_name': secondary_item['common_name'], 'Price': secondary_item['Price']}
                        ],
                        'original_total': bundle_value,
                        'discounted_total': bundle_value * (1 - estimated_discount),
                        'potential_savings': bundle_value * estimated_discount,
                        'bundle_size': 2,
                        'bundle_type': 'cross_category_bundle'
                    })
        
        return cross_bundles[:10]  # Limit cross-category bundles
    
    def _identify_consolidation_opportunities(self, supplier_overlaps: Dict, df: pd.DataFrame) -> List[Dict]:
        """Identify supplier consolidation opportunities."""
        consolidation_opportunities = []
        
        for primary_supplier, overlaps in supplier_overlaps.items():
            for overlap_info in overlaps[:3]:  # Top 3 overlapping suppliers
                secondary_supplier = overlap_info['supplier']
                common_categories = overlap_info['common_categories']
                
                # Calculate potential savings from consolidation
                primary_items = df[
                    (df['rep_office'] == primary_supplier) &
                    (df['common_name'].isin(common_categories))
                ]
                secondary_items = df[
                    (df['rep_office'] == secondary_supplier) &
                    (df['common_name'].isin(common_categories))
                ]
                
                if len(primary_items) > 0 and len(secondary_items) > 0:
                    primary_avg_price = primary_items.groupby('common_name')['Price'].mean()
                    secondary_avg_price = secondary_items.groupby('common_name')['Price'].mean()
                    
                    # Calculate potential savings by choosing better prices
                    potential_savings = 0
                    for category in common_categories:
                        if category in primary_avg_price and category in secondary_avg_price:
                            price_diff = abs(primary_avg_price[category] - secondary_avg_price[category])
                            category_volume = len(primary_items[primary_items['common_name'] == category])
                            potential_savings += price_diff * category_volume * 0.5  # Conservative estimate
                    
                    consolidation_opportunities.append({
                        'primary_supplier': primary_supplier,
                        'secondary_supplier': secondary_supplier,
                        'common_categories': common_categories,
                        'overlap_count': len(common_categories),
                        'potential_annual_savings': potential_savings,
                        'consolidation_score': len(common_categories) * potential_savings / 1000  # Normalized score
                    })
        
        return sorted(consolidation_opportunities, key=lambda x: x['consolidation_score'], reverse=True)[:10]
    
    def _find_related_items_by_pattern(self, df: pd.DataFrame, item: pd.Series, pattern_info: Dict) -> List[pd.Series]:
        """Find items related to a given item based on patterns."""
        related_items = []
        
        item_words = set(item['common_name'].lower().split())
        
        for idx, potential_related in df.iterrows():
            if potential_related['NSN'] != item['NSN']:
                related_words = set(potential_related['common_name'].lower().split())
                
                # Check for word overlap (simple relationship detection)
                word_overlap = len(item_words & related_words)
                
                if word_overlap >= 1:  # At least one common word
                    related_items.append(potential_related)
        
        return related_items[:5]  # Limit to top 5 related items
    
    def _calculate_relationship_confidence(self, item1: pd.Series, item2: pd.Series) -> float:
        """Calculate confidence score for a relationship."""
        
        # Word overlap in names
        words1 = set(item1['common_name'].lower().split())
        words2 = set(item2['common_name'].lower().split())
        word_overlap = len(words1 & words2)
        
        # Price similarity (inverse of price difference)
        price_diff = abs(item1['Price'] - item2['Price'])
        max_price = max(item1['Price'], item2['Price'])
        price_similarity = 1 - min(price_diff / max_price, 1) if max_price > 0 else 0
        
        # Category match
        category_match = 1 if item1['common_name'] == item2['common_name'] else 0.5
        
        # Weighted confidence score
        confidence = (
            word_overlap * 0.4 +
            price_similarity * 0.3 +
            category_match * 0.3
        )
        
        return min(confidence, 1.0)

def analyze_nsn_dependencies(df: pd.DataFrame) -> Dict:
    """
    Convenience function to perform comprehensive dependency analysis.
    
    Args:
        df: DataFrame with NSN procurement data
        
    Returns:
        Dictionary with comprehensive dependency analysis
    """
    mapper = NSNDependencyMapper()
    return mapper.analyze_item_dependencies(df)

def create_bundle_recommendations(df: pd.DataFrame, min_bundle_size: int = 2) -> List[Dict]:
    """
    Convenience function to create bundle recommendations.
    
    Args:
        df: DataFrame with NSN procurement data
        min_bundle_size: Minimum number of items in a bundle
        
    Returns:
        List of bundle recommendations
    """
    mapper = NSNDependencyMapper()
    return mapper.create_bundle_recommendations(df, min_bundle_size)

# Example usage
if __name__ == "__main__":
    print("NSN Dependency Mapper - Ready for import")
    print("Usage: from src.nsn_dependency_mapper import analyze_nsn_dependencies")