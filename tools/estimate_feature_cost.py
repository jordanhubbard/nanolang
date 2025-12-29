#!/usr/bin/env python3
"""Estimate implementation cost for new NanoLang features.

This tool quantifies the dual-implementation overhead where every feature
requires implementation in both C and NanoLang compilers.
"""

import sys
import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ComponentCost:
    """Cost estimate for a compiler component."""
    c_hours: float
    nano_hours: float
    complexity: str  # "trivial", "simple", "moderate", "complex", "expert"
    
    @property
    def total_hours(self) -> float:
        return self.c_hours + self.nano_hours
    
    @property
    def multiplier(self) -> float:
        """How much more expensive than single implementation."""
        single_impl = max(self.c_hours, self.nano_hours)
        return self.total_hours / single_impl if single_impl > 0 else 2.0

# Cost matrix based on historical data and component complexity
COST_MATRIX = {
    "lexer": {
        "keyword": ComponentCost(0.5, 0.5, "trivial"),
        "token": ComponentCost(1.0, 1.0, "simple"),
        "syntax": ComponentCost(2.0, 3.0, "moderate"),
    },
    "parser": {
        "expression": ComponentCost(2.0, 3.0, "simple"),
        "statement": ComponentCost(4.0, 5.0, "moderate"),
        "declaration": ComponentCost(6.0, 8.0, "moderate"),
        "complex_syntax": ComponentCost(12.0, 16.0, "complex"),
    },
    "typechecker": {
        "simple_check": ComponentCost(2.0, 3.0, "simple"),
        "type_inference": ComponentCost(8.0, 10.0, "complex"),
        "generic_instantiation": ComponentCost(12.0, 15.0, "complex"),
        "constraint_solving": ComponentCost(20.0, 25.0, "expert"),
    },
    "transpiler": {
        "simple_codegen": ComponentCost(2.0, 3.0, "simple"),
        "expression_codegen": ComponentCost(4.0, 6.0, "moderate"),
        "control_flow": ComponentCost(6.0, 8.0, "moderate"),
        "complex_transform": ComponentCost(10.0, 12.0, "complex"),
    },
    "runtime": {
        "helper_function": ComponentCost(1.0, 0.0, "trivial"),  # Only C
        "gc_integration": ComponentCost(4.0, 0.0, "moderate"),  # Only C
        "ffi_binding": ComponentCost(3.0, 2.0, "simple"),
    },
    "stdlib": {
        "simple_function": ComponentCost(0.0, 2.0, "simple"),  # Only Nano
        "complex_function": ComponentCost(0.0, 4.0, "moderate"),
        "module": ComponentCost(0.0, 8.0, "complex"),
    },
    "testing": {
        "unit_test": ComponentCost(1.0, 1.0, "simple"),
        "integration_test": ComponentCost(2.0, 2.0, "moderate"),
        "shadow_tests": ComponentCost(0.0, 2.0, "simple"),
        "differential_tests": ComponentCost(3.0, 0.0, "moderate"),
    },
    "docs": {
        "spec_update": ComponentCost(1.0, 0.0, "trivial"),
        "memory_md_update": ComponentCost(1.0, 0.0, "trivial"),
        "examples": ComponentCost(2.0, 0.0, "simple"),
    }
}

class FeatureEstimate:
    """Estimate for a complete feature."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.components: List[tuple[str, str, ComponentCost]] = []
        self.breaking_change = False
        self.affects_bootstrap = False
    
    def add_component(self, category: str, component: str):
        """Add a component to the estimate."""
        if category not in COST_MATRIX:
            print(f"Warning: Unknown category '{category}'", file=sys.stderr)
            return
        if component not in COST_MATRIX[category]:
            print(f"Warning: Unknown component '{component}' in '{category}'", file=sys.stderr)
            return
        cost = COST_MATRIX[category][component]
        self.components.append((category, component, cost))
    
    def total_c_hours(self) -> float:
        return sum(cost.c_hours for _, _, cost in self.components)
    
    def total_nano_hours(self) -> float:
        return sum(cost.nano_hours for _, _, cost in self.components)
    
    def total_hours(self) -> float:
        return self.total_c_hours() + self.total_nano_hours()
    
    def multiplier(self) -> float:
        """Dual-implementation multiplier."""
        single = max(self.total_c_hours(), self.total_nano_hours())
        return self.total_hours() / single if single > 0 else 2.0
    
    def equivalent_single_impl_hours(self) -> float:
        """What this would cost in a single-implementation language."""
        return max(self.total_c_hours(), self.total_nano_hours())
    
    def overhead_hours(self) -> float:
        """Extra hours due to dual implementation."""
        return self.total_hours() - self.equivalent_single_impl_hours()
    
    def complexity_distribution(self) -> Dict[str, int]:
        """Count of components by complexity."""
        dist = {"trivial": 0, "simple": 0, "moderate": 0, "complex": 0, "expert": 0}
        for _, _, cost in self.components:
            dist[cost.complexity] += 1
        return dist
    
    def print_estimate(self):
        """Print detailed cost estimate."""
        print(f"\n{'='*70}")
        print(f"FEATURE COST ESTIMATE: {self.name}")
        print(f"{'='*70}")
        if self.description:
            print(f"\nDescription: {self.description}")
        
        print(f"\n{'Component':<30} {'C Hours':<12} {'Nano Hours':<12} {'Complexity'}")
        print("-" * 70)
        
        for category, component, cost in self.components:
            label = f"{category}/{component}"
            print(f"{label:<30} {cost.c_hours:<12.1f} {cost.nano_hours:<12.1f} {cost.complexity}")
        
        print("-" * 70)
        print(f"{'TOTALS':<30} {self.total_c_hours():<12.1f} {self.total_nano_hours():<12.1f}")
        print()
        
        print(f"Total Implementation Time:     {self.total_hours():.1f} hours")
        print(f"Equivalent Single-Impl Time:   {self.equivalent_single_impl_hours():.1f} hours")
        print(f"Dual-Impl Overhead:            {self.overhead_hours():.1f} hours ({self.multiplier():.2f}x)")
        
        dist = self.complexity_distribution()
        print(f"\nComplexity Distribution:")
        for level in ["trivial", "simple", "moderate", "complex", "expert"]:
            if dist[level] > 0:
                print(f"  {level.capitalize():<12} {dist[level]} components")
        
        if self.breaking_change:
            print(f"\n⚠️  WARNING: This is a BREAKING CHANGE")
            print(f"   Add migration guide and deprecation period")
        
        if self.affects_bootstrap:
            print(f"\n⚠️  WARNING: Affects bootstrap chain")
            print(f"   Requires careful staged implementation")
        
        # Recommendation
        complexity_score = sum(
            {"trivial": 1, "simple": 2, "moderate": 3, "complex": 4, "expert": 5}[cost.complexity]
            for _, _, cost in self.components
        )
        avg_complexity = complexity_score / len(self.components) if self.components else 0
        
        print(f"\nRECOMMENDATION:")
        if self.total_hours() > 80:
            print(f"  ⛔ This is a VERY LARGE feature ({self.total_hours():.0f}h = ~{self.total_hours()/40:.1f} weeks)")
            print(f"     Consider breaking into smaller incremental features")
        elif self.total_hours() > 40:
            print(f"  ⚠️  This is a LARGE feature ({self.total_hours():.0f}h = ~{self.total_hours()/40:.1f} weeks)")
            print(f"     Prioritize carefully, ensure high value")
        elif avg_complexity >= 4:
            print(f"  ⚠️  This feature has HIGH complexity (avg: {avg_complexity:.1f}/5)")
            print(f"     Ensure expert review and extensive testing")
        else:
            print(f"  ✓ Reasonable feature size ({self.total_hours():.0f}h, complexity: {avg_complexity:.1f}/5)")
        
        print(f"{'='*70}\n")

# Pre-defined feature estimates
PREDEFINED_FEATURES = {
    "error_propagation_operator": FeatureEstimate(
        "Error Propagation Operator (?)",
        "Rust-style ? operator to unwrap Result<T,E> or propagate errors"
    ),
    "effect_system": FeatureEstimate(
        "Effect System (IO tracking)",
        "Track function purity and side effects (IO monad style)"
    ),
    "totality_checker": FeatureEstimate(
        "Totality Checker",
        "Verify all functions are total (no runtime crashes possible)"
    ),
    "unicode_strings": FeatureEstimate(
        "Unicode-aware String Operations",
        "Grapheme-aware string functions (length, index, slice)"
    ),
}

# Configure pre-defined features
def configure_error_propagation():
    feat = PREDEFINED_FEATURES["error_propagation_operator"]
    feat.add_component("lexer", "token")  # ? token
    feat.add_component("parser", "expression")  # Parse (expr)?
    feat.add_component("typechecker", "simple_check")  # Verify Result type
    feat.add_component("transpiler", "expression_codegen")  # Desugar to match
    feat.add_component("testing", "unit_test")  # Multiple tests
    feat.add_component("testing", "shadow_tests")
    feat.add_component("docs", "spec_update")
    feat.add_component("docs", "memory_md_update")
    feat.add_component("docs", "examples")
    return feat

def configure_effect_system():
    feat = PREDEFINED_FEATURES["effect_system"]
    feat.add_component("parser", "declaration")  # @pure/@impure annotations
    feat.add_component("typechecker", "type_inference")  # Infer effects
    feat.add_component("typechecker", "constraint_solving")  # Effect constraints
    feat.add_component("transpiler", "complex_transform")  # Generate effect-aware code
    feat.add_component("testing", "integration_test")
    feat.add_component("docs", "spec_update")
    feat.breaking_change = True
    feat.affects_bootstrap = True
    return feat

def configure_totality_checker():
    feat = PREDEFINED_FEATURES["totality_checker"]
    feat.add_component("typechecker", "constraint_solving")  # Prove totality
    feat.add_component("typechecker", "type_inference")  # Path analysis
    feat.add_component("testing", "integration_test")
    feat.add_component("docs", "spec_update")
    return feat

def configure_unicode_strings():
    feat = PREDEFINED_FEATURES["unicode_strings"]
    feat.add_component("runtime", "ffi_binding")  # ICU/utf8proc bindings
    feat.add_component("stdlib", "module")  # unicode module
    feat.add_component("stdlib", "simple_function")  # grapheme_length
    feat.add_component("stdlib", "simple_function")  # grapheme_at
    feat.add_component("stdlib", "complex_function")  # normalize
    feat.add_component("testing", "unit_test")
    feat.add_component("testing", "shadow_tests")
    feat.add_component("docs", "examples")
    feat.breaking_change = True  # str_length semantics change
    return feat

def main():
    if len(sys.argv) < 2:
        print("Usage: estimate_feature_cost.py <feature_name>")
        print("\nPredefined features:")
        for name, feat in PREDEFINED_FEATURES.items():
            print(f"  {name:<30} {feat.description}")
        print("\nOr provide component list as JSON:")
        print('  estimate_feature_cost.py --json \'{"lexer": ["keyword"], "parser": ["statement"]}\'')
        sys.exit(1)
    
    feature_name = sys.argv[1]
    
    if feature_name == "error_propagation_operator":
        feat = configure_error_propagation()
    elif feature_name == "effect_system":
        feat = configure_effect_system()
    elif feature_name == "totality_checker":
        feat = configure_totality_checker()
    elif feature_name == "unicode_strings":
        feat = configure_unicode_strings()
    elif feature_name == "--json" and len(sys.argv) > 2:
        # Custom feature from JSON
        components = json.loads(sys.argv[2])
        feat = FeatureEstimate("Custom Feature")
        for category, component_list in components.items():
            for component in component_list:
                feat.add_component(category, component)
    else:
        print(f"Unknown feature: {feature_name}")
        print("Use --help to see available features")
        sys.exit(1)
    
    feat.print_estimate()

if __name__ == "__main__":
    main()

