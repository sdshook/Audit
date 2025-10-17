#!/usr/bin/env python3
"""
SRCA Built-in Visualization Demo
================================

Demonstrates the built-in matplotlib visualization capabilities in SRCA.py
Shows comprehensive performance metrics including:
- Learning progress (rewards & confidence)
- Self-regulation (coherence & arrogance)
- Action selection patterns
- Alert distribution analysis
- Performance statistics
"""

from SRCA import run_simulation

def demo_srca_visualization():
    """Run SRCA simulation and generate built-in visualizations."""
    
    print("=" * 60)
    print("SRCA BUILT-IN VISUALIZATION DEMO")
    print("=" * 60)
    print("Running SRCA simulation with built-in matplotlib visualizations...")
    print()
    
    # Run simulation with built-in visualization
    # This will automatically generate comprehensive plots
    episodes = run_simulation(n_episodes=100, verbose=True)
    
    print("\n" + "=" * 60)
    print("SRCA VISUALIZATION COMPLETE")
    print("=" * 60)
    print("The built-in SRCA visualization includes:")
    print("✅ Learning Progress: Rewards & Confidence over time")
    print("✅ Self-Regulation: Coherence & Arrogance monitoring")
    print("✅ Action Selection: Heatmap of action choices")
    print("✅ Action Distribution: Total action counts")
    print("✅ Alert Analysis: Distribution of threat types")
    print("✅ Performance Summary: Comprehensive statistics")
    print()
    print("All visualizations use matplotlib from SRCA.py")
    
    return episodes

if __name__ == "__main__":
    episodes = demo_srca_visualization()
    print(f"\nCompleted simulation with {len(episodes)} episodes!")
