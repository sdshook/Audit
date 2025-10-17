#!/usr/bin/env python3
"""
SRCA Visualization Demo with File Output
========================================

Demonstrates SRCA's built-in matplotlib visualization capabilities
and saves the comprehensive performance plots to files.
"""

from SRCA import run_simulation, plot_results
from datetime import datetime

def demo_srca_with_saved_plots():
    """Run SRCA simulation and save visualizations to files."""
    
    print("=" * 70)
    print("SRCA VISUALIZATION DEMO - WITH FILE OUTPUT")
    print("=" * 70)
    print("Running SRCA simulation and saving matplotlib visualizations...")
    print()
    
    # Run simulation (this will show plots but not save them)
    episodes = run_simulation(n_episodes=100, verbose=False)
    
    print("\n" + "=" * 70)
    print("GENERATING SAVED VISUALIZATIONS")
    print("=" * 70)
    
    # Generate saved visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comprehensive SRCA visualization
    srca_filename = f"srca_comprehensive_{timestamp}.png"
    plot_results(episodes, save_path=srca_filename)
    
    print(f"✅ SRCA Comprehensive Visualization: {srca_filename}")
    
    # Analysis summary
    print("\n" + "=" * 70)
    print("SRCA BUILT-IN VISUALIZATION FEATURES")
    print("=" * 70)
    
    # Calculate some statistics
    rewards = [ep['reward'] for ep in episodes]
    confidences = [ep['confidence'] for ep in episodes]
    actions = [ep['action'] for ep in episodes]
    
    print(f"\n📊 VISUALIZATION COMPONENTS:")
    print(f"   1. Learning Progress: Rewards & Confidence evolution")
    print(f"   2. Self-Regulation: Coherence & Arrogance monitoring")
    print(f"   3. Action Selection: Heatmap showing decision patterns")
    print(f"   4. Action Distribution: Bar chart of total action counts")
    print(f"   5. Alert Analysis: Pie chart of threat type distribution")
    print(f"   6. Performance Summary: Comprehensive statistics panel")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Average Reward: {sum(rewards)/len(rewards):.3f}")
    print(f"   Success Rate: {sum(1 for r in rewards if r > 0)/len(rewards):.1%}")
    print(f"   Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}")
    
    print(f"\n🎨 MATPLOTLIB FEATURES USED:")
    print(f"   ✅ Subplots (3x2 grid layout)")
    print(f"   ✅ Line plots with moving averages")
    print(f"   ✅ Heatmaps with color mapping")
    print(f"   ✅ Bar charts with custom colors")
    print(f"   ✅ Pie charts with percentage labels")
    print(f"   ✅ Text annotations and statistics")
    print(f"   ✅ Grid lines and legends")
    print(f"   ✅ Color bars and custom styling")
    
    print(f"\n🔬 LEARNING EVIDENCE:")
    early_rewards = rewards[:len(rewards)//3]
    late_rewards = rewards[2*len(rewards)//3:]
    improvement = sum(late_rewards)/len(late_rewards) - sum(early_rewards)/len(early_rewards)
    
    early_conf = sum(confidences[:len(confidences)//3])/len(confidences[:len(confidences)//3])
    late_conf = sum(confidences[2*len(confidences)//3:])/len(confidences[2*len(confidences)//3:])
    conf_change = (late_conf - early_conf) / early_conf * 100
    
    print(f"   Performance Improvement: {improvement:+.3f} ({improvement/abs(sum(early_rewards)/len(early_rewards))*100:+.1f}%)")
    print(f"   Confidence Calibration: {conf_change:+.1f}% change")
    print(f"   Learning Status: {'ACTIVE' if abs(improvement) > 0.1 else 'STABLE'}")
    
    return episodes, srca_filename

if __name__ == "__main__":
    episodes, filename = demo_srca_with_saved_plots()
    print(f"\n🎯 DEMO COMPLETE!")
    print(f"Generated comprehensive SRCA visualization: {filename}")
    print(f"Processed {len(episodes)} episodes with full matplotlib integration.")
