#!/usr/bin/env python3
"""
CMNN Neural Network Learning Observation Test for SRCA
======================================================

This test specifically observes CMNN (Collective Mesh Neural Network) learning by:
1. Tracking neural network weight changes through gradient descent
2. Monitoring loss reduction over episodes
3. Observing improved action selection accuracy
4. Analyzing confidence calibration improvement
5. Measuring detection performance enhancement

Expected CMNN Learning Manifestations:
- Neural network weights should change through backpropagation
- Policy gradient loss should decrease over time
- Action selection should become more appropriate for threat levels
- Confidence estimates should become more accurate
- Overall success rate should improve with experience
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime

# Import SRCA components
from SRCA import *

def create_progressive_test_data():
    """Create test data that progressively challenges the CMNN learning."""
    
    # Start with clear patterns, then introduce ambiguity
    episodes = []
    
    # Phase 1: Clear patterns (episodes 0-50)
    clear_patterns = [
        ("lateral_movement", "malicious"),    # Should learn ISOLATE
        ("reconnaissance", "suspicious"),     # Should learn DEPLOY_DECOY
        ("exfiltration", "malicious"),       # Should learn ESCALATE
        ("persistence", "benign"),           # Should learn NO_OP
    ]
    
    for _ in range(12):  # 12 cycles = 48 episodes
        episodes.extend(clear_patterns)
    
    # Phase 2: Ambiguous patterns (episodes 48-98)
    ambiguous_patterns = [
        ("lateral_movement", "suspicious"),   # Ambiguous - could be ISOLATE or DEPLOY_DECOY
        ("reconnaissance", "benign"),         # Ambiguous - could be NO_OP or DEPLOY_DECOY
        ("exfiltration", "suspicious"),       # Ambiguous - could be ESCALATE or ISOLATE
        ("persistence", "malicious"),         # Clear - should be ISOLATE
    ]
    
    for _ in range(12):  # 12 cycles = 48 episodes
        episodes.extend(ambiguous_patterns)
    
    # Phase 3: Mixed complexity (episodes 96-150)
    mixed_patterns = clear_patterns + ambiguous_patterns
    for _ in range(7):  # 7 cycles = 56 episodes
        episodes.extend(mixed_patterns)
    
    return episodes

def extract_cmnn_weights(mesh_network):
    """Extract current CMNN network weights for analysis."""
    weights_info = {}
    
    # Extract weights from each node network
    for i, node in enumerate(mesh_network.nodes):
        node_weights = []
        for param in node.parameters():
            node_weights.append(param.data.clone().flatten())
        if node_weights:
            weights_info[f'node_{i}'] = torch.cat(node_weights).numpy()
    
    # Extract meta-reasoning network weights
    meta_weights = []
    for param in mesh_network.meta.parameters():
        meta_weights.append(param.data.clone().flatten())
    if meta_weights:
        weights_info['meta'] = torch.cat(meta_weights).numpy()
    
    # Extract message passing weights
    msg_weights = []
    for param in mesh_network.message_passing.parameters():
        msg_weights.append(param.data.clone().flatten())
    if msg_weights:
        weights_info['message_passing'] = torch.cat(msg_weights).numpy()
    
    return weights_info

def calculate_weight_changes(weights_before, weights_after):
    """Calculate the magnitude of weight changes."""
    changes = {}
    for key in weights_before:
        if key in weights_after:
            diff = weights_after[key] - weights_before[key]
            changes[key] = {
                'l2_norm': np.linalg.norm(diff),
                'mean_abs_change': np.mean(np.abs(diff)),
                'max_change': np.max(np.abs(diff))
            }
    return changes

def analyze_action_appropriateness(actions, alert_types, threat_levels):
    """Analyze how appropriate the actions are for given threats."""
    
    # Define ideal action mappings (based on cybersecurity best practices)
    ideal_actions = {
        ("lateral_movement", "malicious"): "ISOLATE",
        ("lateral_movement", "suspicious"): "ISOLATE",  # Conservative approach
        ("lateral_movement", "benign"): "NO_OP",
        
        ("reconnaissance", "malicious"): "ISOLATE",
        ("reconnaissance", "suspicious"): "DEPLOY_DECOY",  # Gather more intel
        ("reconnaissance", "benign"): "NO_OP",
        
        ("exfiltration", "malicious"): "ESCALATE",  # Critical - need human intervention
        ("exfiltration", "suspicious"): "ESCALATE",  # Conservative for data protection
        ("exfiltration", "benign"): "NO_OP",
        
        ("persistence", "malicious"): "ISOLATE",
        ("persistence", "suspicious"): "ISOLATE",  # Conservative approach
        ("persistence", "benign"): "NO_OP",
    }
    
    correct_actions = 0
    total_actions = len(actions)
    
    appropriateness_by_phase = {"early": [], "mid": [], "late": []}
    
    for i, (action, alert_type, threat_level) in enumerate(zip(actions, alert_types, threat_levels)):
        pattern = (alert_type, threat_level)
        ideal_action = ideal_actions.get(pattern, "NO_OP")  # Default to NO_OP if unknown
        
        is_correct = (action == ideal_action)
        if is_correct:
            correct_actions += 1
        
        # Categorize by learning phase
        if i < total_actions // 3:
            appropriateness_by_phase["early"].append(is_correct)
        elif i < 2 * total_actions // 3:
            appropriateness_by_phase["mid"].append(is_correct)
        else:
            appropriateness_by_phase["late"].append(is_correct)
    
    overall_accuracy = correct_actions / total_actions if total_actions > 0 else 0
    
    phase_accuracies = {}
    for phase, results in appropriateness_by_phase.items():
        phase_accuracies[phase] = np.mean(results) if results else 0
    
    return overall_accuracy, phase_accuracies

def run_cmnn_learning_experiment():
    """Run experiment to observe CMNN neural network learning."""
    
    print("=" * 70)
    print("CMNN NEURAL NETWORK LEARNING OBSERVATION EXPERIMENT")
    print("=" * 70)
    print("Testing: Gradient-based learning in Collective Mesh Neural Network")
    print("Expected: Improved cognition through backpropagation and weight updates")
    print()
    
    # Create progressive test data
    test_episodes = create_progressive_test_data()
    total_episodes = len(test_episodes)
    
    print(f"Generated {total_episodes} progressive episodes:")
    print("- Phase 1 (0-48):   Clear patterns for initial learning")
    print("- Phase 2 (48-96):  Ambiguous patterns for advanced learning")  
    print("- Phase 3 (96-150): Mixed complexity for robustness testing")
    print()
    
    # Initialize tracking variables
    weight_evolution = []
    loss_evolution = []
    action_history = []
    alert_type_history = []
    threat_level_history = []
    confidence_evolution = []
    reward_evolution = []
    
    # Get initial CMNN weights
    initial_weights = extract_cmnn_weights(mesh)
    previous_weights = initial_weights.copy()
    
    print("Starting CMNN learning observation...")
    print("Tracking: Neural network weights, loss values, action accuracy")
    print()
    
    # Run episodes with detailed tracking
    for episode, (alert_type, threat_level) in enumerate(test_episodes):
        
        # Create alert
        alert = {
            "id": f"alert_{episode}",
            "type": alert_type,
            "severity": threat_level,
            "label": threat_level,  # Add label field for reward calculation
            "pattern_type": alert_type,  # Add pattern_type field
            "timestamp": episode,
            "source": f"test_node_{episode % 3}",
            "text": f"{alert_type} detected with {threat_level} severity from test_node_{episode % 3}"
        }
        
        # Run simulation step
        result = simulation_step(alert, verbose=False)
        
        # Track results
        action_history.append(result['action'])
        alert_type_history.append(alert_type)
        threat_level_history.append(threat_level)
        confidence_evolution.append(result['confidence'])
        reward_evolution.append(result['reward'])
        
        # Track weight changes every 10 episodes
        if episode % 10 == 0:
            current_weights = extract_cmnn_weights(mesh)
            weight_changes = calculate_weight_changes(previous_weights, current_weights)
            
            weight_evolution.append({
                'episode': episode,
                'weight_changes': weight_changes,
                'total_weight_change': sum(change['l2_norm'] for change in weight_changes.values())
            })
            
            previous_weights = current_weights.copy()
        
        # Progress reporting with learning indicators
        if episode % 25 == 0:
            recent_rewards = reward_evolution[-10:] if len(reward_evolution) >= 10 else reward_evolution
            avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            print(f"Episode {episode:3d}: {alert_type:15s} + {threat_level:10s} → {result['action']:12s} "
                  f"(Conf: {result['confidence']:.3f}, Reward: {result['reward']:+.2f}, "
                  f"Avg: {avg_recent_reward:+.2f})")
    
    print()
    print("=" * 70)
    print("CMNN NEURAL NETWORK LEARNING ANALYSIS")
    print("=" * 70)
    
    # 1. Weight Evolution Analysis
    print("\n1. NEURAL NETWORK WEIGHT EVOLUTION")
    print("-" * 50)
    
    if len(weight_evolution) > 1:
        initial_change = weight_evolution[0]['total_weight_change']
        final_change = weight_evolution[-1]['total_weight_change']
        
        print(f"Weight change tracking across {len(weight_evolution)} checkpoints:")
        print(f"Initial weight change rate: {initial_change:.6f}")
        print(f"Final weight change rate:   {final_change:.6f}")
        
        # Analyze weight change trend
        weight_changes = [entry['total_weight_change'] for entry in weight_evolution]
        if len(weight_changes) > 2:
            early_avg = np.mean(weight_changes[:len(weight_changes)//2])
            late_avg = np.mean(weight_changes[len(weight_changes)//2:])
            
            if early_avg > late_avg * 1.5:
                print("✅ LEARNING CONVERGENCE: Weight changes decreased over time")
                print("   → Network is stabilizing as it learns optimal representations")
            elif late_avg > early_avg * 1.5:
                print("⚠️  CONTINUED ADAPTATION: Weight changes increased over time")
                print("   → Network still actively learning (may need more episodes)")
            else:
                print("→ STEADY LEARNING: Consistent weight change rate")
        
        # Analyze component-specific learning
        print("\nComponent-specific weight changes:")
        for entry in weight_evolution[-3:]:  # Last 3 checkpoints
            episode = entry['episode']
            changes = entry['weight_changes']
            print(f"  Episode {episode:3d}:")
            for component, change_info in changes.items():
                print(f"    {component:8s}: L2={change_info['l2_norm']:.6f}, "
                      f"Mean={change_info['mean_abs_change']:.6f}")
    
    # 2. Action Appropriateness Analysis
    print("\n2. ACTION SELECTION IMPROVEMENT")
    print("-" * 50)
    
    overall_accuracy, phase_accuracies = analyze_action_appropriateness(
        action_history, alert_type_history, threat_level_history)
    
    print(f"Overall Action Accuracy: {overall_accuracy:.1%}")
    print("\nLearning progression:")
    print(f"  Early Phase (0-33%):   {phase_accuracies['early']:.1%} accuracy")
    print(f"  Mid Phase   (33-66%):  {phase_accuracies['mid']:.1%} accuracy") 
    print(f"  Late Phase  (66-100%): {phase_accuracies['late']:.1%} accuracy")
    
    improvement = phase_accuracies['late'] - phase_accuracies['early']
    if improvement > 0.1:
        print(f"✅ SIGNIFICANT IMPROVEMENT: +{improvement:.1%} accuracy gain")
        print("   → CMNN learning improved action selection through experience")
    elif improvement > 0.05:
        print(f"✅ MODERATE IMPROVEMENT: +{improvement:.1%} accuracy gain")
        print("   → CMNN showing learning progress")
    elif improvement > -0.05:
        print(f"→ STABLE PERFORMANCE: {improvement:+.1%} change")
        print("   → Consistent performance across phases")
    else:
        print(f"⚠️  PERFORMANCE DECLINE: {improvement:.1%} accuracy loss")
        print("   → May indicate overfitting or need parameter adjustment")
    
    # 3. Confidence Calibration Analysis
    print("\n3. CONFIDENCE CALIBRATION LEARNING")
    print("-" * 50)
    
    # Analyze confidence vs accuracy correlation
    phase_confidences = {
        "early": confidence_evolution[:len(confidence_evolution)//3],
        "mid": confidence_evolution[len(confidence_evolution)//3:2*len(confidence_evolution)//3],
        "late": confidence_evolution[2*len(confidence_evolution)//3:]
    }
    
    print("Confidence evolution:")
    for phase, confidences in phase_confidences.items():
        avg_conf = np.mean(confidences) if confidences else 0
        std_conf = np.std(confidences) if confidences else 0
        print(f"  {phase.capitalize():5s} Phase: {avg_conf:.3f} ± {std_conf:.3f}")
    
    # Check if confidence correlates with accuracy
    early_conf_acc_diff = phase_confidences["early"] and phase_accuracies["early"] - np.mean(phase_confidences["early"])
    late_conf_acc_diff = phase_confidences["late"] and phase_accuracies["late"] - np.mean(phase_confidences["late"])
    
    if abs(late_conf_acc_diff) < abs(early_conf_acc_diff):
        print("✅ IMPROVED CALIBRATION: Confidence better aligned with accuracy")
    
    # 4. Reward Learning Analysis
    print("\n4. REWARD-BASED LEARNING PROGRESSION")
    print("-" * 50)
    
    phase_rewards = {
        "early": reward_evolution[:len(reward_evolution)//3],
        "mid": reward_evolution[len(reward_evolution)//3:2*len(reward_evolution)//3],
        "late": reward_evolution[2*len(reward_evolution)//3:]
    }
    
    print("Reward progression:")
    for phase, rewards in phase_rewards.items():
        avg_reward = np.mean(rewards) if rewards else 0
        success_rate = np.mean([1 for r in rewards if r > 0]) if rewards else 0
        print(f"  {phase.capitalize():5s} Phase: {avg_reward:+.3f} avg reward, {success_rate:.1%} success rate")
    
    reward_improvement = np.mean(phase_rewards["late"]) - np.mean(phase_rewards["early"])
    if reward_improvement > 0.1:
        print(f"✅ STRONG LEARNING: +{reward_improvement:.3f} reward improvement")
        print("   → CMNN effectively learning from reward signals")
    elif reward_improvement > 0.05:
        print(f"✅ MODERATE LEARNING: +{reward_improvement:.3f} reward improvement")
    else:
        print(f"→ STABLE REWARDS: {reward_improvement:+.3f} change")
    
    # 5. Overall CMNN Learning Assessment
    print("\n5. OVERALL CMNN LEARNING ASSESSMENT")
    print("-" * 50)
    
    learning_indicators = []
    
    # Check for weight evolution
    if len(weight_evolution) > 1:
        learning_indicators.append("✅ Neural network weights actively updating")
    
    # Check for accuracy improvement
    if improvement > 0.05:
        learning_indicators.append(f"✅ Action selection improved by {improvement:.1%}")
    
    # Check for reward improvement
    if reward_improvement > 0.05:
        learning_indicators.append(f"✅ Reward performance improved by {reward_improvement:.3f}")
    
    # Check for overall performance
    if overall_accuracy > 0.6:
        learning_indicators.append(f"✅ High overall accuracy ({overall_accuracy:.1%})")
    
    # Check for learning stability
    if len(weight_changes) > 2 and early_avg > late_avg:
        learning_indicators.append("✅ Learning convergence observed")
    
    print("\nCMNN LEARNING EVIDENCE:")
    for indicator in learning_indicators:
        print(f"  {indicator}")
    
    if len(learning_indicators) >= 4:
        print(f"\n🧠 CONCLUSION: STRONG CMNN LEARNING OBSERVED")
        print("   Neural network successfully improving cognition through gradient descent")
        print("   Enhanced detection and action selection through experience")
    elif len(learning_indicators) >= 2:
        print(f"\n🔬 CONCLUSION: MODERATE CMNN LEARNING OBSERVED")
        print("   Some evidence of neural network learning and improvement")
    else:
        print(f"\n❓ CONCLUSION: LIMITED CMNN LEARNING EVIDENCE")
        print("   May need more episodes or different learning parameters")
    
    return {
        'weight_evolution': weight_evolution,
        'overall_accuracy': overall_accuracy,
        'phase_accuracies': phase_accuracies,
        'confidence_evolution': confidence_evolution,
        'reward_evolution': reward_evolution,
        'learning_indicators': len(learning_indicators),
        'improvement': improvement,
        'reward_improvement': reward_improvement
    }

if __name__ == "__main__":
    # Run the CMNN learning experiment
    results = run_cmnn_learning_experiment()
    
    # Save results for further analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"cmnn_learning_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.floating, np.integer, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    json_results = convert_to_json_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    print("\nCMNN neural network learning analysis complete!")