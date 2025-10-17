# Self-Regulated Cognitive Architecture (SRCA)

**A Novel AI System for Autonomous Cybersecurity Decision-Making**

*© 2025 Shane D. Shook, All Rights Reserved*

## Overview

SRCA (Self-Regulated Cognitive Architecture) is a sophisticated proof-of-concept implementation that demonstrates advanced cognitive architectures for autonomous cybersecurity operations. The system combines multiple AI techniques including biologically-inspired memory systems, neural networks, and reinforcement learning to create a self-aware, self-regulating AI capable of making complex security decisions.

## Key Features

### 🧠 **Novel Memory Architecture**
- **Bidirectional Hebbian Memory (BDH)**: Dual-store system (reflective/empathic) with reward-gated synaptic plasticity
- **Persistent Semantic Index (PSI)**: Long-term semantic memory with protected guardrail entries
- **Episode Memory**: Complete decision episode storage for experiential learning

### 🔗 **Cognitive Mesh Neural Network (CMNN)**
- Distributed reasoning across multiple interconnected nodes
- Message passing for collective intelligence
- Individual nodes with policy, confidence, and value heads
- Meta-reasoning over distributed outputs

### 🛡️ **Self-Awareness & Safety**
- Real-time monitoring of coherence, confidence, and arrogance levels
- Built-in guardrails prevent dangerous actions without sufficient confidence
- Protected knowledge entries that resist modification
- Valence controller for empathic and reflective balancing

### 📊 **Comprehensive Visualization**
- Six-panel performance dashboard
- Real-time learning progress tracking
- Action distribution analysis
- Memory system statistics

## Technical Architecture

### Memory Systems

#### Bidirectional Hebbian Memory (BDH) - Novel Dual-Store Architecture
**Conceptual Advance**: Extension of Dragon Hatchling's single-network approach to dual-processing cognitive architecture.

SRCA implements a **novel dual-store BDH system** that advances beyond the original Dragon Hatchling architecture:

**Mathematical Formulation:**
```
For each memory trace i with state vector x_i and reward signal r:

Dual-Store Update:
W_reflective[i] = W_reflective[i] + η_pot * r * (x_i ⊗ x_i + E_pos[i] ⊗ E_pos[i])  [if r > 0]
W_empathic[i] = W_empathic[i] + η_pot * r * (x_i ⊗ x_i + E_pos[i] ⊗ E_pos[i])     [if r > 0]

W_reflective[i] = W_reflective[i] - η_dep * |r| * (x_i ⊗ x_i + E_neg[i] ⊗ E_neg[i]) [if r < 0, not protected]
W_empathic[i] = W_empathic[i] - η_dep * |r| * (x_i ⊗ x_i + E_neg[i] ⊗ E_neg[i])    [if r < 0, not protected]

Eligibility Trace Evolution:
E_pos[i](t+1) = γ_E * E_pos[i](t) + max(0, x_i ⊗ y_t).mean(axis=1)
E_neg[i](t+1) = γ_E * E_neg[i](t) + max(0, -(x_i ⊗ y_t)).mean(axis=1)

Memory Consolidation Criterion:
if |∑_t r_t| > τ_consolidation: transfer_to_PSI(trace_i)
```

**Conceptual Innovations:**
- **Dual-Processing Architecture**: Reflective (System 2) and Empathic (System 1) stores mirror human cognitive dual-process theory
- **Protected Memory Mechanism**: Ethical guardrails resist modification even under negative rewards
- **Eligibility Trace Enhancement**: Bidirectional traces for both potentiation and depression pathways
- **Dynamic Consolidation**: Automatic transfer to long-term semantic memory based on cumulative significance

**AGI Advancement**: This dual-store architecture enables the system to maintain both analytical reasoning and emotional/contextual processing simultaneously, a critical requirement for general intelligence.

#### Persistent Semantic Index (PSI) - Enhanced Context Management
**Conceptual Advance**: Extension of Anthropic's context management to include valence-weighted semantic associations and protected ethical memory.

SRCA's PSI advances beyond Anthropic's file-based memory tool with sophisticated semantic processing:

**Mathematical Formulation:**
```
Semantic Similarity with Valence Weighting:
score(query, doc_i) = cos_sim(query, doc_i.vector) * (1 + α * doc_i.valence)

where cos_sim(a, b) = (a · b) / (||a|| * ||b||)

Access-Based Strengthening:
access_weight(doc_i) = log(1 + access_count[doc_i])

Protected Memory Constraint:
if doc_i.protected and doc_i.valence < -0.5:
    doc_i.vector = immutable  // Ethical guardrails resist modification

Valence Evolution:
doc_i.valence(t+1) = λ * doc_i.valence(t) + (1-λ) * reward_signal

Multi-Criteria Retrieval:
retrieval_score = β₁ * semantic_sim + β₂ * valence_boost + β₃ * access_frequency
```

**Conceptual Innovations:**
- **Valence-Weighted Retrieval**: Memories with positive associations are preferentially recalled
- **Protected Ethical Memory**: Guardrail entries resist modification despite learning pressure
- **Dynamic Access Patterns**: Frequently accessed memories become more accessible (use-dependent plasticity)
- **Multi-Modal Indexing**: Tags, valence, and semantic vectors create rich associative structure

**AGI Advancement**: This creates a semantic memory system that maintains ethical constraints while adapting to experience, essential for safe AGI development.

#### Memory Consolidation
Mimics hippocampal-cortical transfer:
- Short-term episodic traces in BDH (hippocampus-like)
- Long-term semantic consolidation in PSI (cortex-like)
- Threshold-based transfer based on cumulative reward

### Cognitive Processing

#### Cognitive Mesh Neural Network (CMNN) - Novel Distributed Reasoning
**Conceptual Advance**: Original architecture combining message passing with meta-reasoning for collective intelligence.

SRCA's CMNN creates a novel distributed reasoning system not present in either source:

**Mathematical Formulation:**
```
Individual Node Processing:
h_i = ReLU(Linear([state_vec; context_vec]))
logits_i = W_policy * h_i
conf_i = σ(W_conf * h_i)
value_i = W_value * h_i

Message Passing Between Nodes:
H = [h_1, h_2, ..., h_N]  // Concatenated node states
M = W_message * flatten(H)  // Inter-node communication
H_updated = reshape(M, [N, hidden_dim])

Meta-Reasoning Over Collective:
meta_input = [flatten(logits), conf_vector, value_vector]
final_logits = W_meta * meta_input
action_probs = softmax(final_logits)

Collective Decision:
action ~ Categorical(action_probs)
```

**Conceptual Innovations:**
- **Distributed Reasoning**: Multiple nodes process information in parallel like cortical columns
- **Message Passing**: Inter-node communication enables collective intelligence
- **Meta-Reasoning**: Higher-order reasoning over individual node outputs
- **Confidence Aggregation**: System-level confidence emerges from node consensus

#### Self-Awareness Module - Novel Metacognitive Architecture
**Conceptual Advance**: First implementation of real-time self-awareness monitoring in cognitive architectures.

**Mathematical Formulation:**
```
Self-Monitoring State:
cognitive_state = [flatten(node_states), confidence_vector, value_vector]

Self-Awareness Metrics:
coherence = σ(W_coh * cognitive_state)
confidence = σ(W_conf * cognitive_state)  
arrogance = σ(W_arr * cognitive_state)

Metacognitive Loss:
L_meta = MSE(coherence, reward_signal) + 
         MSE(confidence, actual_performance) + 
         MSE(arrogance, overconfidence_penalty)

Self-Regulation Signal:
if confidence > 0.8 and actual_outcome < 0.5:
    arrogance_penalty += 0.1
else:
    arrogance_penalty = max(0, arrogance_penalty - 0.05)
```

#### Valence Controller - Novel Self-Regulation System
**Conceptual Advance**: First implementation of empathic self-regulation with arrogance detection.

**Mathematical Formulation:**
```
Empathy Factor Evolution:
empathy(t+1) = 0.7 * empathy(t) + 0.3 * human_feedback

Arrogance Penalty Dynamics:
prediction_error = |confidence - actual_outcome|
if confidence > 0.8 and actual_outcome < 0.5:
    arrogance_penalty = min(0.5, arrogance_penalty + 0.1)
else:
    arrogance_penalty = max(0, arrogance_penalty - 0.05)

Valence Regulation:
regulated_reward = base_reward * (1 + empathy - arrogance_penalty)

Guardrail Activation:
if regulated_reward < -1.0:
    trigger_safety_override()
    regulated_reward = max(-1.0, regulated_reward)
```

**AGI Advancement**: These systems enable real-time self-monitoring and behavioral regulation, critical capabilities for safe and beneficial AGI.

## Cybersecurity Application

### Alert Processing
Handles various security alert patterns:
- **Lateral Movement**: PsExec, WMI, RDP, SMB activities
- **Exfiltration**: DNS tunneling, large HTTPS transfers, cloud uploads
- **Persistence**: Registry modifications, scheduled tasks, service creation
- **Reconnaissance**: Port scans, LDAP queries, network enumeration

### Action Selection
Four primary security actions:
- **NO_OP**: No action required
- **ESCALATE**: Alert human operators
- **ISOLATE**: Quarantine affected systems
- **DEPLOY_DECOY**: Deploy deception technology

### Context Awareness
- Maintenance window detection
- Critical asset identification
- Multi-objective reward calculation
- Collateral damage minimization

## Installation & Requirements

### Dependencies
```bash
pip install torch numpy matplotlib
```

### System Requirements
- Python 3.6+
- PyTorch (CPU or GPU)
- NumPy
- Matplotlib
- Standard library modules: time, random, collections, typing, json, hashlib

### Hardware Recommendations
- **CPU**: Multi-core processor for parallel processing
- **Memory**: 4GB+ RAM for large-scale simulations
- **GPU**: Optional, CUDA-compatible for accelerated training

## Usage

### Basic Simulation
```python
python SRCA.py
```

### Custom Configuration
```python
# Modify configuration constants
EMBED_DIM = 64        # Embedding dimensions
N_NODES = 5           # Number of mesh nodes
ACTION_DIM = 4        # Number of possible actions
LR = 1e-3            # Learning rate

# Run simulation
episodes = run_simulation(n_episodes=200, verbose=True)
```

### Advanced Usage
```python
# Access individual components
from SRCA import *

# Initialize custom PSI with domain knowledge
psi.add_doc("custom_rule", "Custom security policy", 
           embedder.embed("policy text"), 
           tags=["policy"], valence=1.0, protected=True)

# Run single decision cycle
alert = alert_gen.generate()
episode = simulation_step(alert, verbose=True)
```

## Configuration Parameters

### Core Architecture
- `EMBED_DIM`: Embedding vector dimensions (default: 32)
- `N_NODES`: Number of CMNN nodes (default: 3)
- `ACTION_DIM`: Number of possible actions (default: 4)

### Learning Parameters
- `LR`: Neural network learning rate (default: 1e-3)
- `BDH_ETA_POT`: Hebbian potentiation rate (default: 1e-3)
- `BDH_ETA_DEP`: Hebbian depression rate (default: 5e-4)
- `GAMMA_E`: Eligibility trace decay (default: 0.9)
- `TAU_CONSOLIDATION`: Memory consolidation threshold (default: 0.7)

## Output & Visualization

The system generates comprehensive visualizations including:

1. **Learning Progress**: Rewards and confidence over time
2. **Self-Regulation**: Coherence and arrogance monitoring
3. **Action Distribution**: Heatmap of action selection patterns
4. **Performance Statistics**: Success rates and memory utilization
5. **Alert Analysis**: Distribution of processed alert types
6. **System Status**: Memory statistics and guardrail activations

## Research Applications

### Cognitive AI Research
- Study of multi-memory system integration
- Biological plausibility in artificial systems
- Self-awareness and introspection in AI

### AI Safety Research
- Guardrail effectiveness in autonomous systems
- Overconfidence detection and mitigation
- Human-AI collaboration mechanisms

### Cybersecurity AI
- Autonomous security operations
- Context-aware threat response
- Explainable security decision-making

## Biological Inspiration

The architecture closely mirrors biological memory systems:

- **Hebbian Learning**: "Cells that fire together, wire together"
- **Synaptic Tagging**: Eligibility traces for temporal credit assignment
- **Memory Consolidation**: Hippocampal-cortical transfer mechanisms
- **Emotional Modulation**: Valence affects memory strength and recall
- **Cortical Columns**: CMNN nodes simulate cortical processing units

## Innovation & Contributions

### Technical Advances
- **Symbolic-Connectionist Integration**: PSI semantic memory + neural network learning
- **Biological Plausibility**: Hebbian mechanisms with modern RL techniques
- **Safety-by-Design**: Embedded ethical constraints and self-regulation
- **Emergent Properties**: Self-improving confidence calibration and wisdom accumulation

## Research Inspirations & References

### Core Architecture Inspirations

#### Dragon Hatchling (BDH) - Pathway.com Research
The BDH (Bidirectional Hebbian Memory) component draws inspiration from Pathway's groundbreaking research:

**"The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"**
- **Paper**: https://arxiv.org/pdf/2509.26507
- **Technical Blog**: https://pathway.com/research/bdh
- **Code Repository**: https://github.com/pathwaycom/bdh

**Key Inspirations:**
- **Biologically-Plausible Neural Networks**: Scale-free networks of locally-interacting neuron particles
- **Hebbian Learning Mechanisms**: Synaptic plasticity with integrate-and-fire thresholding
- **Sparse Positive Activations**: Monosemantic neuron representations for interpretability
- **Graph-Based Distributed Computing**: Local edge-reweighting processes as "equations of reasoning"

The Dragon Hatchling research demonstrates how attention mechanisms in state-of-the-art language models correspond to attention mechanisms observed in the brain, formally converging as closed-form local graph dynamics at neurons and synapses.

#### Anthropic's Context Management
The PSI (Persistent Semantic Index) component is inspired by Anthropic's context management research:

**"Managing context on the Claude Developer Platform"**
- **Article**: https://www.anthropic.com/news/context-management
- **Documentation**: https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool

**Key Inspirations:**
- **Memory Tool Architecture**: File-based system for storing information outside context windows
- **Context Editing**: Automatic clearing of stale information while preserving conversation flow
- **Persistent Knowledge**: Building knowledge bases that improve performance over time
- **Cross-Session Learning**: Maintaining insights across successive agentic sessions

### Technical Synthesis

SRCA combines these inspirations into a novel architecture:

1. **BDH from Dragon Hatchling**: Implements true synaptic plasticity with reward-gated Hebbian learning
2. **PSI from Anthropic**: Creates persistent semantic memory with protected guardrail entries
3. **Original Contributions**: Adds self-awareness, valence control, and cybersecurity-specific applications

This synthesis creates a system that maintains biological plausibility while achieving practical performance in autonomous decision-making scenarios.

## Future Development

### Potential Enhancements
- Integration with real SIEM/SOAR platforms
- Distributed processing across multiple systems
- Advanced hyperparameter optimization
- Extended action repertoires for complex scenarios

### Research Directions
- Comparative studies with baseline security systems
- Human-in-the-loop learning mechanisms
- Transfer learning across security domains
- Scalability analysis for enterprise environments

## License & Citation

© 2025 Shane D. Shook, All Rights Reserved

If you use this work in research, please cite:
```
Shook, S.D. (2025). Self-Regulated Cognitive Architecture (SRCA): 
A Novel AI System for Autonomous Cybersecurity Decision-Making.
```

## Contact & Support

For questions, collaborations, or technical support, please contact the author through the repository issues system.

---

*This implementation represents cutting-edge research in cognitive AI architectures and autonomous cybersecurity systems. The code is provided for research and educational purposes.*