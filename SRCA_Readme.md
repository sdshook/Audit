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

#### Bidirectional Hebbian Memory (BDH)
Implements true biological-inspired persistence through synaptic plasticity:

```python
# Reward-gated synaptic updates
if reward > 0:
    entry["W"] += BDH_ETA_POT * reward * (outer + np.outer(entry["elig_pos"], entry["elig_pos"]))
else:
    if not entry["protected"]:
        entry["W"] -= BDH_ETA_DEP * abs(reward) * (outer + np.outer(entry["elig_neg"], entry["elig_neg"]))
```

**Features:**
- **Long-Term Potentiation (LTP)**: Positive rewards strengthen synaptic connections
- **Long-Term Depression (LTD)**: Negative rewards weaken connections (unless protected)
- **Eligibility Traces**: Temporal credit assignment mimicking biological synaptic tagging
- **Dual Stores**: Reflective (analytical) and Empathic (contextual) processing

#### Persistent Semantic Index (PSI)
Functions as cortical memory system with:
- Valence-weighted semantic associations
- Protected memories resistant to modification
- Access-based memory strengthening
- Content-addressable retrieval

#### Memory Consolidation
Mimics hippocampal-cortical transfer:
- Short-term episodic traces in BDH (hippocampus-like)
- Long-term semantic consolidation in PSI (cortex-like)
- Threshold-based transfer based on cumulative reward

### Cognitive Processing

#### CMNN Synaptic Integration
- Each mesh node acts like a cortical column
- Message passing simulates inter-columnar communication
- Collective intelligence emerges from synaptic integration

#### Self-Regulation Mechanisms
- **Valence Controller**: Emotional modulation of cognitive processes
- **Arrogance Detection**: Prevents overconfident decisions
- **Empathy Integration**: Incorporates human feedback

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

### Novel Memory Constructs
- **Synaptic Persistence**: True biological-inspired memory through synaptic plasticity
- **Multi-Scale Processing**: Local BDH updates, distributed CMNN processing, global PSI integration
- **Cognitive Control**: Persistent monitoring and regulation of cognitive processes

### Technical Advances
- Integration of symbolic (PSI) and connectionist (neural network) approaches
- Real-time self-awareness and behavioral regulation
- Domain-specific application to cybersecurity operations

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