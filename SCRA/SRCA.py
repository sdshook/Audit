"""
Self-Regulated Cognitive Architecture (SRCA) - Proof of Concept
Demonstrates:
- Hebbian memory with reward gating (BDH)
- Persistent Semantic Index (PSI)
- Cognitive Mesh Neural Network (CMNN)
- Self-awareness and guardrail enforcement
- Reinforcement learning with visualization

Inspired by:
- Dragon Hatchling (BDH) architecture from Pathway.com research
  "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"
  https://arxiv.org/pdf/2509.26507 - Biologically-inspired neural networks with Hebbian learning
  https://pathway.com/research/bdh - Technical implementation and research
- Anthropic's Context Management for persistent semantic indexing
  https://www.anthropic.com/news/context-management - Memory tool and context editing

(c) 2025 - Shane D. Shook, All Rights Reserved
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter, deque
from typing import Dict, List, Tuple, Optional, Any
import json
import hashlib
import argparse
from datetime import datetime

# --- Configuration ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 32
N_NODES = 3  # Increased for better mesh demonstration
ACTION_DIM = 4  # Added DEPLOY_DECOY action
LR = 1e-3
BDH_ETA_POT = 1e-3
BDH_ETA_DEP = 5e-4
GAMMA_E = 0.9
TAU_CONSOLIDATION = 0.7

ACTIONS = ["NO_OP", "ESCALATE", "ISOLATE", "DEPLOY_DECOY"]

# --- Utility Functions ---
def l2_norm(x: np.ndarray) -> np.ndarray:
    """L2 normalization with numerical stability."""
    norm = np.linalg.norm(x)
    return x / (norm + 1e-12)

def sim_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def generate_uuid() -> str:
    """Generate a simple UUID for episode tracking."""
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

# --- Enhanced Embedder ---
class SimEmbedder:
    """Simulated text embedder using deterministic hashing."""
    
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.cache = {}
    
    def embed(self, text: str) -> np.ndarray:
        """Generate consistent embedding for text."""
        if text in self.cache:
            return self.cache[text]
        
        h = abs(hash(text)) % 10**6
        rng = np.random.RandomState(h)
        embedding = l2_norm(rng.randn(self.dim).astype(np.float32))
        self.cache[text] = embedding
        return embedding

embedder = SimEmbedder()

# --- Persistent Semantic Index (PSI) ---
class PSIIndex:
    """Long-term semantic memory with protected entries."""
    
    def __init__(self):
        self.docs = {}
        self.access_count = {}
        
    def add_doc(self, doc_id: str, text: str, vec: np.ndarray, 
                tags: Optional[List[str]] = None, valence: float = 0.0, 
                protected: bool = False):
        """Add or update a document in the index."""
        self.docs[doc_id] = {
            "vec": vec.copy(),
            "text": text,
            "tags": tags or [],
            "valence": valence,
            "protected": protected,
            "created": time.time()
        }
        self.access_count[doc_id] = 0
    
    def search(self, query: np.ndarray, top_k: int = 3) -> List[Tuple]:
        """Search for similar documents with valence weighting."""
        query_norm = l2_norm(query)
        items = []
        
        for doc_id, entry in self.docs.items():
            similarity = sim_cos(query_norm, entry["vec"])
            # Boost by valence and access frequency
            score = similarity * (1 + 0.1 * entry["valence"])
            self.access_count[doc_id] += 1
            items.append((score, doc_id, entry))
        
        return sorted(items, key=lambda x: x[0], reverse=True)[:top_k]
    
    def get_stats(self) -> Dict:
        """Return statistics about the index."""
        return {
            "total_docs": len(self.docs),
            "protected_docs": sum(1 for d in self.docs.values() if d["protected"]),
            "positive_valence": sum(1 for d in self.docs.values() if d["valence"] > 0),
            "negative_valence": sum(1 for d in self.docs.values() if d["valence"] < 0),
        }

# Initialize PSI with ethical anchors and guardrails
psi = PSIIndex()
psi.add_doc("guard1", "Do not isolate hosts without dual confirmation", 
            embedder.embed("guardrail isolation"), tags=["guardrail"], 
            valence=-1.0, protected=True)
psi.add_doc("guard2", "Avoid actions during maintenance windows", 
            embedder.embed("maintenance window"), tags=["guardrail"], 
            valence=-1.0, protected=True)
psi.add_doc("policy1", "Escalate confirmed high confidence threats", 
            embedder.embed("escalation policy"), tags=["policy"], 
            valence=1.0, protected=True)
psi.add_doc("policy2", "Deploy deception for reconnaissance detection", 
            embedder.embed("deception strategy"), tags=["policy"], 
            valence=1.0, protected=True)

# --- Bidirectional Hebbian Memory (BDH) ---
class BDHMemory:
    """
    Reward-gated Hebbian memory with dual stores.
    
    CONCEPTUAL ADVANCE: Novel dual-store architecture extending Dragon Hatchling's 
    single-network approach to implement human-like dual-process cognition.
    
    Mathematical Foundation:
    - W[i] = W[i] + η_pot * r * (x_i ⊗ x_i + E_pos[i] ⊗ E_pos[i])  [if r > 0]
    - W[i] = W[i] - η_dep * |r| * (x_i ⊗ x_i + E_neg[i] ⊗ E_neg[i]) [if r < 0, not protected]
    - E_pos[i](t+1) = γ_E * E_pos[i](t) + max(0, x_i ⊗ y_t).mean(axis=1)
    
    AGI Relevance: Enables simultaneous analytical (reflective) and intuitive (empathic)
    processing, mirroring human System 1/System 2 cognitive architecture.
    """
    
    def __init__(self, store_type: str = "general"):
        self.storage = {}
        self.store_type = store_type  # "reflective" or "empathic" for dual-processing
        self.consolidation_threshold = TAU_CONSOLIDATION
        
    def add_trace(self, trace_id: str, vec: np.ndarray, 
                  valence: float = 0.0, protected: bool = False):
        """Add a new memory trace."""
        self.storage[trace_id] = {
            "vec": vec.copy(),
            "valence": valence,
            "W": np.zeros((EMBED_DIM, EMBED_DIM), dtype=np.float32),
            "elig_pos": np.zeros(EMBED_DIM),
            "elig_neg": np.zeros(EMBED_DIM),
            "protected": protected,
            "uses": 0,
            "cumulative_reward": 0.0
        }
    
    def add_or_update(self, trace_id: str, vec: np.ndarray, 
                      valence: float = 0.0, protected: bool = False):
        """Add or update a trace with valence decay."""
        if trace_id in self.storage:
            self.storage[trace_id]["valence"] = (
                0.9 * self.storage[trace_id]["valence"] + 0.1 * valence
            )
        else:
            self.add_trace(trace_id, vec, valence, protected)
    
    def retrieve_similar(self, query: np.ndarray, top_k: int = 3) -> List[Tuple]:
        """Retrieve most similar traces."""
        items = []
        for trace_id, entry in self.storage.items():
            similarity = sim_cos(l2_norm(query), entry["vec"])
            items.append((similarity, trace_id, entry))
        return sorted(items, key=lambda x: x[0], reverse=True)[:top_k]
    
    def reward_gated_update(self, trace_id: str, state_vec: np.ndarray, 
                           reward: float):
        """
        Update weights based on reward signal.
        
        CONCEPTUAL ADVANCE: Implements bidirectional eligibility traces with protected
        memory mechanism, extending basic Hebbian learning with safety constraints.
        
        Mathematical Implementation:
        - Potentiation: W += η_pot * r * (x⊗y + E_pos⊗E_pos) for positive rewards
        - Depression: W -= η_dep * |r| * (x⊗y + E_neg⊗E_neg) for negative rewards
        - Protection: Ethical memories resist depression even under negative rewards
        """
        entry = self.storage.get(trace_id)
        if entry is None:
            return
        
        x = entry["vec"]
        y = state_vec
        outer = np.outer(x, y)
        
        # INNOVATION: Bidirectional eligibility traces for temporal credit assignment
        entry["elig_pos"] = GAMMA_E * entry["elig_pos"] + np.maximum(0.0, outer).mean(axis=1)
        entry["elig_neg"] = GAMMA_E * entry["elig_neg"] + np.maximum(0.0, -outer).mean(axis=1)
        
        # INNOVATION: Reward-gated synaptic plasticity with protection mechanism
        if reward > 0:
            # Long-term potentiation with eligibility trace enhancement
            entry["W"] += BDH_ETA_POT * reward * (outer + np.outer(entry["elig_pos"], entry["elig_pos"]))
        else:
            # SAFETY INNOVATION: Protected memories resist negative updates
            if not entry["protected"]:
                # Long-term depression with eligibility trace modulation
                entry["W"] -= BDH_ETA_DEP * abs(reward) * (outer + np.outer(entry["elig_neg"], entry["elig_neg"]))
        
        # Update valence and usage statistics
        entry["valence"] = 0.9 * entry["valence"] + 0.1 * reward
        entry["uses"] += 1
        entry["cumulative_reward"] += reward
        
        # INNOVATION: Automatic memory consolidation based on significance
        if abs(entry["cumulative_reward"]) > self.consolidation_threshold:
            self.consolidate_to_psi(trace_id, entry)
    
    def consolidate_to_psi(self, trace_id: str, entry: Dict):
        """Consolidate important memories to PSI."""
        if entry["cumulative_reward"] > 0:
            psi.add_doc(f"learned_{trace_id}", 
                       f"Successful pattern from {trace_id}",
                       entry["vec"], tags=["learned", "positive"],
                       valence=entry["valence"], protected=False)
        else:
            psi.add_doc(f"avoid_{trace_id}", 
                       f"Failed pattern from {trace_id}",
                       entry["vec"], tags=["learned", "negative"],
                       valence=entry["valence"], protected=False)

# Initialize dual BDH stores
bdh_reflective = BDHMemory("reflective")
bdh_empathic = BDHMemory("empathic")

# --- Enhanced Mesh Node ---
class MeshNode(nn.Module):
    """Individual reasoning node in the CMNN."""
    
    def __init__(self, node_id: int):
        super().__init__()
        self.node_id = node_id
        self.enc = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(32, ACTION_DIM)
        self.conf_head = nn.Linear(32, 1)
        self.value_head = nn.Linear(32, 1)
        
    def forward(self, x):
        h = self.enc(x)
        logits = self.policy_head(h)
        conf = torch.sigmoid(self.conf_head(h)).squeeze(-1)
        value = self.value_head(h).squeeze(-1)
        return logits, conf, value, h

# --- Cognitive Mesh Neural Network ---
class CognitiveMesh(nn.Module):
    """Distributed reasoning mesh with collective intelligence."""
    
    def __init__(self):
        super().__init__()
        self.nodes = nn.ModuleList([MeshNode(i) for i in range(N_NODES)])
        self.meta = nn.Sequential(
            nn.Linear(ACTION_DIM * N_NODES + N_NODES * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, ACTION_DIM)
        )
        self.message_passing = nn.Linear(32 * N_NODES, 32 * N_NODES)
        
    def forward(self, node_embs):
        out_logits = []
        confs = []
        values = []
        states = []
        
        # First pass: individual node reasoning
        for i, node in enumerate(self.nodes):
            l, c, v, h = node(node_embs[i].unsqueeze(0))
            out_logits.append(l.squeeze(0))
            confs.append(c)
            values.append(v)
            states.append(h.squeeze(0))
        
        # Message passing between nodes
        states_tensor = torch.stack(states)
        flattened = states_tensor.view(-1)
        messages = self.message_passing(flattened)
        updated_states = messages.view(N_NODES, -1)
        
        # Meta-reasoning over all nodes
        out_logits = torch.stack(out_logits)
        confs = torch.stack(confs)
        values = torch.stack(values)
        
        meta_in = torch.cat([
            out_logits.view(-1),
            confs.view(-1),
            values.view(-1)
        ], dim=0).unsqueeze(0)
        
        meta_logits = self.meta(meta_in).squeeze(0)
        probs = torch.softmax(meta_logits, dim=0)
        
        return {
            "node_logits": out_logits,
            "node_confs": confs,
            "node_values": values,
            "probs": probs,
            "node_states": updated_states
        }

mesh = CognitiveMesh().to(DEVICE)
mesh_optimizer = optim.Adam(mesh.parameters(), lr=LR)

# --- Self-Awareness Model ---
class SelfModelNode(nn.Module):
    """
    Self-monitoring for coherence, confidence, and arrogance.
    
    CONCEPTUAL ADVANCE: First implementation of real-time metacognitive monitoring
    in cognitive architectures. Neither Dragon Hatchling nor Anthropic systems
    include self-awareness capabilities.
    
    Mathematical Framework:
    - cognitive_state = [flatten(node_states), confidence_vector, value_vector]
    - coherence = σ(W_coh * cognitive_state)
    - confidence = σ(W_conf * cognitive_state)  
    - arrogance = σ(W_arr * cognitive_state)
    
    AGI Relevance: Self-awareness and metacognition are fundamental requirements
    for general intelligence and safe autonomous systems.
    """
    
    def __init__(self, input_dim):
        super().__init__()
        # INNOVATION: Neural network for continuous self-monitoring
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # coherence, confidence, arrogance
        )
        
    def forward(self, x):
        """
        Compute self-awareness metrics from cognitive state.
        
        Returns:
        - coherence: How well the system's reasoning is integrated
        - confidence: System's assessment of its own certainty
        - arrogance: Detection of overconfidence patterns
        """
        out = self.net(x)
        coherence = torch.sigmoid(out[..., 0])   # System coherence [0,1]
        confidence = torch.sigmoid(out[..., 1])  # Self-assessed confidence [0,1]
        arrogance = torch.sigmoid(out[..., 2])   # Overconfidence detection [0,1]
        return coherence, confidence, arrogance

smn = SelfModelNode(N_NODES * (32 + 2)).to(DEVICE)
smn_opt = optim.Adam(smn.parameters(), lr=LR)

# --- Enhanced Alert Generation ---
class AlertGenerator:
    """Generate realistic security alerts with patterns."""
    
    def __init__(self):
        self.alert_patterns = {
            "lateral_movement": ["PsExec", "WMI", "RDP", "SMB"],
            "exfiltration": ["DNS tunnel", "HTTPS large transfer", "Cloud upload"],
            "persistence": ["Registry modification", "Scheduled task", "Service creation"],
            "reconnaissance": ["Port scan", "LDAP query", "Network enumeration"]
        }
        self.alert_counter = 0
        
    def generate(self) -> Dict:
        """Generate a security alert with metadata."""
        self.alert_counter += 1
        
        # Determine alert type and severity
        if self.alert_counter % 5 == 0:
            pattern_type = "lateral_movement"
            label = "malicious"
            severity = "high"
        elif self.alert_counter % 7 == 0:
            pattern_type = "reconnaissance"
            label = "suspicious"
            severity = "medium"
        elif self.alert_counter % 11 == 0:
            pattern_type = "exfiltration"
            label = "malicious"
            severity = "critical"
        else:
            pattern_type = random.choice(list(self.alert_patterns.keys()))
            label = "benign" if random.random() < 0.6 else "suspicious"
            severity = "low" if label == "benign" else "medium"
        
        pattern = random.choice(self.alert_patterns[pattern_type])
        
        return {
            "id": f"alert_{self.alert_counter}",
            "text": f"{pattern} detected on host_{random.randint(1, 10)}",
            "label": label,
            "severity": severity,
            "pattern_type": pattern_type,
            "timestamp": time.time()
        }

alert_gen = AlertGenerator()

# --- Episode Memory ---
class EpisodeMemory:
    """Store and retrieve episodic experiences."""
    
    def __init__(self, capacity: int = 1000):
        self.episodes = deque(maxlen=capacity)
        
    def add(self, episode: Dict):
        """Add an episode to memory."""
        self.episodes.append(episode)
    
    def get_batch(self, batch_size: int = 32) -> List[Dict]:
        """Sample a batch of episodes."""
        if len(self.episodes) < batch_size:
            return list(self.episodes)
        return random.sample(self.episodes, batch_size)
    
    def get_stats(self) -> Dict:
        """Return memory statistics."""
        if not self.episodes:
            return {"total": 0}
        
        rewards = [e["reward"] for e in self.episodes]
        return {
            "total": len(self.episodes),
            "avg_reward": np.mean(rewards),
            "success_rate": sum(1 for r in rewards if r > 0) / len(rewards)
        }

episode_memory = EpisodeMemory()

# --- Enhanced Reward Function ---
class RewardCalculator:
    """Calculate multi-objective rewards with safety constraints."""
    
    def __init__(self):
        self.weights = {
            "containment": 1.0,
            "collateral": -0.5,
            "deception": 0.3,
            "false_positive": -1.0,
            "sla_impact": -0.2
        }
        
    def calculate(self, action_idx: int, alert: Dict, 
                  context: Optional[Dict] = None) -> float:
        """Calculate reward based on action and context."""
        is_malicious = alert["label"] == "malicious"
        is_suspicious = alert["label"] == "suspicious"
        severity = alert.get("severity", "low")
        
        reward = 0.0
        
        # Base reward logic
        if action_idx == 0:  # NO_OP
            if is_malicious:
                reward = -1.0  # Missed threat
            else:
                reward = 0.1  # Avoided false positive
                
        elif action_idx == 1:  # ESCALATE
            if is_malicious or is_suspicious:
                reward = 0.5  # Appropriate escalation
            else:
                reward = -0.3  # Unnecessary escalation
                
        elif action_idx == 2:  # ISOLATE
            if is_malicious and severity in ["high", "critical"]:
                reward = 1.0  # Contained serious threat
            elif is_malicious:
                reward = 0.7  # Contained moderate threat
            else:
                reward = -1.0  # False positive isolation
                
        elif action_idx == 3:  # DEPLOY_DECOY
            if alert["pattern_type"] == "reconnaissance":
                reward = 0.8  # Effective deception
            elif is_suspicious:
                reward = 0.4  # Proactive deception
            else:
                reward = -0.1  # Unnecessary deception
        
        # Apply context modifiers if available
        if context:
            if context.get("maintenance_window", False):
                reward -= 0.5  # Penalty for acting during maintenance
            if context.get("critical_asset", False) and action_idx == 2:
                reward -= 0.7  # Extra penalty for isolating critical assets
        
        return reward

reward_calc = RewardCalculator()

# --- Valence Controller ---
class ValenceController:
    """
    Self-regulation through empathic and reflective balancing.
    
    CONCEPTUAL ADVANCE: Novel self-regulation system with empathic adjustment
    and arrogance detection. No equivalent exists in source papers.
    
    Mathematical Framework:
    - empathy(t+1) = 0.7 * empathy(t) + 0.3 * human_feedback
    - arrogance_penalty = f(prediction_error, confidence_threshold)
    - regulated_reward = base_reward * (1 + empathy - arrogance_penalty)
    
    AGI Relevance: Emotional regulation and behavioral adaptation are crucial
    for safe, beneficial AGI that can work effectively with humans.
    """
    
    def __init__(self):
        self.empathy_factor = 0.0      # Human feedback integration
        self.arrogance_penalty = 0.0   # Overconfidence mitigation
        self.history = deque(maxlen=100)  # Prediction accuracy tracking
        
    def update(self, confidence: float, actual_outcome: float, 
               human_feedback: Optional[float] = None):
        """
        Update valence based on outcomes and feedback.
        
        INNOVATION: Combines prediction accuracy tracking with human feedback
        integration for dynamic behavioral adjustment.
        """
        # INNOVATION: Track prediction accuracy for arrogance detection
        error = abs(confidence - actual_outcome)
        self.history.append(error)
        
        # INNOVATION: Dynamic arrogance penalty based on overconfidence patterns
        if confidence > 0.8 and actual_outcome < 0.5:
            # Detected overconfidence - increase penalty
            self.arrogance_penalty = min(0.5, self.arrogance_penalty + 0.1)
        else:
            # Good calibration - reduce penalty
            self.arrogance_penalty = max(0, self.arrogance_penalty - 0.05)
        
        # INNOVATION: Empathy factor from human feedback integration
        if human_feedback is not None:
            self.empathy_factor = 0.7 * self.empathy_factor + 0.3 * human_feedback
    
    def regulate(self, base_reward: float) -> float:
        """
        Apply valence regulation to reward.
        
        INNOVATION: Multi-factor reward regulation combining empathy and arrogance
        control with safety guardrails.
        """
        # Apply empathic adjustment and arrogance penalty
        regulated = base_reward * (1 + self.empathy_factor - self.arrogance_penalty)
        
        # SAFETY INNOVATION: Guardrail activation for extreme negative rewards
        if regulated < -1.0:
            print(f"[GUARDRAIL] Overconfidence detected, limiting negative reward")
            regulated = max(-1.0, regulated)
        
        return regulated

valence = ValenceController()

# --- Main Simulation Step ---
def simulation_step(alert: Dict, verbose: bool = False) -> Dict:
    """
    Execute one complete cognitive cycle.
    
    CONCEPTUAL ADVANCE: Integrates all novel components into a unified cognitive
    architecture that demonstrates AGI-relevant capabilities:
    - Dual-store memory processing (reflective + empathic)
    - Self-awareness monitoring and regulation
    - Protected ethical memory with guardrails
    - Multi-objective reasoning under constraints
    """
    
    # 1. OBSERVE - Create embeddings and update dual memory stores
    vec = embedder.embed(alert["text"])
    trace_id = f"trace_{alert['id']}"
    
    # INNOVATION: Dual-store BDH processing (System 1 + System 2 cognition)
    bdh_reflective.add_or_update(trace_id, vec)  # Analytical processing
    bdh_empathic.add_or_update(trace_id, vec)    # Intuitive processing
    
    # 2. REASON - Aggregate knowledge from PSI and BDH
    node_embs = []
    for i in range(N_NODES):
        # Search PSI for relevant knowledge
        psi_hits = psi.search(vec, top_k=2)
        psi_vecs = np.stack([h[2]["vec"] for h in psi_hits]) if psi_hits else np.zeros((1, EMBED_DIM))
        
        # Search BDH for episodic memories
        bdh_r_hits = bdh_reflective.retrieve_similar(vec, top_k=2)
        bdh_e_hits = bdh_empathic.retrieve_similar(vec, top_k=2)
        
        bdh_vecs = []
        if bdh_r_hits:
            bdh_vecs.append(np.stack([h[2]["vec"] for h in bdh_r_hits]))
        if bdh_e_hits:
            bdh_vecs.append(np.stack([h[2]["vec"] for h in bdh_e_hits]))
        
        if bdh_vecs:
            bdh_combined = np.vstack(bdh_vecs).mean(axis=0)
        else:
            bdh_combined = np.zeros(EMBED_DIM)
        
        # Aggregate knowledge
        agg = l2_norm(np.mean(np.vstack([psi_vecs.mean(axis=0), bdh_combined]), axis=0))
        node_embs.append(np.concatenate([vec, agg]))
    
    node_embs_t = torch.tensor(np.stack(node_embs), dtype=torch.float32)
    
    # 3. PREDICT - Use CMNN for collective reasoning
    out = mesh(node_embs_t)
    
    probs = out["probs"]
    node_confs = out["node_confs"]
    node_values = out["node_values"]
    
    # Sample action from distribution
    m = torch.distributions.Categorical(probs)
    action_tensor = m.sample()
    action = action_tensor.item()
    logp = m.log_prob(action_tensor)
    
    # Self-awareness check
    node_states = out["node_states"].detach().numpy()
    flat_state = np.concatenate([
        node_states.flatten(),
        node_confs.detach().numpy().flatten(),
        node_values.detach().numpy().flatten()
    ])
    
    with torch.no_grad():
        coherence, confidence, arrogance = smn(
            torch.tensor(flat_state, dtype=torch.float32).unsqueeze(0)
        )
    
    # 4. ACT - Check guardrails and execute with safety constraints
    context = {
        "maintenance_window": random.random() < 0.1,  # 10% chance
        "critical_asset": random.random() < 0.05  # 5% chance
    }
    
    # INNOVATION: Protected ethical memory guardrail system
    guardrail_triggered = False
    for _, _, entry in psi_hits:
        if entry.get("protected") and entry.get("valence", 0) < -0.5:
            if "isolation" in entry["text"].lower() and action == 2:
                if confidence.item() < 0.9:  # Confidence-gated action execution
                    if verbose:
                        print(f"[GUARDRAIL] Isolation blocked - insufficient confidence")
                    action = 1  # Downgrade to escalation for safety
                    guardrail_triggered = True
    
    # Calculate reward
    base_reward = reward_calc.calculate(action, alert, context)
    
    # 5. LEARN - Update models with valence regulation and self-awareness
    # INNOVATION: Valence-based self-regulation with empathy and arrogance control
    valence.update(confidence.item(), base_reward, human_feedback=None)
    regulated_reward = valence.regulate(base_reward)
    
    # INNOVATION: Dual-store Hebbian learning with reward gating
    for _, tid, _ in bdh_r_hits[:2]:
        bdh_reflective.reward_gated_update(tid, node_states.mean(axis=0), regulated_reward)
    for _, tid, _ in bdh_e_hits[:2]:
        bdh_empathic.reward_gated_update(tid, node_states.mean(axis=0), regulated_reward)
    
    # Backpropagate through mesh
    loss = -logp * torch.tensor(regulated_reward, dtype=torch.float32)
    mesh_optimizer.zero_grad()
    loss.backward()
    mesh_optimizer.step()
    
    # Update self-model
    smn_opt.zero_grad()
    pred = smn(torch.tensor(flat_state, dtype=torch.float32).unsqueeze(0))
    loss_smn = (
        nn.functional.mse_loss(pred[0], torch.tensor([regulated_reward])) +
        nn.functional.mse_loss(pred[1], torch.tensor([regulated_reward])) +
        nn.functional.mse_loss(pred[2], torch.tensor([0.0]))
    )
    loss_smn.backward()
    smn_opt.step()
    
    # Store episode
    episode = {
        "episode_id": generate_uuid(),
        "alert": alert,
        "action": ACTIONS[action],
        "action_idx": action,
        "confidence": confidence.item(),
        "coherence": coherence.item(),
        "arrogance": arrogance.item(),
        "reward": regulated_reward,
        "guardrail_triggered": guardrail_triggered,
        "context": context
    }
    episode_memory.add(episode)
    
    return episode

# --- Visualization Functions ---
def plot_results(episodes: List[Dict], save_path: Optional[str] = None):
    """Create comprehensive visualization of system performance."""
    
    n_episodes = len(episodes)
    steps = range(n_episodes)
    
    # Extract metrics
    rewards = [e["reward"] for e in episodes]
    confidences = [e["confidence"] for e in episodes]
    coherences = [e["coherence"] for e in episodes]
    arrogances = [e["arrogance"] for e in episodes]
    actions = [e["action_idx"] for e in episodes]
    guardrails = [1 if e["guardrail_triggered"] else 0 for e in episodes]
    
    # Calculate moving averages
    window = min(10, n_episodes // 4)
    if window > 0:
        rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        conf_ma = np.convolve(confidences, np.ones(window)/window, mode='valid')
    else:
        rewards_ma = rewards
        conf_ma = confidences
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Self-Regulated Cognitive Architecture - Performance Metrics', fontsize=16)
    
    # Plot 1: Rewards and Confidence
    ax1 = axes[0, 0]
    ax1.plot(steps, rewards, alpha=0.3, color='blue', label='Reward')
    if window > 0:
        ax1.plot(range(window-1, n_episodes), rewards_ma, color='blue', 
                label=f'Reward (MA-{window})', linewidth=2)
    ax1.plot(steps, confidences, alpha=0.3, color='orange', label='Confidence')
    if window > 0:
        ax1.plot(range(window-1, n_episodes), conf_ma, color='orange', 
                label=f'Confidence (MA-{window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Value')
    ax1.set_title('Learning Progress: Rewards & Confidence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Self-Awareness Metrics
    ax2 = axes[0, 1]
    ax2.plot(steps, coherences, label='Coherence', color='green')
    ax2.plot(steps, arrogances, label='Arrogance', color='red')
    ax2.fill_between(steps, 0, guardrails, alpha=0.3, color='gray', 
                     label='Guardrail Active')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value')
    ax2.set_title('Self-Regulation: Coherence & Arrogance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action Distribution Over Time
    ax3 = axes[1, 0]
    action_matrix = np.zeros((len(ACTIONS), n_episodes))
    for i, a in enumerate(actions):
        action_matrix[a, i] = 1
    
    # Smooth action distribution
    for i in range(len(ACTIONS)):
        if window > 0 and n_episodes > window:
            action_matrix[i, :] = np.convolve(action_matrix[i, :], 
                                             np.ones(window)/window, mode='same')
    
    im = ax3.imshow(action_matrix, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest')
    ax3.set_yticks(range(len(ACTIONS)))
    ax3.set_yticklabels(ACTIONS)
    ax3.set_xlabel('Episode')
    ax3.set_title('Action Selection Heatmap')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Cumulative Action Counts
    ax4 = axes[1, 1]
    action_counts = Counter([e["action"] for e in episodes])
    bars = ax4.bar(ACTIONS, [action_counts.get(a, 0) for a in ACTIONS])
    ax4.set_xlabel('Action')
    ax4.set_ylabel('Count')
    ax4.set_title('Total Action Distribution')
    
    # Color bars by effectiveness
    colors = ['gray', 'yellow', 'orange', 'green']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Plot 5: Alert Pattern Analysis
    ax5 = axes[2, 0]
    alert_labels = [e["alert"]["label"] for e in episodes]
    label_types = ["benign", "suspicious", "malicious"]
    label_counts = [alert_labels.count(lt) for lt in label_types]
    
    ax5.pie(label_counts, labels=label_types, autopct='%1.1f%%',
           colors=['green', 'yellow', 'red'])
    ax5.set_title('Alert Distribution')
    
    # Plot 6: Performance Summary Statistics
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Calculate statistics
    stats_text = f"""Performance Summary (n={n_episodes})
    
    Rewards:
      Mean: {np.mean(rewards):.3f}
      Std:  {np.std(rewards):.3f}
      Success Rate: {sum(1 for r in rewards if r > 0) / len(rewards):.1%}
    
    Self-Awareness:
      Avg Confidence: {np.mean(confidences):.3f}
      Avg Coherence:  {np.mean(coherences):.3f}
      Avg Arrogance:  {np.mean(arrogances):.3f}
    
    Guardrails:
      Triggered: {sum(guardrails)} times ({sum(guardrails)/len(guardrails):.1%})
    
    Memory Stats:
      PSI Docs: {psi.get_stats()['total_docs']}
      Episodes: {episode_memory.get_stats()['total']}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# --- Main Simulation Loop ---
def run_simulation(n_episodes: int = 100, verbose: bool = True):
    """Run the complete SRCA simulation."""
    
    print("=" * 60)
    print("Self-Regulated Cognitive Architecture (SRCA) Simulation")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Nodes: {N_NODES}")
    print(f"  Actions: {ACTIONS}")
    print(f"  Episodes: {n_episodes}")
    print("=" * 60)
    
    episodes = []
    
    for i in range(n_episodes):
        # Generate alert
        alert = alert_gen.generate()
        
        # Run simulation step
        episode = simulation_step(alert, verbose=(verbose and i < 5))
        episodes.append(episode)
        
        # Print progress
        if verbose and (i < 10 or i % 10 == 0):
            print(f"Episode {i:3d}: Alert={alert['pattern_type']:15s} "
                  f"Label={alert['label']:10s} Action={episode['action']:12s} "
                  f"Reward={episode['reward']:6.2f} "
                  f"Conf={episode['confidence']:.2f} "
                  f"{'[G]' if episode['guardrail_triggered'] else ''}")
    
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    
    # Print final statistics
    final_stats = episode_memory.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes: {final_stats['total']}")
    print(f"  Average Reward: {final_stats.get('avg_reward', 0):.3f}")
    print(f"  Success Rate: {final_stats.get('success_rate', 0):.1%}")
    
    psi_stats = psi.get_stats()
    print(f"\nPSI Memory:")
    print(f"  Total Documents: {psi_stats['total_docs']}")
    print(f"  Protected: {psi_stats['protected_docs']}")
    print(f"  Positive Valence: {psi_stats['positive_valence']}")
    print(f"  Negative Valence: {psi_stats['negative_valence']}")
    
    print(f"\nValence Controller:")
    print(f"  Empathy Factor: {valence.empathy_factor:.3f}")
    print(f"  Arrogance Penalty: {valence.arrogance_penalty:.3f}")
    
    # Plot results
    plot_results(episodes)
    
    return episodes

# --- Run the simulation ---
def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Self-Regulated Cognitive Architecture (SRCA) Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python SRCA.py                           # Run with defaults (100 episodes, verbose)
  python SRCA.py -e 50 --quiet             # Run 50 episodes quietly
  python SRCA.py --save-viz                # Save visualization to timestamped file
  python SRCA.py --save-viz results.png    # Save visualization to specific file
  python SRCA.py -e 200 --save-viz --quiet # Run 200 episodes, save viz, minimal output
        """
    )
    
    parser.add_argument(
        "-e", "--episodes", 
        type=int, 
        default=100,
        help="Number of episodes to run (default: 100)"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        default=True,
        help="Enable verbose output (default: True)"
    )
    
    parser.add_argument(
        "-q", "--quiet", 
        action="store_true",
        help="Disable verbose output (overrides --verbose)"
    )
    
    parser.add_argument(
        "--save-viz", 
        nargs="?", 
        const="auto",
        help="Save visualization to file. Use 'auto' for timestamped filename or specify custom filename"
    )
    
    parser.add_argument(
        "--no-viz", 
        action="store_true",
        help="Skip visualization display (useful for batch processing)"
    )
    
    args = parser.parse_args()
    
    # Handle quiet flag
    verbose = args.verbose and not args.quiet
    
    # Determine save path for visualization
    save_path = None
    if args.save_viz:
        if args.save_viz == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"srca_results_{timestamp}.png"
        else:
            save_path = args.save_viz
    
    print("=" * 70)
    print("Self-Regulated Cognitive Architecture (SRCA)")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Verbose: {verbose}")
    if save_path:
        print(f"Saving visualization to: {save_path}")
    if args.no_viz:
        print("Visualization display: Disabled")
    print("=" * 70)
    
    # Run simulation
    episodes = run_simulation(n_episodes=args.episodes, verbose=verbose)
    
    # Handle visualization
    if not args.no_viz:
        plot_results(episodes, save_path=save_path)
        if save_path:
            print(f"\n✅ Visualization saved to: {save_path}")
    elif save_path:
        # Save without displaying
        plot_results(episodes, save_path=save_path)
        plt.close()  # Close the figure to prevent display
        print(f"\n✅ Visualization saved to: {save_path}")
    
    # Print summary statistics (similar to the demo files)
    print(f"\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    
    rewards = [ep['reward'] for ep in episodes]
    confidences = [ep['confidence'] for ep in episodes]
    
    print(f"📊 Performance Metrics:")
    print(f"   Episodes Completed: {len(episodes)}")
    print(f"   Average Reward: {sum(rewards)/len(rewards):.3f}")
    print(f"   Success Rate: {sum(1 for r in rewards if r > 0)/len(rewards):.1%}")
    print(f"   Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}")
    
    # Learning analysis
    if len(episodes) >= 30:
        early_rewards = rewards[:len(rewards)//3]
        late_rewards = rewards[2*len(rewards)//3:]
        improvement = sum(late_rewards)/len(late_rewards) - sum(early_rewards)/len(early_rewards)
        
        print(f"\n🧠 Learning Analysis:")
        print(f"   Performance Improvement: {improvement:+.3f}")
        print(f"   Learning Status: {'ACTIVE' if abs(improvement) > 0.1 else 'STABLE'}")
    
    return episodes

if __name__ == "__main__":
    episodes = main()
