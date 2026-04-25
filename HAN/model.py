"""
HAN Model Architectures
Contains HAN++ and HGT-HAN hybrid models for medical predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import (NodeLevelAttentionImproved, SemanticAttentionImproved,
                   PatientConditionedSemanticAttention, HGTLayerSingle)


class HANPP(nn.Module):
    """
    HAN++ Model (Version B)
    
    Improved Hierarchical Attention Network with:
    - Multi-head node-level attention per meta-path
    - Semantic-level attention across meta-paths
    - Multi-organ severity classification
    - Organ damage score regression
    
    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension
        out_dim: output embedding dimension
        metapath_names: list of meta-path names to use
        num_heads: number of attention heads
        num_organs: number of organs to predict
        num_severity: number of severity classes
        dropout: dropout rate
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, metapath_names,
                 num_heads=4, num_organs=25, num_severity=4, dropout=0.3):
        super().__init__()
        self.metapath_names = metapath_names

        # Input projection
        self.project = nn.Linear(in_dim, hidden_dim)

        # Node-level attention for each meta-path
        self.node_atts = nn.ModuleList([
            NodeLevelAttentionImproved(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in metapath_names
        ])

        # Patient-Conditioned Semantic Attention (novel contribution):
        # replaces HAN's global query vector with a patient-specific query
        # q_i = W_q * h_i, so each patient conditions its own meta-path weights.
        self.semantic_att = PatientConditionedSemanticAttention(hidden_dim, dropout=dropout)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        # Organ-specific classifiers (one per organ)
        self.organ_classifiers = nn.ModuleList([
            nn.Linear(out_dim, num_severity) for _ in range(num_organs)
        ])
        
        # Organ damage regression head
        self.organ_regression = nn.Linear(out_dim, num_organs)
        
        self.dropout = nn.Dropout(dropout)
    
    def set_vectorized_neighbors(self, neighbor_tensors):
        """
        Pre-set vectorized neighbor tensors for all meta-paths.
        
        Args:
            neighbor_tensors: dict of {metapath_name: (neighbor_idx, neighbor_mask)}
        """
        for i, name in enumerate(self.metapath_names):
            if name in neighbor_tensors:
                idx, mask = neighbor_tensors[name]
                self.node_atts[i].set_neighbors(idx, mask)
    
    def forward(self, patient_feats, patient_neighbor_dicts):
        """
        Forward pass.

        Args:
            patient_feats: patient feature tensor [N, in_dim]
            patient_neighbor_dicts: dict of {metapath_name: neighbor_dict}

        Returns:
            organ_logits: [N, num_organs, num_severity] classification logits
            organ_scores: [N, num_organs] regression scores
            z: [N, out_dim] final embeddings
            beta: [N, num_metapaths] per-patient attention weights over meta-paths
        """
        # Project to hidden dimension
        h = F.gelu(self.project(patient_feats))  # [N, hidden_dim]

        # Apply node-level attention for each meta-path
        Zs = []
        for i, name in enumerate(self.metapath_names):
            neigh = patient_neighbor_dicts[name]
            Z_phi = self.node_atts[i](h, neigh)
            Zs.append(Z_phi)

        # Patient-Conditioned Semantic Attention:
        # each patient's own embedding h_i conditions its meta-path weights
        Z_final, beta = self.semantic_att(Zs, h_patient=h)  # beta: [N, K]

        # Final output projection
        z = F.gelu(self.out_proj(Z_final))

        # Organ-specific predictions
        organ_logits = [clf(self.dropout(z)) for clf in self.organ_classifiers]
        organ_logits = torch.stack(organ_logits, dim=1)  # [N, num_organs, num_severity]

        # Organ damage scores
        organ_scores = torch.sigmoid(self.organ_regression(z))  # [N, num_organs]

        return organ_logits, organ_scores, z, beta


class HANPP_Disease(nn.Module):
    """
    HAN++ adapted for binary multi-label disease classification.

    Identical structure to HANPP but the output head is a single
    nn.Linear(out_dim, num_diseases) instead of per-organ classifiers.
    Returns logits [N, num_diseases] for BCEWithLogitsLoss.

    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension
        out_dim: output embedding dimension
        metapath_names: list of meta-path names to use
        num_heads: number of attention heads
        num_diseases: number of binary disease labels to predict
        dropout: dropout rate
    """

    def __init__(self, in_dim, hidden_dim, out_dim, metapath_names,
                 num_heads=4, num_diseases=5, dropout=0.3,
                 use_global_query=False):
        super().__init__()
        self.metapath_names = metapath_names

        self.project = nn.Linear(in_dim, hidden_dim)

        self.node_atts = nn.ModuleList([
            NodeLevelAttentionImproved(hidden_dim, hidden_dim,
                                       num_heads=num_heads, dropout=dropout)
            for _ in metapath_names
        ])

        self.semantic_att = PatientConditionedSemanticAttention(
            hidden_dim, dropout=dropout, use_global_query=use_global_query
        )

        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.disease_classifier = nn.Linear(out_dim, num_diseases)
        self.dropout = nn.Dropout(dropout)

    def set_vectorized_neighbors(self, neighbor_tensors):
        """Pre-set padded neighbor tensors for vectorized attention."""
        for i, name in enumerate(self.metapath_names):
            if name in neighbor_tensors:
                idx, mask = neighbor_tensors[name]
                self.node_atts[i].set_neighbors(idx, mask)

    def forward(self, patient_feats, patient_neighbor_dicts):
        """
        Forward pass.

        Args:
            patient_feats: [N, in_dim]
            patient_neighbor_dicts: dict of {metapath_name: neighbor_dict}

        Returns:
            disease_logits: [N, num_diseases]  (raw logits for BCEWithLogitsLoss)
            z: [N, out_dim]  final embeddings
            beta: [N, num_metapaths]  per-patient meta-path weights
        """
        h = F.gelu(self.project(patient_feats))

        Zs = []
        for i, name in enumerate(self.metapath_names):
            neigh = patient_neighbor_dicts[name]
            Zs.append(self.node_atts[i](h, neigh))

        Z_final, beta = self.semantic_att(Zs, h_patient=h)
        z = F.gelu(self.out_proj(Z_final))

        disease_logits = self.disease_classifier(self.dropout(z))
        return disease_logits, z, beta


# ── Link Prediction Components ────────────────────────────────────────────────

class DiseaseEncoder(nn.Module):
    """
    Encodes disease node feature vectors into the same embedding space as patients.

    Input: disease_feat [D, S]  — binary test-association vector per disease.
    Output: h_D [D, hidden_dim] — disease embeddings compatible with patient z.

    This is the key enabler for zero-shot new disease prediction: a new disease
    only needs its test associations defined; no model retraining is required.

    Args:
        in_dim:     size of disease feature vector (= number of symptoms S)
        hidden_dim: target embedding dimension (must equal HANPP_Disease out_dim)
        dropout:    dropout rate
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, disease_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disease_feats: [D, in_dim]
        Returns:
            h_D: [D, hidden_dim]
        """
        return self.net(disease_feats)


class LinkPredDecoder(nn.Module):
    """
    DistMult link prediction decoder for Patient-Disease edges.

    Scores a candidate (Patient, Disease) edge as:
        score(P, D) = (h_P ⊙ R ⊙ h_D).sum()

    where R is a learnable per-dimension relation vector. This is the standard
    DistMult factorisation (Yang et al., 2015), which is efficient and effective
    for heterogeneous graphs with a single relation type.

    For the all-pairs matrix: scores = (h_P * R) @ h_D.T  →  [N, D]

    Args:
        hidden_dim: embedding dimension (must match DiseaseEncoder and HANPP_Disease)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Relation embedding initialised to ones so that early training
        # approximates a plain dot-product (stable starting point)
        self.relation = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, h_patient: torch.Tensor,
                h_disease: torch.Tensor) -> torch.Tensor:
        """
        Compute all-pairs link scores.

        Args:
            h_patient: [N, hidden_dim]  patient embeddings
            h_disease: [D, hidden_dim]  disease embeddings
        Returns:
            scores: [N, D]  unnormalised logits (apply sigmoid for probability)
        """
        p_weighted = h_patient * self.relation   # [N, H]
        return p_weighted @ h_disease.t()        # [N, D]

    def score_pair(self, h_patient: torch.Tensor,
                   h_disease: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of (patient, disease) pairs element-wise.

        Args:
            h_patient: [B, hidden_dim]
            h_disease: [B, hidden_dim]   (one disease per patient in the batch)
        Returns:
            scores: [B]  unnormalised logits
        """
        return (h_patient * self.relation * h_disease).sum(dim=-1)


class HGT_HAN(nn.Module):
    """
    HGT-HAN Hybrid Model (Version C)
    
    Combines HGT-style attention with HAN's hierarchical structure:
    - HGT-style multi-head attention per meta-path
    - Semantic-level attention across meta-paths  
    - Multi-organ severity classification
    - Organ damage score regression
    
    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension
        out_dim: output embedding dimension
        metapath_names: list of meta-path names to use
        num_heads: number of attention heads
        num_organs: number of organs to predict
        num_severity: number of severity classes
        dropout: dropout rate
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, metapath_names,
                 num_heads=4, num_organs=25, num_severity=4, dropout=0.3):
        super().__init__()
        self.metapath_names = metapath_names
        
        # Input projection
        self.project = nn.Linear(in_dim, hidden_dim)
        
        # HGT-style attention layers for each meta-path
        self.hgt_layers = nn.ModuleList([
            HGTLayerSingle(hidden_dim, hidden_dim, nhead=num_heads, dropout=dropout)
            for _ in metapath_names
        ])
        
        # Semantic-level attention to aggregate meta-paths
        self.semantic_att = SemanticAttentionImproved(hidden_dim, dropout=dropout)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        # Organ-specific classifiers (one per organ)
        self.organ_classifiers = nn.ModuleList([
            nn.Linear(out_dim, num_severity) for _ in range(num_organs)
        ])
        
        # Organ damage regression head
        self.organ_regression = nn.Linear(out_dim, num_organs)
        
        self.dropout = nn.Dropout(dropout)
    
    def set_vectorized_neighbors(self, neighbor_tensors):
        """
        Pre-set vectorized neighbor tensors for all meta-paths.
        
        Args:
            neighbor_tensors: dict of {metapath_name: (neighbor_idx, neighbor_mask)}
        """
        for i, name in enumerate(self.metapath_names):
            if name in neighbor_tensors:
                idx, mask = neighbor_tensors[name]
                self.hgt_layers[i].set_neighbors(idx, mask)
    
    def forward(self, patient_feats, patient_neighbor_dicts):
        """
        Forward pass.
        
        Args:
            patient_feats: patient feature tensor [N, in_dim]
            patient_neighbor_dicts: dict of {metapath_name: neighbor_dict}
        
        Returns:
            organ_logits: [N, num_organs, num_severity] classification logits
            organ_scores: [N, num_organs] regression scores
            z: [N, out_dim] final embeddings
            beta: [num_metapaths] attention weights over meta-paths
        """
        # Project to hidden dimension
        h = F.gelu(self.project(patient_feats))
        
        # Apply HGT-style attention for each meta-path
        Zs = []
        for i, name in enumerate(self.metapath_names):
            neigh = patient_neighbor_dicts[name]
            Z_phi = self.hgt_layers[i](h, neigh)
            Zs.append(Z_phi)
        
        # Aggregate meta-paths with semantic attention
        Z_final, beta = self.semantic_att(Zs)
        
        # Final output projection
        z = F.gelu(self.out_proj(Z_final))
        
        # Organ-specific predictions
        organ_logits = [clf(self.dropout(z)) for clf in self.organ_classifiers]
        organ_logits = torch.stack(organ_logits, dim=1)  # [N, num_organs, num_severity]
        
        # Organ damage scores
        organ_scores = torch.sigmoid(self.organ_regression(z))  # [N, num_organs]
        
        return organ_logits, organ_scores, z, beta
