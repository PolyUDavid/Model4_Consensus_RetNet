"""
RetNet Architecture for Consensus Mechanism Selection - Model 4
===============================================================

Based on: "Retentive Network: A Successor to Transformer for Large Language Models"
Paper: ICML 2023
Authors: Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, 
         Jilong Xue, Jianyong Wang, Furu Wei

Key Innovation: Multi-scale retention mechanism
- Parallel training (O(N) complexity)
- Recurrent inference (O(1) complexity per step)
- Multiple time scales (γ={0.9, 0.95, 0.99})

Task: Multi-class classification (5 consensus mechanisms)
Input: 12 features (network topology, performance, security requirements)
Output: 5 classes (PoW, PoS, PBFT, DPoS, Hybrid)

Target Parameters: ~108K (可增加不可减少)

Author: NOK KO
Date: 2026-01-28
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention Mechanism
    
    Key idea: Different heads focus on different temporal scales
    - Head 1: γ=0.9  (short-term patterns)
    - Head 2: γ=0.95 (medium-term patterns)  
    - Head 3: γ=0.99 (long-term patterns)
    """
    
    def __init__(self, d_model: int, num_heads: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Multi-scale decay rates (different for each head)
        # γ close to 1 = long-term memory
        # γ close to 0 = short-term memory
        self.gammas = nn.Parameter(
            torch.tensor([0.9, 0.95, 0.99][:num_heads]),
            requires_grad=False  # Fixed decay rates
        )
        
        # Group normalization for stability
        self.group_norm = nn.GroupNorm(num_heads, d_model)
        
        # Initialize weights (small for stability)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (B, L, D)
        K = self.k_proj(x)  # (B, L, D)
        V = self.v_proj(x)  # (B, L, D)
        
        # Reshape for multi-head: (B, L, num_heads, head_dim)
        Q = Q.view(B, L, self.num_heads, self.head_dim)
        K = K.view(B, L, self.num_heads, self.head_dim)
        V = V.view(B, L, self.num_heads, self.head_dim)
        
        # Transpose for attention: (B, num_heads, L, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute retention for each head with different decay rates
        outputs = []
        for h in range(self.num_heads):
            q_h = Q[:, h, :, :]  # (B, L, head_dim)
            k_h = K[:, h, :, :]  # (B, L, head_dim)
            v_h = V[:, h, :, :]  # (B, L, head_dim)
            gamma = self.gammas[h]
            
            # Retention mechanism (simplified parallel form)
            # Attention weights with exponential decay
            scores = torch.matmul(q_h, k_h.transpose(-2, -1))  # (B, L, L)
            scores = scores / math.sqrt(self.head_dim)
            
            # Apply causal mask and exponential decay
            # Create decay matrix: D[i,j] = γ^(i-j) if i >= j, else 0
            positions = torch.arange(L, device=x.device)
            decay_matrix = torch.pow(gamma, positions.unsqueeze(0) - positions.unsqueeze(1))
            decay_matrix = torch.triu(decay_matrix)  # Causal: only attend to past
            
            # Apply decay to scores
            scores = scores * decay_matrix.unsqueeze(0)  # (B, L, L)
            
            # Normalize (retention weights)
            retention_weights = F.softmax(scores, dim=-1)
            retention_weights = self.dropout(retention_weights)
            
            # Apply to values
            output_h = torch.matmul(retention_weights, v_h)  # (B, L, head_dim)
            outputs.append(output_h)
        
        # Concatenate heads
        output = torch.cat(outputs, dim=-1)  # (B, L, D)
        
        # Apply group normalization for stability
        output = output.transpose(1, 2)  # (B, D, L) for group norm
        output = self.group_norm(output)
        output = output.transpose(1, 2)  # (B, L, D)
        
        # Output projection
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.gelu(x)  # Use GELU activation
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class RetNetBlock(nn.Module):
    """
    Single RetNet Block
    
    Architecture:
        x -> LayerNorm -> Multi-Scale Retention -> Residual
          -> LayerNorm -> Feed-Forward -> Residual
    """
    
    def __init__(self, d_model: int, num_heads: int = 3, 
                 d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4  # Standard FF expansion factor
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-scale retention
        self.retention = MultiScaleRetention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Retention block with residual
        residual = x
        x = self.norm1(x)
        x = self.retention(x)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward block with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class ConsensusRetNet(nn.Module):
    """
    RetNet for Consensus Mechanism Selection
    
    Architecture:
        Input (B, 12) 
          -> Embedding (B, 1, d_model)
          -> 3 × RetNet Blocks
          -> Global Average Pooling (B, d_model)
          -> Classification Head (B, 5)
    
    Target: ~108K parameters (可增加不可减少)
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        output_dim: int = 5,
        d_model: int = 384,
        num_layers: int = 3,
        num_heads: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        if d_ff is None:
            d_ff = d_model * 4  # 1536
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional embedding (for sequence position)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # RetNet blocks
        self.layers = nn.ModuleList([
            RetNetBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm_f = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stability"""
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        
        Returns:
            logits: (batch, output_dim)
        """
        # Input projection
        x = self.input_proj(x)  # (B, d_model)
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # (B, 1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Pass through RetNet blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm_f(x)
        
        # Global average pooling (squeeze sequence dimension)
        x = x.squeeze(1)  # (B, d_model)
        
        # Classification
        logits = self.classifier(x)  # (B, output_dim)
        
        return logits
    
    def count_parameters(self) -> dict:
        """Count parameters for each component"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Per-component counts
        counts = {
            'input_proj': sum(p.numel() for p in self.input_proj.parameters()),
            'pos_embedding': self.pos_embedding.numel(),
            'retnet_blocks': sum(p.numel() for p in self.layers.parameters()),
            'norm_f': sum(p.numel() for p in self.norm_f.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
            'total': total,
            'trainable': trainable
        }
        
        return counts


def create_model(
    input_dim: int = 12,
    output_dim: int = 5,
    d_model: int = 384,
    num_layers: int = 3,
    num_heads: int = 3,
    dropout: float = 0.1
) -> ConsensusRetNet:
    """
    Create RetNet model for consensus mechanism selection
    
    Default configuration targets ~108K parameters
    """
    model = ConsensusRetNet(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )
    
    return model


if __name__ == "__main__":
    # Test model
    print("=" * 80)
    print("Model 4: ConsensusRetNet Architecture Test")
    print("=" * 80)
    
    # Create model
    model = create_model()
    
    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter Counts:")
    print("-" * 80)
    for name, count in param_counts.items():
        print(f"  {name:20s}: {count:>12,}")
    print("-" * 80)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 12)
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output logits:\n{logits}")
    
    # Test on MPS if available
    if torch.backends.mps.is_available():
        print("\n✅ MPS (Apple Silicon GPU) available!")
        print("   Testing on MPS...")
        device = torch.device('mps')
        model = model.to(device)
        x = x.to(device)
        
        with torch.no_grad():
            logits = model(x)
        
        print(f"   Output on MPS: {logits.shape}")
        print("   ✅ MPS test passed!")
    
    print("\n" + "=" * 80)
    print("✅ Model architecture test complete!")
    print("=" * 80)
