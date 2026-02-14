"""
Generate RetNet Backbone Architecture Diagram for Model 4
=========================================================

Visual representation of the ConsensusRetNet architecture with:
- Input layer and projection
- 3x RetNet blocks with multi-scale retention
- Classification head
- Parameter counts
- Performance metrics

Author: NOK KO
Date: 2026-01-28
Version: 1.0
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Figure setup
fig, ax = plt.subplots(figsize=(18, 24))
ax.set_xlim(0, 10)
ax.set_ylim(0, 30)
ax.axis('off')

# Colors
color_input = '#E8F4F8'
color_projection = '#B8E6F0'
color_retnet = '#FFE5B4'
color_norm = '#D4E6F1'
color_retention = '#FFD1DC'
color_ffn = '#E0BBE4'
color_output = '#C7CEEA'
color_classifier = '#B5D4B5'

def draw_box(x, y, w, h, text, color, fontsize=10, fontweight='normal'):
    """Draw a colored box with text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)

def draw_arrow(x1, y1, x2, y2, style='->', color='black', width=2):
    """Draw an arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, linewidth=width,
                           mutation_scale=20)
    ax.add_patch(arrow)

# Title
ax.text(5, 29, 'Model 4: ConsensusRetNet Architecture', 
        ha='center', fontsize=20, fontweight='bold')
ax.text(5, 28.3, 'Multi-Scale Retention Network for Consensus Mechanism Selection',
        ha='center', fontsize=14, style='italic')

# Input
y = 26
draw_box(3.5, y, 3, 0.8, 'Input Features\n12 dimensions\n(network, security, resource)', 
         color_input, fontsize=11, fontweight='bold')

# Arrow
draw_arrow(5, y, 5, y-0.8)

# Input Projection
y = 24
draw_box(3, y, 4, 1, 'Input Projection\nLinear(12 → 384)\n+ Positional Embedding\nParams: 4,992 + 384', 
         color_projection, fontsize=10, fontweight='bold')

# Arrow
draw_arrow(5, y, 5, y-0.8)

# ===== RetNet Block 1 =====
y = 21.5
# Block container
draw_box(1.5, y-4.3, 7, 4.5, '', '#F5F5F5', fontsize=10)
ax.text(5, y+0.4, 'RetNet Block 1', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))

# Layer Norm 1
draw_box(2.5, y-0.5, 5, 0.6, 'LayerNorm (d=384)\nParams: 768', color_norm, fontsize=9)
draw_arrow(5, y+0.4, 5, y-0.5)

# Multi-Scale Retention
y_ret = y-1.8
draw_box(2, y_ret, 6, 1.5, '', color_retention, fontsize=9)
ax.text(5, y_ret+1.3, 'Multi-Scale Retention (3 heads)', ha='center', 
        fontsize=10, fontweight='bold')

# Retention details (3 columns)
ret_details = [
    ('Head 1\nγ=0.9\nShort-term', 2.2, y_ret+0.5),
    ('Head 2\nγ=0.95\nMedium-term', 3.7, y_ret+0.5),
    ('Head 3\nγ=0.99\nLong-term', 5.2, y_ret+0.5)
]
for text, x, y_pos in ret_details:
    draw_box(x, y_pos-0.5, 1.3, 0.8, text, '#FFF0F5', fontsize=8, fontweight='bold')

ax.text(5, y_ret+0.05, 'Q, K, V Projections + Retention Weights + GroupNorm\nParams: ~1,771,032',
        ha='center', fontsize=8, style='italic')
draw_arrow(5, y-0.5, 5, y_ret+1.5)

# Residual connection
draw_arrow(1.5, y-0.5, 1.5, y_ret-0.5, style='-', color='blue', width=1.5)
draw_arrow(1.5, y_ret-0.5, 2, y_ret-0.5, style='->', color='blue', width=1.5)
ax.text(1.2, y-2, 'Skip', ha='center', fontsize=8, color='blue', fontweight='bold')

# Layer Norm 2
y_ln2 = y_ret-1
draw_box(2.5, y_ln2, 5, 0.6, 'LayerNorm (d=384)\nParams: 768', color_norm, fontsize=9)
draw_arrow(5, y_ret, 5, y_ln2)

# Feed-Forward Network
y_ffn = y_ln2-1.2
draw_box(2, y_ffn, 6, 1, 'Feed-Forward Network\nLinear(384 → 1536) + GELU + Dropout\nLinear(1536 → 384)\nParams: ~1,180,032', 
         color_ffn, fontsize=9, fontweight='bold')
draw_arrow(5, y_ln2, 5, y_ffn+1)

# Residual connection 2
draw_arrow(1.5, y_ln2, 1.5, y_ffn+0.2, style='-', color='blue', width=1.5)
draw_arrow(1.5, y_ffn+0.2, 2, y_ffn+0.5, style='->', color='blue', width=1.5)

draw_arrow(5, y_ffn, 5, y_ffn-0.8)

# ===== RetNet Block 2 (simplified) =====
y2 = 15.5
draw_box(1.5, y2-1.5, 7, 1.8, '', '#F5F5F5', fontsize=10)
ax.text(5, y2-0.2, 'RetNet Block 2', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
ax.text(5, y2-0.7, 'LayerNorm → Multi-Scale Retention → LayerNorm → FFN',
        ha='center', fontsize=9, style='italic')
ax.text(5, y2-1.1, '(Same structure as Block 1)', ha='center', fontsize=9, style='italic')

draw_arrow(5, 17.3, 5, y2+0.1)
draw_arrow(5, y2-1.5, 5, y2-2.3)

# ===== RetNet Block 3 (simplified) =====
y3 = 12
draw_box(1.5, y3-1.5, 7, 1.8, '', '#F5F5F5', fontsize=10)
ax.text(5, y3-0.2, 'RetNet Block 3', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
ax.text(5, y3-0.7, 'LayerNorm → Multi-Scale Retention → LayerNorm → FFN',
        ha='center', fontsize=9, style='italic')
ax.text(5, y3-1.1, '(Same structure as Block 1)', ha='center', fontsize=9, style='italic')

draw_arrow(5, y3-1.5, 5, y3-2.3)

# Final LayerNorm
y_final = 8.5
draw_box(2.5, y_final, 5, 0.7, 'Final LayerNorm (d=384)\nParams: 768', 
         color_norm, fontsize=10, fontweight='bold')
draw_arrow(5, 9.5, 5, y_final+0.7)

# Global Pooling
y_pool = 7.3
draw_box(3, y_pool, 4, 0.7, 'Squeeze Sequence Dim\n(B, 1, 384) → (B, 384)', 
         color_output, fontsize=9, fontweight='bold')
draw_arrow(5, y_final, 5, y_pool+0.7)

# Classification Head
y_class = 5
draw_box(2, y_class-1.8, 6, 2, '', color_classifier, fontsize=9)
ax.text(5, y_class-0.3, 'Classification Head', ha='center', fontsize=11, fontweight='bold')
ax.text(5, y_class-0.7, 'Linear(384 → 192) + GELU + Dropout', ha='center', fontsize=9)
ax.text(5, y_class-1.1, 'Linear(192 → 5)', ha='center', fontsize=9)
ax.text(5, y_class-1.5, 'Params: 74,885', ha='center', fontsize=9, fontweight='bold')
draw_arrow(5, y_pool, 5, y_class)

# Output
y_out = 2.2
draw_box(3, y_out, 4, 0.8, 'Output Logits\n5 classes (PoW, PoS, PBFT, DPoS, Hybrid)', 
         color_input, fontsize=10, fontweight='bold')
draw_arrow(5, y_class-1.8, 5, y_out+0.8)

# ===== Legend and Stats =====
# Architecture specs box
specs_text = """Architecture Specifications:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Model: ConsensusRetNet
• d_model: 384
• d_ff: 1536 (4× expansion)
• n_layers: 3 (RetNet blocks)
• n_heads: 3 (retention heads)
• Retention scales: γ={0.9, 0.95, 0.99}
• Dropout: 0.1
• Device: MPS (Apple Silicon GPU)
• Total Parameters: 5,402,126
• Trainable: 5,402,117"""

ax.text(9.7, 23, specs_text, ha='left', fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFF9E6', edgecolor='black', linewidth=2),
        verticalalignment='top')

# Performance box
perf_text = """Performance Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target: >96.9% accuracy

✓ Val Acc:  99.98%
✓ Test Acc: 99.98%

Per-Class Accuracy:
  PoW:    100.00%
  PoS:     99.22%
  PBFT:   100.00%
  DPoS:   100.00%
  Hybrid: 100.00%

Training: ~3-4 minutes
Data: 30K balanced samples"""

ax.text(9.7, 13.5, perf_text, ha='left', fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E8F8E8', edgecolor='green', linewidth=2),
        verticalalignment='top')

# Key features box
features_text = """Key Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Multi-scale retention
✓ Parallel training
✓ O(1) recurrent inference
✓ Balanced dataset
✓ Label smoothing
✓ 3-stage LR schedule
✓ Early stopping"""

ax.text(9.7, 5.5, features_text, ha='left', fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F0E6FF', edgecolor='purple', linewidth=2),
        verticalalignment='top')

# Footer
ax.text(5, 0.5, 'Model 4 - Consensus Mechanism Selection | RetNet Architecture | NOK KO © 2026',
        ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()

# Save
output_path = '/Volumes/Shared U/SCS Python Simulation/Paper_Models_Suite/Model_4_Consensus_RetNet/MODEL_BACKBONE_ARCHITECTURE.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved backbone diagram to: {output_path}")

plt.close()
print("🎉 Backbone architecture diagram generated successfully!")
