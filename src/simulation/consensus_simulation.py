#!/usr/bin/env python3
"""
Model 4: ConsensusRetNet - Blockchain Vehicular Network Simulation
==================================================================

Real-time simulation of a vehicular blockchain network where the
trained RetNet model selects optimal consensus mechanisms under
dynamically changing network conditions.

UI Layout: Adapted from Model 1 (NexusPredict 2D Animation)

Scenarios:
  1. Normal V2X Network  - Moderate nodes, stable conditions
  2. Byzantine Attack     - High attack risk, nodes going rogue
  3. Emergency Consensus  - Ultra-low latency requirement, mass event

Controls:
  SPACE  - Switch scenario
  ESC    - Quit
  1/2/3  - Jump to scenario
  P      - Pause/Resume

Author: NOK KO
Date: 2026-02-05
"""

import pygame
import sys
import math
import random
import time
import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Add parent directories for model import
SRC_DIR = Path(__file__).parent.parent       # reaches src/
BASE_DIR = SRC_DIR.parent                    # reaches GIT_MODEL4/
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / 'models'))

try:
    import torch
    from models.retnet_consensus import create_model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("WARNING: PyTorch or model not available. Running in demo mode.")

# Try PIL for text rendering (same approach as Model 1)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Initialize Pygame
pygame.init()
pygame.display.init()

# Window config
WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 1000
FPS = 60

# Color palette (blockchain-themed, dark mode)
COLORS = {
    'bg': (10, 15, 30),
    'road': (25, 32, 50),
    'road_line': (55, 65, 90),
    'grass': (12, 40, 28),
    'panel_bg': (15, 25, 55),
    'panel_border': (40, 80, 180),
    'panel_highlight': (60, 120, 220),
    # Node types
    'node_validator': (46, 204, 113),    # Green - validator
    'node_miner': (241, 196, 15),        # Gold - miner
    'node_delegate': (52, 152, 219),     # Blue - delegate
    'node_byzantine': (231, 76, 60),     # Red - byzantine
    'node_normal': (149, 165, 166),      # Gray - normal
    # Consensus mechanism colors
    'pow_color': (241, 196, 15),         # Gold
    'pos_color': (46, 204, 113),         # Green
    'pbft_color': (155, 89, 182),        # Purple
    'dpos_color': (52, 152, 219),        # Blue
    'hybrid_color': (230, 126, 34),      # Orange
    # Effects
    'chain_link': (100, 140, 200),
    'transaction': (0, 255, 200),
    'attack_glow': (255, 50, 50, 120),
    'signal_wave': (100, 180, 255),
    # UI
    'text_white': (255, 255, 255),
    'text_dim': (160, 170, 190),
    'text_success': (46, 204, 113),
    'text_warning': (241, 196, 15),
    'text_danger': (231, 76, 60),
}

# Consensus mechanism info
CONSENSUS_INFO = {
    'PoW': {'color': COLORS['pow_color'], 'icon': 'M', 'name': 'Proof of Work'},
    'PoS': {'color': COLORS['pos_color'], 'icon': 'S', 'name': 'Proof of Stake'},
    'PBFT': {'color': COLORS['pbft_color'], 'icon': 'B', 'name': 'PBFT'},
    'DPoS': {'color': COLORS['dpos_color'], 'icon': 'D', 'name': 'Delegated PoS'},
    'Hybrid': {'color': COLORS['hybrid_color'], 'icon': 'H', 'name': 'Hybrid'},
}
CONSENSUS_CLASSES = ['PoW', 'PoS', 'PBFT', 'DPoS', 'Hybrid']

# Layout — top panel is 280px, road starts below it
PANEL_HEIGHT = 280
ROAD_Y = PANEL_HEIGHT + 60   # 340 — enough gap below panel
ROAD_HEIGHT = 200
LANE_WIDTH = 60
NUM_LANES = 3


# ==============================================================================
# Font system: PIL with TrueType fonts — crisp rendering at native resolution
# ==============================================================================
_pil_font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

def _get_pil_font(size: int):
    """Get a PIL TrueType font at exact pixel size (no scaling needed)."""
    if size not in _pil_font_cache:
        # Try system TrueType fonts (macOS paths)
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Geneva.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        for fp in font_paths:
            try:
                _pil_font_cache[size] = ImageFont.truetype(fp, size)
                break
            except (OSError, IOError):
                continue
        if size not in _pil_font_cache:
            # Last resort: default bitmap (will still be cleaner than before because
            # we render at 2x and down-sample with LANCZOS, giving basic anti-aliasing)
            _pil_font_cache[size] = ImageFont.load_default()
    return _pil_font_cache[size]


def render_text(text: str, size: int, color: Tuple[int, int, int]) -> pygame.Surface:
    """Render crisp text with PIL TrueType — no blurry scaling."""
    if not PIL_AVAILABLE:
        surf = pygame.Surface((max(len(text) * size // 2, 1), max(size, 1)), pygame.SRCALPHA)
        return surf
    try:
        font = _get_pil_font(size)
        
        # Measure text at native size
        dummy = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        
        # Render at native size — no rescaling needed
        pad_x, pad_y = 4, 2
        img = Image.new('RGBA', (tw + pad_x * 2, th + pad_y * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(*color, 255))
        
        raw = img.tobytes()
        return pygame.image.fromstring(raw, img.size, 'RGBA')
    except Exception:
        surf = pygame.Surface((max(len(text) * size // 2, 1), max(size, 1)), pygame.SRCALPHA)
        return surf


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class BlockchainNode:
    """A node in the vehicular blockchain network"""
    x: float
    y: float
    lane: int
    speed: float
    node_type: str          # 'validator', 'miner', 'delegate', 'byzantine', 'normal'
    node_id: int = 0
    stake: float = 0.0      # For PoS
    compute_power: float = 0.0  # For PoW
    reputation: float = 1.0
    is_byzantine: bool = False
    pulse_phase: float = 0.0
    target_speed: float = 0.0
    width: float = 45
    height: float = 28
    
    def update(self, dt, traffic_lights, nodes):
        self.pulse_phase += dt * 3.0
        
        # Traffic light response
        for light in traffic_lights:
            if light.x - 120 < self.x < light.x + 50 and light.state == 'red':
                self.speed = max(0, self.speed - 120 * dt)
                return
        
        # Collision avoidance
        for n in nodes:
            if n != self and n.lane == self.lane and 0 < n.x - self.x < 90:
                if n.speed < self.speed:
                    self.speed = max(n.speed, self.speed - 100 * dt)
                    return
        
        # Speed adjustment
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + 70 * dt)
        elif self.speed > self.target_speed:
            self.speed = max(self.target_speed, self.speed - 50 * dt)
        
        self.x += self.speed * dt
    
    def draw(self, screen, camera_x):
        draw_x = self.x - camera_x
        if -150 < draw_x < WINDOW_WIDTH + 150:
            # Glow effect for special nodes
            if self.node_type != 'normal':
                type_colors = {
                    'validator': COLORS['node_validator'],
                    'miner': COLORS['node_miner'],
                    'delegate': COLORS['node_delegate'],
                    'byzantine': COLORS['node_byzantine'],
                }
                glow_color = type_colors.get(self.node_type, COLORS['node_normal'])
                glow_alpha = int(60 + 40 * math.sin(self.pulse_phase))
                glow_surf = pygame.Surface((self.width + 30, self.height + 30), pygame.SRCALPHA)
                pygame.draw.ellipse(glow_surf, (*glow_color, glow_alpha), glow_surf.get_rect())
                screen.blit(glow_surf, (draw_x - 15, self.y - 15))
            
            # Vehicle body
            color = {
                'validator': COLORS['node_validator'],
                'miner': COLORS['node_miner'],
                'delegate': COLORS['node_delegate'],
                'byzantine': COLORS['node_byzantine'],
                'normal': COLORS['node_normal'],
            }.get(self.node_type, COLORS['node_normal'])
            
            rect = pygame.Rect(draw_x, self.y, self.width, self.height)
            pygame.draw.rect(screen, color, rect, 0, border_radius=6)
            pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=6)
            
            # Windshield
            wind = pygame.Rect(draw_x + self.width - 16, self.y + 6, 12, self.height - 12)
            pygame.draw.rect(screen, (80, 120, 200), wind, 0, border_radius=3)
            
            # Byzantine attack indicator
            if self.is_byzantine:
                danger_surf = pygame.Surface((self.width + 20, self.height + 20), pygame.SRCALPHA)
                alpha = int(80 + 60 * math.sin(self.pulse_phase * 2))
                pygame.draw.ellipse(danger_surf, (255, 0, 0, alpha), danger_surf.get_rect())
                screen.blit(danger_surf, (draw_x - 10, self.y - 10))
                # X marker
                pygame.draw.line(screen, (255, 0, 0), (draw_x + 5, self.y + 5), 
                               (draw_x + self.width - 5, self.y + self.height - 5), 3)
                pygame.draw.line(screen, (255, 0, 0), (draw_x + self.width - 5, self.y + 5),
                               (draw_x + 5, self.y + self.height - 5), 3)


@dataclass
class TrafficLight:
    x: float
    state: str
    timer: float = 0
    green_duration: float = 60
    yellow_duration: float = 3
    red_duration: float = 40
    
    def update(self, dt):
        self.timer += dt
        if self.state == 'green' and self.timer >= self.green_duration:
            self.state, self.timer = 'yellow', 0
        elif self.state == 'yellow' and self.timer >= self.yellow_duration:
            self.state, self.timer = 'red', 0
        elif self.state == 'red' and self.timer >= self.red_duration:
            self.state, self.timer = 'green', 0
    
    def draw(self, screen, camera_x):
        draw_x = self.x - camera_x
        if -150 < draw_x < WINDOW_WIDTH + 150:
            # Pole
            pygame.draw.rect(screen, (70, 70, 70), (draw_x - 6, ROAD_Y - 90, 12, 90), 0, border_radius=4)
            # Box
            box = pygame.Rect(draw_x - 22, ROAD_Y - 130, 44, 95)
            pygame.draw.rect(screen, (40, 40, 40), box, 0, border_radius=10)
            pygame.draw.rect(screen, (200, 200, 200), box, 3, border_radius=10)
            # Lights
            light_states = {
                'red': [(239, 68, 68), (60, 20, 20), (15, 60, 15)],
                'yellow': [(60, 20, 20), (251, 191, 36), (15, 60, 15)],
                'green': [(60, 20, 20), (60, 50, 15), (34, 197, 94)],
            }
            colors = light_states.get(self.state, light_states['red'])
            for i, (ly, color) in enumerate(zip([ROAD_Y - 110, ROAD_Y - 80, ROAD_Y - 50], colors)):
                pygame.draw.circle(screen, color, (int(draw_x), int(ly)), 14)
                if color[0] > 100 or color[1] > 100:
                    glow = pygame.Surface((60, 60), pygame.SRCALPHA)
                    pygame.draw.circle(glow, (*color, 100), (30, 30), 30)
                    screen.blit(glow, (draw_x - 30, ly - 30))


@dataclass
class RSU:
    """Roadside Unit / Blockchain Gateway"""
    x: float
    signal_phase: float = 0
    
    def update(self, dt):
        self.signal_phase += dt * 2.5
    
    def draw(self, screen, camera_x):
        draw_x = self.x - camera_x
        if -150 < draw_x < WINDOW_WIDTH + 150:
            # Tower
            tower = pygame.Rect(draw_x - 10, ROAD_Y - 200, 20, 120)
            pygame.draw.rect(screen, (40, 50, 70), tower, 0, border_radius=5)
            pygame.draw.rect(screen, (80, 120, 200), tower, 4, border_radius=5)
            # Antenna
            antenna = [(draw_x, ROAD_Y - 200), (draw_x - 18, ROAD_Y - 235), (draw_x + 18, ROAD_Y - 235)]
            pygame.draw.polygon(screen, (80, 120, 200), antenna)
            # Signal waves
            for i in range(3):
                radius = 35 + i * 25 + int((self.signal_phase % 1) * 25)
                alpha = max(0, int(120 - i * 35 - (self.signal_phase % 1) * 120))
                if alpha > 0:
                    wave_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(wave_surf, (*COLORS['signal_wave'], alpha), (radius, radius), radius, 4)
                    screen.blit(wave_surf, (draw_x - radius, ROAD_Y - 215 - radius))


@dataclass
class BlockTransaction:
    """Visual representation of a blockchain transaction"""
    x: float
    y: float
    target_x: float
    target_y: float
    progress: float = 0.0
    speed: float = 2.0
    color: Tuple[int, int, int] = (0, 255, 200)
    
    def update(self, dt):
        self.progress += self.speed * dt
        return self.progress >= 1.0
    
    def draw(self, screen, camera_x):
        t = self.progress
        cx = self.x + (self.target_x - self.x) * t - camera_x
        cy = self.y + (self.target_y - self.y) * t
        if -50 < cx < WINDOW_WIDTH + 50:
            # Glow
            glow = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*self.color, 100), (10, 10), 10)
            screen.blit(glow, (cx - 10, cy - 10))
            # Core
            pygame.draw.circle(screen, self.color, (int(cx), int(cy)), 4)


@dataclass
class ChainBlock:
    """Visual block in the chain"""
    x: float
    y: float
    block_num: int
    consensus: str
    confirmed: bool = True
    
    def draw(self, screen, x_offset, y_offset):
        dx = self.x + x_offset
        dy = self.y + y_offset
        color = CONSENSUS_INFO.get(self.consensus, {}).get('color', (150, 150, 150))
        # Block rectangle
        rect = pygame.Rect(dx, dy, 35, 25)
        pygame.draw.rect(screen, color, rect, 0, border_radius=4)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=4)


class AttackEffect:
    """Visual effect for Byzantine attack"""
    def __init__(self):
        self.particles = []
    
    def spawn(self, x, y):
        for _ in range(5):
            self.particles.append({
                'x': x, 'y': y,
                'vx': random.uniform(-80, 80),
                'vy': random.uniform(-80, 80),
                'life': random.uniform(0.5, 1.5),
                'age': 0,
            })
    
    def update(self, dt):
        for p in self.particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['age'] += dt
        self.particles = [p for p in self.particles if p['age'] < p['life']]
    
    def draw(self, screen, camera_x):
        for p in self.particles:
            dx = p['x'] - camera_x
            if -50 < dx < WINDOW_WIDTH + 50:
                alpha = int(255 * (1 - p['age'] / p['life']))
                sz = int(6 * (1 - p['age'] / p['life']))
                if sz > 0 and alpha > 0:
                    surf = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (255, 50, 50, alpha), (sz, sz), sz)
                    screen.blit(surf, (dx - sz, p['y'] - sz))


# ==============================================================================
# Main Simulation System
# ==============================================================================

class ConsensusSimulationSystem:
    """Main simulation integrating RetNet model with vehicular network visualization"""
    
    def __init__(self, scenario_idx=0):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Model 4: ConsensusRetNet - Blockchain V2X Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Scenario
        self.scenario_idx = scenario_idx
        self.scenario_names = [
            'NORMAL V2X NETWORK',
            'BYZANTINE ATTACK',
            'EMERGENCY CONSENSUS'
        ]
        self.scenario_descriptions = [
            'Stable vehicular network with moderate load',
            'Adversarial nodes attempting to disrupt consensus',
            'Mass event requiring ultra-fast consensus switching'
        ]
        
        # Time
        self.sim_time = 0.0
        self.camera_x = 0.0
        self.world_length = 6000
        
        # Objects
        self.nodes: List[BlockchainNode] = []
        self.traffic_lights: List[TrafficLight] = []
        self.rsus: List[RSU] = []
        self.transactions: List[BlockTransaction] = []
        self.attack_effects = AttackEffect()
        self.chain_blocks: List[ChainBlock] = []
        
        # Network state (12 features for model input)
        self.network_state = {
            'num_nodes': 100,
            'connectivity': 0.85,
            'latency_requirement_sec': 10.0,
            'throughput_requirement_tps': 500,
            'byzantine_tolerance': 0.15,
            'security_priority': 0.8,
            'energy_budget': 0.5,
            'bandwidth_mbps': 1000,
            'consistency_requirement': 0.75,
            'decentralization_requirement': 0.7,
            'network_load': 0.4,
            'attack_risk': 0.3,
        }
        
        # Model prediction
        self.current_consensus = 'PoS'
        self.consensus_confidence = 0.85
        self.consensus_probs = {c: 0.2 for c in CONSENSUS_CLASSES}
        self.prediction_history: List[Dict] = []
        self.blocks_confirmed = 0
        self.tps_current = 0.0
        self.latency_current = 0.0
        
        # Prediction source tracking
        self.prediction_source = "Loading..."
        
        # Transaction generation
        self.tx_timer = 0.0
        self.tx_spawn_rate = 0.3  # seconds between transactions
        
        # Block generation
        self.block_timer = 0.0
        self.block_interval = 2.0  # seconds between blocks
        
        # Model
        self.model = None
        self.feature_stats = None
        self._load_model()
        
        # Text cache
        self.text_cache = {}
        
        # Initialize
        self._init_objects()
    
    def _load_model(self):
        """Load trained RetNet model"""
        if not MODEL_AVAILABLE:
            print("Running in DEMO mode (no PyTorch)")
            return
        
        try:
            model_path = BASE_DIR / 'training' / 'checkpoints' / 'best_consensus.pth'
            if not model_path.exists():
                print(f"Checkpoint not found: {model_path}")
                return
            
            device = 'cpu'  # Use CPU for simulation (lightweight)
            checkpoint = torch.load(str(model_path), weights_only=False, map_location=device)
            
            self.model = create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(device)
            
            self.feature_stats = checkpoint.get('feature_stats', None)
            
            print(f"Model loaded: ConsensusRetNet ({sum(p.numel() for p in self.model.parameters()):,} params)")
            print(f"  Val accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        except Exception as e:
            print(f"Model load error: {e}")
            self.model = None
    
    def _predict_consensus(self):
        """Use RetNet model to predict optimal consensus mechanism.
        Priority: API > Local model > Rule-based fallback
        """
        # Try API first
        if self._predict_via_api():
            self.prediction_source = "API"
            return
        
        # Local model fallback
        if self.model is not None:
            try:
                feature_keys = [
                    'num_nodes', 'connectivity', 'latency_requirement_sec',
                    'throughput_requirement_tps', 'byzantine_tolerance', 'security_priority',
                    'energy_budget', 'bandwidth_mbps', 'consistency_requirement',
                    'decentralization_requirement', 'network_load', 'attack_risk'
                ]
                features = [self.network_state[k] for k in feature_keys]
                x = torch.tensor([features], dtype=torch.float32)
                
                if self.feature_stats is not None:
                    mean = self.feature_stats['mean']
                    std = self.feature_stats['std']
                    if isinstance(mean, torch.Tensor):
                        x = (x - mean.unsqueeze(0)) / (std.unsqueeze(0) + 1e-8)
                    else:
                        mean_t = torch.tensor(mean, dtype=torch.float32)
                        std_t = torch.tensor(std, dtype=torch.float32)
                        x = (x - mean_t.unsqueeze(0)) / (std_t.unsqueeze(0) + 1e-8)
                
                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                
                pred_idx = probs.argmax().item()
                self.current_consensus = CONSENSUS_CLASSES[pred_idx]
                self.consensus_confidence = probs[pred_idx].item()
                self.consensus_probs = {c: probs[i].item() for i, c in enumerate(CONSENSUS_CLASSES)}
                self.prediction_source = "Local Model"
                return
            except Exception:
                pass
        
        # Rule-based fallback
        self._rule_based_prediction()
        self.prediction_source = "Rule-based"
    
    def _predict_via_api(self) -> bool:
        """Try to get prediction from API"""
        try:
            import requests
            resp = requests.post(
                "http://localhost:8000/api/v1/predict",
                json={"network_state": self.network_state},
                timeout=0.5
            )
            if resp.status_code == 200:
                data = resp.json()
                pred = data.get('prediction', {})
                self.current_consensus = pred.get('predicted_consensus', self.current_consensus)
                self.consensus_confidence = pred.get('confidence', self.consensus_confidence)
                self.consensus_probs = pred.get('probabilities', self.consensus_probs)
                return True
        except Exception:
            pass
        return False
    
    def _rule_based_prediction(self):
        """Fallback rule-based consensus selection"""
        s = self.network_state
        if s['security_priority'] > 0.85 and s['energy_budget'] > 0.7:
            self.current_consensus = 'PoW'
            self.consensus_confidence = 0.82
        elif s['num_nodes'] < 100 and s['latency_requirement_sec'] < 5:
            self.current_consensus = 'PBFT'
            self.consensus_confidence = 0.88
        elif s['num_nodes'] > 200 and s['throughput_requirement_tps'] > 2000:
            self.current_consensus = 'DPoS'
            self.consensus_confidence = 0.85
        elif s['energy_budget'] < 0.4:
            self.current_consensus = 'PoS'
            self.consensus_confidence = 0.80
        else:
            self.current_consensus = 'Hybrid'
            self.consensus_confidence = 0.75
        
        self.consensus_probs = {c: 0.05 for c in CONSENSUS_CLASSES}
        self.consensus_probs[self.current_consensus] = self.consensus_confidence
    
    def _init_objects(self):
        """Initialize simulation objects for current scenario"""
        self.nodes.clear()
        self.traffic_lights.clear()
        self.rsus.clear()
        self.transactions.clear()
        self.chain_blocks.clear()
        self.text_cache.clear()
        
        # Traffic lights
        for x in [500, 900, 1300]:
            self.traffic_lights.append(TrafficLight(
                x=x, state=random.choice(['red', 'green']),
                green_duration=50 + random.randint(-8, 8),
                red_duration=38 + random.randint(-8, 8)
            ))
        
        # RSUs
        for x in [400, 800, 1200, 1600]:
            self.rsus.append(RSU(x=x))
        
        # Scenario-specific initialization
        if self.scenario_idx == 0:
            num_nodes = 25
            self._set_network_state_normal()
        elif self.scenario_idx == 1:
            num_nodes = 35
            self._set_network_state_attack()
        else:
            num_nodes = 45
            self._set_network_state_emergency()
        
        for i in range(num_nodes):
            self._spawn_node(node_id=i)
        
        # Initial prediction
        self._predict_consensus()
    
    def _set_network_state_normal(self):
        """Hybrid scenario: balanced requirements -> expects Hybrid consensus"""
        self.network_state = {
            'num_nodes': 150,
            'connectivity': 0.72,
            'latency_requirement_sec': 12.0,
            'throughput_requirement_tps': 1200,
            'byzantine_tolerance': 0.17,
            'security_priority': 0.80,
            'energy_budget': 0.60,
            'bandwidth_mbps': 1500,
            'consistency_requirement': 0.75,
            'decentralization_requirement': 0.70,
            'network_load': 0.45,
            'attack_risk': 0.55,
        }
        self.block_interval = 2.0
        self.tx_spawn_rate = 0.3
    
    def _set_network_state_attack(self):
        """Byzantine Attack: high security, high energy, high decentralization -> expects PoW"""
        self.network_state = {
            'num_nodes': 700,
            'connectivity': 0.82,
            'latency_requirement_sec': 45.0,
            'throughput_requirement_tps': 30,
            'byzantine_tolerance': 0.28,
            'security_priority': 0.94,
            'energy_budget': 0.88,
            'bandwidth_mbps': 500,
            'consistency_requirement': 0.90,
            'decentralization_requirement': 0.92,
            'network_load': 0.30,
            'attack_risk': 0.85,
        }
        self.block_interval = 3.5
        self.tx_spawn_rate = 0.5
    
    def _set_network_state_emergency(self):
        """Emergency Scale-up: large network, very high throughput -> expects DPoS"""
        self.network_state = {
            'num_nodes': 400,
            'connectivity': 0.72,
            'latency_requirement_sec': 7.0,
            'throughput_requirement_tps': 7700,
            'byzantine_tolerance': 0.12,
            'security_priority': 0.72,
            'energy_budget': 0.30,
            'bandwidth_mbps': 5500,
            'consistency_requirement': 0.65,
            'decentralization_requirement': 0.62,
            'network_load': 0.70,
            'attack_risk': 0.40,
        }
        self.block_interval = 0.5
        self.tx_spawn_rate = 0.1
    
    def _spawn_node(self, node_id=0):
        """Spawn a blockchain node (vehicle)"""
        if self.scenario_idx == 0:
            speed = random.randint(60, 100)
            r = random.random()
            if r < 0.25:
                ntype, byz = 'validator', False
            elif r < 0.45:
                ntype, byz = 'delegate', False
            elif r < 0.55:
                ntype, byz = 'miner', False
            else:
                ntype, byz = 'normal', False
        elif self.scenario_idx == 1:
            speed = random.randint(30, 70)
            r = random.random()
            if r < 0.20:
                ntype, byz = 'byzantine', True
            elif r < 0.40:
                ntype, byz = 'validator', False
            elif r < 0.55:
                ntype, byz = 'miner', False
            else:
                ntype, byz = 'normal', False
        else:
            speed = random.randint(40, 80)
            r = random.random()
            if r < 0.30:
                ntype, byz = 'delegate', False
            elif r < 0.50:
                ntype, byz = 'validator', False
            elif r < 0.60:
                ntype, byz = 'miner', False
            else:
                ntype, byz = 'normal', False
        
        lane = random.randint(0, NUM_LANES - 1)
        node = BlockchainNode(
            x=random.randint(-300, self.world_length),
            y=ROAD_Y + 25 + lane * LANE_WIDTH,
            lane=lane,
            speed=speed,
            node_type=ntype,
            node_id=node_id,
            stake=random.uniform(10, 1000),
            compute_power=random.uniform(0.1, 1.0),
            reputation=random.uniform(0.5, 1.0) if not byz else random.uniform(0.1, 0.4),
            is_byzantine=byz,
            target_speed=speed,
        )
        self.nodes.append(node)
    
    def _update_network_dynamics(self, dt):
        """Dynamically update network conditions based on simulation state"""
        visible = [n for n in self.nodes if self.camera_x - 200 < n.x < self.camera_x + WINDOW_WIDTH + 200]
        n_visible = len(visible)
        n_byzantine = sum(1 for n in visible if n.is_byzantine)
        
        # Dynamic updates
        self.network_state['num_nodes'] = max(10, int(n_visible * 5))  # Scale up
        self.network_state['connectivity'] = max(0.6, min(0.95, 0.9 - n_visible * 0.003))
        self.network_state['network_load'] = min(0.95, 0.2 + n_visible * 0.015)
        
        if n_byzantine > 0:
            byz_ratio = n_byzantine / max(n_visible, 1)
            self.network_state['attack_risk'] = min(1.0, 0.3 + byz_ratio * 2)
            self.network_state['byzantine_tolerance'] = min(0.33, 0.1 + byz_ratio)
            self.network_state['security_priority'] = min(1.0, 0.7 + byz_ratio)
        
        # Scenario-specific dynamics
        if self.scenario_idx == 1:
            # Byzantine attack waves
            wave = 0.5 + 0.5 * math.sin(self.sim_time * 0.3)
            self.network_state['attack_risk'] = min(1.0, 0.6 + 0.35 * wave)
            # Spawn attack effects
            for n in visible:
                if n.is_byzantine and random.random() < 0.02:
                    self.attack_effects.spawn(n.x, n.y)
        
        elif self.scenario_idx == 2:
            # Emergency: throughput spikes
            spike = 1 + 0.5 * math.sin(self.sim_time * 0.5)
            self.network_state['throughput_requirement_tps'] = int(3000 * spike)
            self.network_state['latency_requirement_sec'] = max(0.5, 2.0 - self.sim_time * 0.01)
        
        # Compute derived metrics
        self.tps_current = self.network_state['throughput_requirement_tps'] * (1 - self.network_state['network_load'] * 0.3)
        self.latency_current = self.network_state['latency_requirement_sec'] * (1 + self.network_state['network_load'])
    
    def _spawn_transactions(self, dt):
        """Generate visual transaction flows between nodes"""
        self.tx_timer += dt
        if self.tx_timer >= self.tx_spawn_rate and len(self.nodes) > 1:
            self.tx_timer = 0
            # Pick two random visible nodes
            visible = [n for n in self.nodes if self.camera_x - 100 < n.x < self.camera_x + WINDOW_WIDTH + 100]
            if len(visible) >= 2:
                src, dst = random.sample(visible, 2)
                color = CONSENSUS_INFO.get(self.current_consensus, {}).get('color', (0, 255, 200))
                tx = BlockTransaction(
                    x=src.x, y=src.y + src.height // 2,
                    target_x=dst.x, target_y=dst.y + dst.height // 2,
                    speed=random.uniform(1.5, 3.0),
                    color=color
                )
                self.transactions.append(tx)
    
    def _generate_blocks(self, dt):
        """Generate confirmed blocks on the chain"""
        self.block_timer += dt
        if self.block_timer >= self.block_interval:
            self.block_timer = 0
            self.blocks_confirmed += 1
            block = ChainBlock(
                x=self.blocks_confirmed * 45,
                y=0,
                block_num=self.blocks_confirmed,
                consensus=self.current_consensus
            )
            self.chain_blocks.append(block)
            # Keep last 30 blocks
            if len(self.chain_blocks) > 30:
                self.chain_blocks = self.chain_blocks[-30:]
    
    # ==========================================================================
    # Update & Draw
    # ==========================================================================
    
    def update(self, dt):
        if self.paused:
            return
        
        self.sim_time += dt
        
        # Camera follows nodes
        if self.nodes:
            target_x = sum(n.x for n in self.nodes) / len(self.nodes) - WINDOW_WIDTH // 2
            self.camera_x += (target_x - self.camera_x) * 0.06
        
        # Update objects
        for n in self.nodes:
            n.update(dt, self.traffic_lights, self.nodes)
        for light in self.traffic_lights:
            light.update(dt)
        for rsu in self.rsus:
            rsu.update(dt)
        self.attack_effects.update(dt)
        
        # Update transactions
        self.transactions = [tx for tx in self.transactions if not tx.update(dt)]
        
        # Remove out-of-bounds nodes, respawn
        self.nodes = [n for n in self.nodes if -600 < n.x < self.world_length + 600]
        max_nodes = 25 if self.scenario_idx == 0 else 35 if self.scenario_idx == 1 else 45
        if len(self.nodes) < max_nodes and random.random() < 0.12:
            self._spawn_node(node_id=len(self.nodes))
        
        # Dynamic network
        self._update_network_dynamics(dt)
        
        # Transactions and blocks
        self._spawn_transactions(dt)
        self._generate_blocks(dt)
        
        # Re-predict every 1 second
        if int(self.sim_time * 10) % 10 == 0:
            self._predict_consensus()
            # Record history
            self.prediction_history.append({
                'time': self.sim_time,
                'consensus': self.current_consensus,
                'confidence': self.consensus_confidence,
                'attack_risk': self.network_state['attack_risk'],
                'network_load': self.network_state['network_load'],
            })
            if len(self.prediction_history) > 300:
                self.prediction_history = self.prediction_history[-300:]
    
    def draw(self):
        # Background
        self.screen.fill(COLORS['bg'])
        
        # Ground area (dark green for blockchain theme)
        pygame.draw.rect(self.screen, COLORS['grass'], (0, 0, WINDOW_WIDTH, ROAD_Y))
        pygame.draw.rect(self.screen, COLORS['grass'], 
                        (0, ROAD_Y + ROAD_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - ROAD_Y - ROAD_HEIGHT))
        
        # Road
        pygame.draw.rect(self.screen, COLORS['road'], (0, ROAD_Y, WINDOW_WIDTH, ROAD_HEIGHT))
        
        # Lane lines
        for i in range(1, NUM_LANES):
            y = ROAD_Y + 25 + i * LANE_WIDTH - LANE_WIDTH // 2
            for x in range(-int(self.camera_x % 70), WINDOW_WIDTH, 70):
                pygame.draw.line(self.screen, COLORS['road_line'], (x, y), (x + 35, y), 4)
        
        # Draw objects
        for rsu in self.rsus:
            rsu.draw(self.screen, self.camera_x)
        for light in self.traffic_lights:
            light.draw(self.screen, self.camera_x)
        
        # Transaction flows (behind nodes)
        for tx in self.transactions:
            tx.draw(self.screen, self.camera_x)
        
        # Nodes
        for n in self.nodes:
            n.draw(self.screen, self.camera_x)
        
        # Attack effects
        self.attack_effects.draw(self.screen, self.camera_x)
        
        # UI overlay
        self._draw_ui()
        
        # Chain visualization (bottom)
        self._draw_chain_visualization()
        
        pygame.display.flip()
    
    def _get_text(self, text, size, color):
        """Get cached text surface (crisp TrueType rendering)"""
        key = f"{text}_{size}_{color}"
        if key not in self.text_cache:
            self.text_cache[key] = render_text(text, size, color)
        return self.text_cache[key]
    
    def _draw_ui(self):
        """Draw top panel UI — compact layout, no overflow."""
        # --- Panel background ---
        panel_surf = pygame.Surface((WINDOW_WIDTH, PANEL_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (*COLORS['panel_bg'], 235),
                         (0, 0, WINDOW_WIDTH, PANEL_HEIGHT), border_radius=0)
        pygame.draw.line(panel_surf, COLORS['panel_border'],
                         (0, PANEL_HEIGHT - 2), (WINDOW_WIDTH, PANEL_HEIGHT - 2), 3)
        self.screen.blit(panel_surf, (0, 0))
        
        # --- Row 1: Title bar (y=6..50) ---
        title_text = f"SCENARIO: {self.scenario_names[self.scenario_idx]}"
        title_surf = self._get_text(title_text, 32, COLORS['text_white'])
        self.screen.blit(title_surf, (WINDOW_WIDTH // 2 - title_surf.get_width() // 2, 8))
        
        sub_text = f"{self.scenario_descriptions[self.scenario_idx]}  |  SPACE=Switch  1/2/3=Jump  P=Pause  ESC=Quit"
        sub_surf = self._get_text(sub_text, 15, COLORS['text_dim'])
        self.screen.blit(sub_surf, (WINDOW_WIDTH // 2 - sub_surf.get_width() // 2, 44))
        
        # Divider line under title
        pygame.draw.line(self.screen, (40, 60, 100), (20, 64), (WINDOW_WIDTH - 20, 64), 1)
        
        # --- Row 2: Four data columns (y=70..200) ---
        row_y = 72
        line_h = 24   # line spacing
        
        cons_color = CONSENSUS_INFO.get(self.current_consensus, {}).get('color', (200, 200, 200))
        
        # ---- Column 1: Consensus Decision (x=30..370) ----
        c1 = 30
        self._blit(c1, row_y,      "CONSENSUS DECISION",  17, COLORS['text_white'])
        self._blit(c1, row_y + line_h,   f"Selected: {self.current_consensus}", 26, cons_color)
        self._blit(c1, row_y + line_h*2+4, f"Confidence: {self.consensus_confidence:.1%}", 18,
                   COLORS['text_success'] if self.consensus_confidence > 0.8 else COLORS['text_warning'])
        self._blit(c1, row_y + line_h*3+4, f"Blocks: {self.blocks_confirmed}  |  {self.prediction_source}", 15, COLORS['text_dim'])
        
        # Probability bars (compact, 5 rows × 16px = 80px)
        bar_top = row_y + line_h*4 + 10
        for i, c in enumerate(CONSENSUS_CLASSES):
            prob = self.consensus_probs.get(c, 0)
            c_color = CONSENSUS_INFO[c]['color']
            by = bar_top + i * 16
            # Label
            lbl = self._get_text(f"{c}", 14, c_color)
            self.screen.blit(lbl, (c1, by))
            # Bar bg
            bx = c1 + 58
            bw = 140
            pygame.draw.rect(self.screen, (35, 45, 65), (bx, by + 2, bw, 10), border_radius=3)
            # Bar fill
            fw = int(bw * prob)
            if fw > 0:
                pygame.draw.rect(self.screen, c_color, (bx, by + 2, fw, 10), border_radius=3)
            # Percentage
            pct = self._get_text(f"{prob:.0%}", 13, c_color)
            self.screen.blit(pct, (bx + bw + 6, by))
        
        # ---- Column 2: Network Status (x=380..680) ----
        c2 = 390
        self._blit(c2, row_y,          "NETWORK STATUS", 17, COLORS['text_white'])
        self._blit(c2, row_y + line_h,      f"Nodes: {self.network_state['num_nodes']}", 22, (100, 200, 255))
        load_v = self.network_state['network_load']
        self._blit(c2, row_y + line_h*2,    f"Load: {load_v:.0%}", 22,
                   COLORS['text_danger'] if load_v > 0.7 else COLORS['text_success'])
        self._blit(c2, row_y + line_h*3,    f"Connectivity: {self.network_state['connectivity']:.0%}", 17, COLORS['text_dim'])
        self._blit(c2, row_y + line_h*4,    f"Bandwidth: {self.network_state['bandwidth_mbps']:.0f} Mbps", 17, COLORS['text_dim'])
        self._blit(c2, row_y + line_h*5,    f"TPS: {self.tps_current:.0f}", 17, COLORS['text_dim'])
        
        # Node legend (below network status)
        legend_y = row_y + line_h*6 + 8
        legend_items = [
            ("Val", COLORS['node_validator']),
            ("Mine", COLORS['node_miner']),
            ("Del", COLORS['node_delegate']),
            ("Byz", COLORS['node_byzantine']),
            ("Norm", COLORS['node_normal']),
        ]
        for i, (name, color) in enumerate(legend_items):
            lx = c2 + i * 62
            pygame.draw.circle(self.screen, color, (lx, legend_y + 6), 5)
            s = self._get_text(name, 13, color)
            self.screen.blit(s, (lx + 9, legend_y))
        
        # ---- Column 3: Security Status (x=720..1050) ----
        c3 = 730
        risk = self.network_state['attack_risk']
        risk_label = "CRITICAL" if risk > 0.7 else "ELEVATED" if risk > 0.4 else "LOW"
        risk_color = COLORS['text_danger'] if risk > 0.7 else COLORS['text_warning'] if risk > 0.4 else COLORS['text_success']
        
        self._blit(c3, row_y,          "SECURITY STATUS", 17, COLORS['text_white'])
        self._blit(c3, row_y + line_h,      f"Attack Risk: {risk_label}", 22, risk_color)
        self._blit(c3, row_y + line_h*2,    f"Risk Level: {risk:.0%}", 18, risk_color)
        self._blit(c3, row_y + line_h*3,    f"BFT Tolerance: {self.network_state['byzantine_tolerance']:.0%}", 17, COLORS['text_dim'])
        self._blit(c3, row_y + line_h*4,    f"Security: {self.network_state['security_priority']:.0%}", 17, COLORS['text_dim'])
        self._blit(c3, row_y + line_h*5,    f"Decentralization: {self.network_state['decentralization_requirement']:.0%}", 17, COLORS['text_dim'])
        
        # ---- Column 4: Performance (x=1100..1450) ----
        c4 = 1110
        self._blit(c4, row_y,          "PERFORMANCE", 17, COLORS['text_white'])
        self._blit(c4, row_y + line_h,      f"Latency Req: {self.network_state['latency_requirement_sec']:.1f}s", 22, (220, 195, 100))
        self._blit(c4, row_y + line_h*2,    f"Throughput: {self.network_state['throughput_requirement_tps']:.0f} TPS", 18, (220, 195, 100))
        self._blit(c4, row_y + line_h*3,    f"Energy: {self.network_state['energy_budget']:.0%}", 17, COLORS['text_dim'])
        self._blit(c4, row_y + line_h*4,    f"Consistency: {self.network_state['consistency_requirement']:.0%}", 17, COLORS['text_dim'])
        self._blit(c4, row_y + line_h*5,    f"Time: {self.sim_time:.1f}s", 17, COLORS['text_dim'])
    
    def _blit(self, x, y, text, size, color):
        """Helper: render text and blit at (x, y)."""
        # Dynamic text (changes every frame) should NOT be cached
        # Static text (headers) can be cached
        if any(ch in text for ch in '0123456789.%:'):
            # Dynamic — render fresh each frame
            surf = render_text(text, size, color)
        else:
            surf = self._get_text(text, size, color)
        self.screen.blit(surf, (x, y))
    
    def _draw_chain_visualization(self):
        """Draw blockchain visualization at the bottom."""
        chain_panel_h = 70
        chain_y = WINDOW_HEIGHT - chain_panel_h
        
        # Panel
        panel_surf = pygame.Surface((WINDOW_WIDTH, chain_panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (*COLORS['panel_bg'], 220),
                         (0, 0, WINDOW_WIDTH, chain_panel_h))
        pygame.draw.line(panel_surf, COLORS['panel_border'], (0, 0), (WINDOW_WIDTH, 0), 3)
        self.screen.blit(panel_surf, (0, chain_y))
        
        # Label
        lbl = render_text("BLOCKCHAIN LEDGER", 15, COLORS['text_white'])
        self.screen.blit(lbl, (20, chain_y + 6))
        
        # Blocks
        if self.chain_blocks:
            start_x = 190
            for i, block in enumerate(self.chain_blocks[-26:]):
                bx = start_x + i * 52
                by = chain_y + 6
                color = CONSENSUS_INFO.get(block.consensus, {}).get('color', (150, 150, 150))
                rect = pygame.Rect(bx, by, 38, 26)
                pygame.draw.rect(self.screen, color, rect, 0, border_radius=4)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=4)
                num_s = render_text(f"#{block.block_num}", 11, (255, 255, 255))
                self.screen.blit(num_s, (bx + 4, by + 5))
                if i > 0:
                    pygame.draw.line(self.screen, COLORS['chain_link'],
                                     (bx - 14, by + 13), (bx, by + 13), 2)
                    pygame.draw.circle(self.screen, COLORS['chain_link'], (bx - 7, by + 13), 3)
        
        # Info line
        info = render_text(
            f"Confirmed: {self.blocks_confirmed} blocks  |  Consensus: {self.current_consensus}  |  {self.prediction_source}",
            14, COLORS['text_dim'])
        self.screen.blit(info, (20, chain_y + 42))
    
    # ==========================================================================
    # Event handling
    # ==========================================================================
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.scenario_idx = (self.scenario_idx + 1) % 3
                    self._switch_scenario()
                elif event.key == pygame.K_1:
                    self.scenario_idx = 0
                    self._switch_scenario()
                elif event.key == pygame.K_2:
                    self.scenario_idx = 1
                    self._switch_scenario()
                elif event.key == pygame.K_3:
                    self.scenario_idx = 2
                    self._switch_scenario()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"{'PAUSED' if self.paused else 'RESUMED'}")
    
    def _switch_scenario(self):
        print(f"Scenario: {self.scenario_names[self.scenario_idx]}")
        self._init_objects()
    
    # ==========================================================================
    # Main loop
    # ==========================================================================
    
    def run(self):
        print("=" * 70)
        print("  Model 4: ConsensusRetNet - Blockchain V2X Simulation")
        print("=" * 70)
        print(f"  Scenario: {self.scenario_names[self.scenario_idx]}")
        print(f"  Model: {'RetNet loaded' if self.model else 'DEMO mode'}")
        print(f"  Controls: SPACE=Switch | 1/2/3=Jump | P=Pause | ESC=Quit")
        print("=" * 70)
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
        
        pygame.quit()
        print("\nSimulation ended.")
        print(f"  Total time: {self.sim_time:.1f}s")
        print(f"  Blocks confirmed: {self.blocks_confirmed}")
        print(f"  Final consensus: {self.current_consensus}")


if __name__ == "__main__":
    scenario = 0
    if len(sys.argv) > 1:
        try:
            scenario = max(0, min(2, int(sys.argv[1])))
        except ValueError:
            pass
    
    sim = ConsensusSimulationSystem(scenario_idx=scenario)
    sim.run()
