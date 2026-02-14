#!/usr/bin/env python3
"""
Run ALL Case Studies and Experiments through the REAL ConsensusRetNet API
========================================================================
Sends requests to http://localhost:8000/api/v1/predict
Saves ALL raw API responses as verified experiment data.

Author: NOK KO
Date: 2026-02-05
"""

import json, time, requests
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

API_URL = "http://localhost:8000"
SRC_DIR = Path(__file__).parent.parent       # reaches src/
BASE_DIR = SRC_DIR.parent                    # reaches GIT_MODEL4/
OUT_DIR = BASE_DIR / "data" / "verified_api_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_KEYS = [
    'num_nodes', 'connectivity', 'latency_requirement_sec',
    'throughput_requirement_tps', 'byzantine_tolerance', 'security_priority',
    'energy_budget', 'bandwidth_mbps', 'consistency_requirement',
    'decentralization_requirement', 'network_load', 'attack_risk'
]


def api_predict(state: dict) -> dict:
    """Call real API and return full response"""
    payload = {"network_state": state}
    t0 = time.perf_counter()
    resp = requests.post(f"{API_URL}/api/v1/predict", json=payload, timeout=10)
    latency = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    data['client_latency_ms'] = round(latency, 3)
    return data


def api_predict_batch(states: list) -> dict:
    """Call real API batch endpoint"""
    payload = {"network_states": states}
    t0 = time.perf_counter()
    resp = requests.post(f"{API_URL}/api/v1/predict/batch", json=payload, timeout=30)
    latency = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    data['client_total_latency_ms'] = round(latency, 3)
    return data


def run_case_studies():
    """Run 5 Case Studies through real API"""
    print("=" * 70)
    print("  Running 5 Case Studies via Real API")
    print("=" * 70)

    cases = [
        {
            "case_id": "Case_1",
            "name": "Normal V2X Commuter Network",
            "description": "Stable vehicular network during regular commute hours. "
                           "Moderate traffic density, balanced network requirements. "
                           "No specific security threats detected.",
            "location": "Urban Highway Corridor, Region A",
            "expected_consensus": "Hybrid",
            "network_state": {
                "num_nodes": 150, "connectivity": 0.72,
                "latency_requirement_sec": 12.0, "throughput_requirement_tps": 1200,
                "byzantine_tolerance": 0.17, "security_priority": 0.80,
                "energy_budget": 0.60, "bandwidth_mbps": 1500,
                "consistency_requirement": 0.75, "decentralization_requirement": 0.70,
                "network_load": 0.45, "attack_risk": 0.55
            }
        },
        {
            "case_id": "Case_2",
            "name": "Byzantine Attack Scenario",
            "description": "Security monitoring detects anomalous node behavior — ~28% of nodes "
                           "sending forged location data. Attack risk surges to 85%. Network operators "
                           "prioritize maximum security over throughput and latency.",
            "location": "Metropolitan Expressway, Region B",
            "expected_consensus": "PoW",
            "network_state": {
                "num_nodes": 700, "connectivity": 0.82,
                "latency_requirement_sec": 45.0, "throughput_requirement_tps": 30,
                "byzantine_tolerance": 0.28, "security_priority": 0.94,
                "energy_budget": 0.88, "bandwidth_mbps": 500,
                "consistency_requirement": 0.90, "decentralization_requirement": 0.92,
                "network_load": 0.30, "attack_risk": 0.85
            }
        },
        {
            "case_id": "Case_3",
            "name": "Mass Event Emergency Scale-up",
            "description": "Post-event dispersal at a major stadium. 400+ autonomous ride-share vehicles "
                           "activated simultaneously. Transaction rate surges to 7700 TPS (payments, routing, "
                           "scheduling). Pure throughput scaling problem.",
            "location": "Sports District, Region C",
            "expected_consensus": "DPoS",
            "network_state": {
                "num_nodes": 400, "connectivity": 0.72,
                "latency_requirement_sec": 7.0, "throughput_requirement_tps": 7700,
                "byzantine_tolerance": 0.12, "security_priority": 0.72,
                "energy_budget": 0.30, "bandwidth_mbps": 5500,
                "consistency_requirement": 0.65, "decentralization_requirement": 0.62,
                "network_load": 0.70, "attack_risk": 0.40
            }
        },
        {
            "case_id": "Case_4",
            "name": "Energy-Constrained Remote Network",
            "description": "Remote mountain highway with 200 solar-powered RSU nodes. Energy budget "
                           "only 25% of nominal capacity. Low traffic volume (60 TPS) but blockchain "
                           "recording must remain uninterrupted.",
            "location": "Remote Highway, Region D",
            "expected_consensus": "PoS",
            "network_state": {
                "num_nodes": 200, "connectivity": 0.75,
                "latency_requirement_sec": 20.0, "throughput_requirement_tps": 60,
                "byzantine_tolerance": 0.18, "security_priority": 0.80,
                "energy_budget": 0.25, "bandwidth_mbps": 800,
                "consistency_requirement": 0.75, "decentralization_requirement": 0.75,
                "network_load": 0.45, "attack_risk": 0.60
            }
        },
        {
            "case_id": "Case_5",
            "name": "Small Network Instant Finality",
            "description": "Enclosed smart parking facility with 40 RSU nodes managing 300 parking spots. "
                           "Each parking command requires absolute finality within 2 seconds (vehicle already "
                           "in motion). High connectivity (90%), strong consistency requirement (92%).",
            "location": "Smart Parking Facility, Region E",
            "expected_consensus": "PBFT",
            "network_state": {
                "num_nodes": 40, "connectivity": 0.90,
                "latency_requirement_sec": 2.0, "throughput_requirement_tps": 3000,
                "byzantine_tolerance": 0.22, "security_priority": 0.85,
                "energy_budget": 0.50, "bandwidth_mbps": 2500,
                "consistency_requirement": 0.92, "decentralization_requirement": 0.45,
                "network_load": 0.55, "attack_risk": 0.50
            }
        },
    ]

    results = []
    for case in cases:
        print(f"\n  {case['case_id']}: {case['name']}")
        
        # Run multiple times for latency statistics
        latencies = []
        api_response = None
        for i in range(10):
            resp = api_predict(case['network_state'])
            latencies.append(resp['client_latency_ms'])
            if i == 0:
                api_response = resp
        
        pred = api_response['prediction']
        correct = pred['predicted_consensus'] == case['expected_consensus']
        
        result = {
            "case_id": case['case_id'],
            "name": case['name'],
            "description": case['description'],
            "location": case['location'],
            "expected_consensus": case['expected_consensus'],
            "api_response": api_response,
            "latency_statistics": {
                "num_runs": 10,
                "mean_ms": round(np.mean(latencies), 3),
                "std_ms": round(np.std(latencies), 3),
                "min_ms": round(np.min(latencies), 3),
                "max_ms": round(np.max(latencies), 3),
                "p50_ms": round(np.percentile(latencies, 50), 3),
                "p95_ms": round(np.percentile(latencies, 95), 3),
                "all_latencies_ms": [round(l, 3) for l in latencies],
            },
            "correct": correct,
        }
        results.append(result)
        
        status = "✅" if correct else "❌"
        print(f"    Predicted: {pred['predicted_consensus']} "
              f"(conf: {pred['confidence']:.6f}) "
              f"Expected: {case['expected_consensus']} {status}")
        print(f"    Latency: mean={np.mean(latencies):.1f}ms, p95={np.percentile(latencies, 95):.1f}ms")
        print(f"    Probabilities: {pred['probabilities']}")
    
    return results


def run_byzantine_resilience():
    """Sweep attack_risk 0→1 via real API"""
    print("\n" + "=" * 70)
    print("  Running Byzantine Resilience Sweep via Real API (50 points)")
    print("=" * 70)
    
    attack_risks = np.linspace(0, 1.0, 50).tolist()
    results = []
    
    for risk in attack_risks:
        state = {
            'num_nodes': int(150 + 550 * risk),
            'connectivity': max(0.60, 0.85 - 0.10 * risk),
            'latency_requirement_sec': 12.0 + 38 * risk,
            'throughput_requirement_tps': int(1200 - 1170 * risk),
            'byzantine_tolerance': min(0.33, 0.12 + 0.21 * risk),
            'security_priority': min(1.0, 0.72 + 0.26 * risk),
            'energy_budget': min(1.0, 0.40 + 0.55 * risk),
            'bandwidth_mbps': 1500 - 1000 * risk,
            'consistency_requirement': 0.75 + 0.15 * risk,
            'decentralization_requirement': min(1.0, 0.65 + 0.30 * risk),
            'network_load': 0.40 + 0.10 * risk,
            'attack_risk': risk,
        }
        resp = api_predict(state)
        results.append({
            'attack_risk': round(risk, 4),
            'predicted': resp['prediction']['predicted_consensus'],
            'confidence': resp['prediction']['confidence'],
            'probabilities': resp['prediction']['probabilities'],
            'inference_ms': resp['metadata']['inference_time_ms'],
        })
    
    # Find transitions
    transitions = []
    prev = results[0]['predicted']
    for r in results:
        if r['predicted'] != prev:
            transitions.append({
                'attack_risk': r['attack_risk'],
                'from': prev,
                'to': r['predicted'],
            })
            prev = r['predicted']
    
    print(f"  Transitions found: {len(transitions)}")
    for t in transitions:
        print(f"    α={t['attack_risk']:.2f}: {t['from']} → {t['to']}")
    
    return {"sweep_data": results, "transitions": transitions}


def run_dynamic_scenario():
    """100-second dynamic scenario via real API"""
    print("\n" + "=" * 70)
    print("  Running Dynamic Scenario (100s, 100 timesteps) via Real API")
    print("=" * 70)
    
    timestamps = np.linspace(0, 100, 100).tolist()
    results = []
    
    for t in timestamps:
        if t < 20:
            phase = t / 20
            state = {
                'num_nodes': 150, 'connectivity': 0.72,
                'latency_requirement_sec': 12.0, 'throughput_requirement_tps': 1200,
                'byzantine_tolerance': 0.17, 'security_priority': 0.80,
                'energy_budget': 0.60, 'bandwidth_mbps': 1500,
                'consistency_requirement': 0.75, 'decentralization_requirement': 0.70,
                'network_load': 0.45 + 0.05 * phase, 'attack_risk': 0.55,
            }
        elif t < 40:
            phase = (t - 20) / 20
            state = {
                'num_nodes': int(150 + 50 * phase), 'connectivity': 0.72 + 0.03 * phase,
                'latency_requirement_sec': 12.0 + 8 * phase,
                'throughput_requirement_tps': int(1200 - 1140 * phase),
                'byzantine_tolerance': 0.17 - 0.02 * phase,
                'security_priority': 0.80 - 0.02 * phase,
                'energy_budget': 0.60 - 0.35 * phase, 'bandwidth_mbps': 1500 - 700 * phase,
                'consistency_requirement': 0.75, 'decentralization_requirement': 0.70 + 0.05 * phase,
                'network_load': 0.50 - 0.05 * phase, 'attack_risk': 0.55 + 0.08 * phase,
            }
        elif t < 60:
            phase = (t - 40) / 20
            state = {
                'num_nodes': int(200 - 160 * phase), 'connectivity': 0.75 + 0.15 * phase,
                'latency_requirement_sec': 20.0 - 17.5 * phase,
                'throughput_requirement_tps': int(60 + 2000 * phase),
                'byzantine_tolerance': 0.15 + 0.07 * phase,
                'security_priority': 0.78 - 0.03 * phase,
                'energy_budget': 0.25 + 0.25 * phase, 'bandwidth_mbps': 800 + 2000 * phase,
                'consistency_requirement': 0.75 + 0.17 * phase,
                'decentralization_requirement': 0.75 - 0.30 * phase,
                'network_load': 0.45 + 0.10 * phase, 'attack_risk': 0.63,
            }
        elif t < 80:
            phase = (t - 60) / 20
            state = {
                'num_nodes': int(40 + 700 * phase), 'connectivity': 0.90 - 0.05 * phase,
                'latency_requirement_sec': 2.5 + 42 * phase,
                'throughput_requirement_tps': int(2060 - 2030 * phase),
                'byzantine_tolerance': 0.22 + 0.06 * phase,
                'security_priority': 0.75 + 0.19 * phase,
                'energy_budget': 0.50 + 0.38 * phase, 'bandwidth_mbps': 2800 - 2300 * phase,
                'consistency_requirement': 0.92 - 0.02 * phase,
                'decentralization_requirement': 0.45 + 0.47 * phase,
                'network_load': 0.55 - 0.25 * phase, 'attack_risk': 0.63 + 0.22 * phase,
            }
        else:
            phase = (t - 80) / 20
            state = {
                'num_nodes': int(740 - 340 * phase), 'connectivity': 0.85 - 0.13 * phase,
                'latency_requirement_sec': 44.5 - 37 * phase,
                'throughput_requirement_tps': int(30 + 7700 * phase),
                'byzantine_tolerance': 0.28 - 0.16 * phase,
                'security_priority': 0.94 - 0.22 * phase,
                'energy_budget': 0.88 - 0.58 * phase, 'bandwidth_mbps': 500 + 5000 * phase,
                'consistency_requirement': 0.90 - 0.25 * phase,
                'decentralization_requirement': 0.92 - 0.30 * phase,
                'network_load': 0.30 + 0.40 * phase, 'attack_risk': 0.85 - 0.45 * phase,
            }
        
        resp = api_predict(state)
        results.append({
            'time_s': round(t, 2),
            'predicted': resp['prediction']['predicted_consensus'],
            'confidence': resp['prediction']['confidence'],
            'probabilities': resp['prediction']['probabilities'],
            'network_state': state,
        })
    
    # Phase summary
    phase_defs = [
        ('Phase 1: Hybrid (Normal)', 0, 20),
        ('Phase 2: PoS (Low Energy)', 20, 40),
        ('Phase 3: PBFT (Small Net)', 40, 60),
        ('Phase 4: PoW (Attack)', 60, 80),
        ('Phase 5: DPoS (Scale-up)', 80, 100),
    ]
    phases = []
    for pname, t_start, t_end in phase_defs:
        phase_preds = [r['predicted'] for r in results if t_start <= r['time_s'] < t_end]
        from collections import Counter
        counts = Counter(phase_preds)
        dominant = counts.most_common(1)[0][0]
        dominance = counts.most_common(1)[0][1] / len(phase_preds)
        phases.append({
            'name': pname,
            'time_range': f"{t_start}-{t_end}s",
            'dominant_consensus': dominant,
            'dominance_ratio': round(dominance, 4),
            'distribution': dict(counts),
        })
        print(f"  {pname:35s}: {dominant} ({dominance:.0%})")
    
    return {"timestep_data": results, "phase_summary": phases}


def main():
    print("=" * 70)
    print("  ConsensusRetNet — Running ALL Experiments via Real API")
    print(f"  API: {API_URL}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    
    # Check API health
    health = requests.get(f"{API_URL}/api/v1/health").json()
    print(f"\n  API Status: {health['status']}")
    print(f"  Model loaded: {health['model_loaded']}")
    
    # Get model info
    model_info = requests.get(f"{API_URL}/api/v1/model/info").json()
    
    total_t0 = time.time()
    
    # Run experiments
    case_results = run_case_studies()
    byzantine_results = run_byzantine_resilience()
    dynamic_results = run_dynamic_scenario()
    
    total_time = time.time() - total_t0
    
    # Assemble final package
    package = {
        "_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "api_url": API_URL,
            "model_info": model_info,
            "api_health": health,
            "total_experiment_time_seconds": round(total_time, 2),
            "data_source": "Real ConsensusRetNet API (best_consensus.pth)",
            "verified": True,
            "author": "NOK KO",
        },
        "case_studies": case_results,
        "byzantine_resilience": byzantine_results,
        "dynamic_scenario": dynamic_results,
    }
    
    # Save
    out_path = OUT_DIR / "verified_experiment_data.json"
    with open(out_path, 'w') as f:
        json.dump(package, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Data saved: {out_path}")
    print(f"  Case Studies: {len(case_results)} ({sum(1 for c in case_results if c['correct'])}/5 correct)")
    print(f"  Byzantine sweep: {len(byzantine_results['sweep_data'])} points")
    print(f"  Dynamic scenario: {len(dynamic_results['timestep_data'])} timesteps")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
