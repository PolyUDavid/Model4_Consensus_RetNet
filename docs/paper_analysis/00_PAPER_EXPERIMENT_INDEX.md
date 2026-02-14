# Model 4: ConsensusRetNet — 论文实验数据完整索引

> **Author:** NOK KO  
> **Model:** ConsensusRetNet (RetNet Architecture, 5,402,126 parameters)  
> **All Data Verified:** Real API inference from `best_consensus.pth`  
> **Date:** 2026-02-05  
> **Status:** Paper-Ready

---

## 📋 Data Integrity Statement

**所有实验数据均通过以下方式生成与验证：**

1. **训练数据**：通过 `generate_consensus_data_v2.py` 生成 30,000 条平衡样本
2. **模型训练**：在 Apple MPS (GPU) 上训练 26 epochs，最优模型保存于 `best_consensus.pth`
3. **API 验证**：所有 Case Study 和实验数据通过 FastAPI (`http://localhost:8000/api/v1/predict`) 真实推理
4. **消融实验**：37 个模型变体在 CPU 上训练 78.6 分钟，每个变体独立训练
5. **数据格式**：原始 JSON + 分析 Markdown，可直接引用

---

## 📁 文件夹结构

```
paper_ready_experiments/
│
├── 00_PAPER_EXPERIMENT_INDEX.md          ← 本文件（总索引）
│
├── 01_training_data/                     ← 训练过程数据
│   ├── TRAINING_ANALYSIS.md              ← 训练曲线、收敛分析、混淆矩阵
│   └── training_history.json             ← 原始训练历史（26 epochs）
│
├── 02_case_studies/                      ← Case Study 详细数据
│   ├── CASE_STUDIES_DETAILED_REPORT.md   ← 5 个 Case 完整细节报告
│   ├── CROSS_CASE_COMPARISON.md          ← 横向纵向完整对比分析
│   └── scenario_predictions.json         ← 原始 JSON 预测数据
│
├── 03_architecture_comparison/           ← 架构对比
│   ├── ARCHITECTURE_COMPARISON.md        ← RetNet vs MLP/LSTM/CNN 完整对比
│   ├── test_results.json                 ← 测试结果 JSON
│   ├── validation_summary.json           ← 验证摘要 JSON
│   └── retnet_architecture.json          ← RetNet 架构定义
│
├── 04_dynamic_scenario/                  ← 动态场景切换实验
│   ├── DYNAMIC_SCENARIO_ANALYSIS.md      ← 5 阶段转换分析
│   └── dynamic_scenario_results.json     ← 100 秒仿真原始数据
│
├── 05_byzantine_resilience/              ← 拜占庭韧性实验
│   ├── BYZANTINE_RESILIENCE.md           ← 50 点扫描分析
│   └── (data in 09_raw_api_data/)
│
├── 06_ablation_study/                    ← 消融实验
│   ├── ABLATION_STUDY_ANALYSIS.md        ← 7 维度消融完整分析
│   └── ablation_study_results.json       ← 37 个变体原始数据
│
├── 07_model_excellence/                  ← 模型卓越性证明
│   ├── MODEL_EXCELLENCE_PROOF.md         ← 8 维度卓越性综合证明
│   └── COMPLETE_EXPERIMENT_RESULTS.md    ← 所有实验结果汇编
│
├── 08_visualizations/                    ← 可视化图表
│   ├── 1_architecture_comparison.png     ← 架构对比柱状图
│   ├── 2_dynamic_scenario_switching.png  ← 动态场景切换曲线
│   ├── 3_byzantine_resilience.png        ← 拜占庭韧性扫描曲线
│   ├── 4_confusion_matrix_f1.png         ← 混淆矩阵 + F1 热图
│   ├── 5_latency_throughput_heatmap.png  ← 延迟-吞吐量热图
│   ├── 1️⃣_loss_curve.png                 ← 训练损失曲线
│   ├── 2️⃣_accuracy_curve.png             ← 训练精度曲线
│   ├── 3️⃣_per_class_accuracy.png         ← 分类精度曲线
│   ├── 4️⃣_learning_rate_schedule.png     ← 学习率调度图
│   ├── 5️⃣_final_performance_summary.png  ← 最终性能总结
│   └── MODEL_BACKBONE_ARCHITECTURE.png   ← RetNet 骨干架构图
│
└── 09_raw_api_data/                      ← 原始 API 验证数据
    └── verified_experiment_data.json     ← 所有 API 实验原始数据
```

---

## 📊 实验清单与论文映射

### Experiment 1: Architecture Comparison (架构对比)

| 内容 | 文件 | 论文章节建议 |
|------|------|------------|
| RetNet vs MLP/LSTM/CNN | `03_architecture_comparison/ARCHITECTURE_COMPARISON.md` | Section IV-A |
| 精度对比表 | `03_architecture_comparison/test_results.json` | Table 2 |
| 可视化 | `08_visualizations/1_architecture_comparison.png` | Figure 3 |

### Experiment 2: Dynamic Scenario Switching (动态场景切换)

| 内容 | 文件 | 论文章节建议 |
|------|------|------------|
| 5 阶段分析 | `04_dynamic_scenario/DYNAMIC_SCENARIO_ANALYSIS.md` | Section IV-B |
| 原始数据 | `04_dynamic_scenario/dynamic_scenario_results.json` | — |
| 可视化 | `08_visualizations/2_dynamic_scenario_switching.png` | Figure 4 |

### Experiment 3: Byzantine Resilience (拜占庭韧性)

| 内容 | 文件 | 论文章节建议 |
|------|------|------------|
| 50 点扫描分析 | `05_byzantine_resilience/BYZANTINE_RESILIENCE.md` | Section IV-C |
| 原始 API 数据 | `09_raw_api_data/verified_experiment_data.json` | — |
| 可视化 | `08_visualizations/3_byzantine_resilience.png` | Figure 5 |

### Case Studies (案例研究)

| 内容 | 文件 | 论文章节建议 |
|------|------|------------|
| 5 个 Case 详细报告 | `02_case_studies/CASE_STUDIES_DETAILED_REPORT.md` | Section V |
| 横向纵向对比 | `02_case_studies/CROSS_CASE_COMPARISON.md` | Section V-F |
| 原始预测数据 | `02_case_studies/scenario_predictions.json` | — |

### Ablation Study (消融实验)

| 内容 | 文件 | 论文章节建议 |
|------|------|------------|
| 7 维度消融分析 | `06_ablation_study/ABLATION_STUDY_ANALYSIS.md` | Section IV-D |
| 37 个变体原始数据 | `06_ablation_study/ablation_study_results.json` | Tables 3-9 |

### Training Analysis (训练分析)

| 内容 | 文件 | 论文章节建议 |
|------|------|------------|
| 训练曲线与收敛 | `01_training_data/TRAINING_ANALYSIS.md` | Section III-D |
| 原始训练历史 | `01_training_data/training_history.json` | — |
| 可视化 | `08_visualizations/1️⃣_loss_curve.png` 等 | Figures 6-10 |

---

## 📈 核心数据速查表 (Quick Reference)

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.98% |
| **Macro F1-Score** | 0.9997 |
| **Parameters** | 5,402,126 |
| **Best Epoch** | 6 |
| **Inference Latency** | < 1ms (mean) |
| **Case Study Accuracy** | 5/5 = 100% |
| **Physical Consistency** | 5/5 = 100% |

### Ablation Robustness

| Worst-case Dimension | Accuracy Drop | Value |
|---------------------|--------------|-------|
| Feature Removal (Performance) | **-0.38%** | 99.58% |
| All other dimensions | < -0.07% | ≥ 99.93% |

### Architecture Comparison

| Architecture | Test Acc | F1 | Parameters |
|-------------|---------|-----|-----------|
| **RetNet** | **99.98%** | **0.9998** | 5.4M |
| MLP | 99.98% | 0.9998 | 70K |
| LSTM | 99.98% | 0.9998 | 205K |
| CNN | 99.09% | 0.9908 | 26K |

### Case Study Results (API-Verified)

| Case | Scenario | Prediction | Confidence |
|------|---------|-----------|-----------|
| Case 1 | Normal V2X | Hybrid ✅ | 92.76% |
| Case 2 | Byzantine Attack | PoW ✅ | 92.01% |
| Case 3 | Mass Event | DPoS ✅ | 92.95% |
| Case 4 | Energy-Limited | PoS ✅ | 92.55% |
| Case 5 | Small Network | PBFT ✅ | 93.32% |

### Byzantine Transition

| Metric | Value |
|--------|-------|
| Critical attack_risk | α ≈ 0.40 |
| Transition width | Δα ≈ 0.12 |
| Sweep points | 50 |
| Phase: Hybrid → PoW | At α = 0.408 |

---

## ✅ Data Verification Checklist

- [x] 训练数据来源: `generate_consensus_data_v2.py` (30,000 samples, balanced)
- [x] 模型检查点: `best_consensus.pth` (Val Acc 99.98%, Epoch 6)
- [x] API 服务: FastAPI at `localhost:8000` (served 3,625+ requests)
- [x] Case Study 数据: 5 cases × 10 runs per case = 50 API calls
- [x] Byzantine Sweep: 50 data points, continuous sweep [0, 1]
- [x] Dynamic Scenario: 100 timesteps, 5 phases
- [x] Ablation Study: 37 model variants, 78.6 minutes total training
- [x] Architecture Comparison: 4 architectures on same dataset
- [x] All JSON data files included for reproducibility
- [x] All visualizations generated from real data

---

*— NOK KO, 2026*  
*ConsensusRetNet: A Retentive Network for Adaptive Blockchain Consensus Selection in V2X Networks*
