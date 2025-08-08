# 实验结果分析与最佳参数配置

## 📊 实验概述

本项目进行了128组不同参数配置的MADDPG算法实验，旨在寻找在机艇协同搜索任务中的最优超参数组合。

### 实验参数范围
- **lr_actor**: [0.001, 0.0005] - Actor网络学习率
- **lr_critic**: [0.001, 0.0005] - Critic网络学习率  
- **gamma**: [0.95, 0.99] - 折扣因子
- **tau**: [0.01, 0.05] - 软更新参数
- **batch_size**: [64, 128] - 批次大小
- **update_freq**: [100, 200] - 更新频率
- **reward_detect**: [200, 300] - 检测奖励值

### 评估指标
- **final_avg_reward**: 最终平均奖励
- **final_detection_rate**: 最终检测率（%）
- **best_episode_reward**: 最佳单回合奖励
- **best_detection_count**: 最佳检测数量
- **composite_score**: 综合得分 = 检测率 × 1000 + 平均奖励

## 🏆 实验结果TOP配置

### 1. 最高综合得分配置（推荐使用）
```
实验编号: exp_7
参数配置:
├── lr_actor: 0.001
├── lr_critic: 0.001  
├── gamma: 0.95
├── tau: 0.01
├── batch_size: 128
├── update_freq: 200
└── reward_detect: 300

性能表现:
├── 综合得分: 66899.7
├── 平均奖励: 56699.7
├── 检测率: 10.2%
├── 最佳单回合奖励: 77406.6
└── 最佳检测数: 15
```

### 2. 最高检测率配置
```
实验编号: exp_80
参数配置:
├── lr_actor: 0.0005
├── lr_critic: 0.001
├── gamma: 0.99
├── tau: 0.01
├── batch_size: 64
├── update_freq: 100
└── reward_detect: 200

性能表现:
├── 综合得分: 64735.2
├── 平均奖励: 52635.2
├── 检测率: 12.1%
├── 最佳单回合奖励: 67043.3
└── 最佳检测数: 18
```

### 3. 最高平均奖励配置
```
实验编号: exp_7 (与综合最优相同)
平均奖励: 56699.7
```

### 4. 最高单回合奖励配置
```
实验编号: exp_77
参数配置:
├── lr_actor: 0.0005
├── lr_critic: 0.001
├── gamma: 0.95
├── tau: 0.05
├── batch_size: 128
├── update_freq: 100
└── reward_detect: 300

最佳单回合奖励: 82949.98
```

## 📈 参数影响分析

### 重要发现

1. **batch_size 影响最显著**
   - batch_size=128 平均奖励: 40745.4
   - batch_size=64 平均奖励: 38323.4
   - **提升约6.3%**

2. **reward_detect 次重要**
   - reward_detect=300 平均奖励: 40455.3
   - reward_detect=200 平均奖励: 38613.5
   - **提升约4.8%**

3. **lr_critic 有一定影响**
   - lr_critic=0.001 平均奖励: 40456.8
   - lr_critic=0.0005 平均奖励: 38612.0
   - **提升约4.8%**

4. **其他参数影响相对较小**
   - lr_actor, gamma, tau, update_freq 对结果影响不超过3%

### 参数优先级排序
1. **batch_size** (最重要)
2. **reward_detect** 
3. **lr_critic**
4. **gamma**
5. **update_freq**
6. **tau**
7. **lr_actor** (影响最小)

## 🎯 推荐配置策略

### 通用最优配置
```python
config = {
    "lr_actor": 0.001,
    "lr_critic": 0.001,
    "gamma": 0.95,
    "tau": 0.01,
    "batch_size": 128,
    "update_freq": 200,
    "reward_detect": 300
}
```
**使用场景**: 追求整体性能最优，综合考虑奖励和检测率

### 检测优先配置
```python
config = {
    "lr_actor": 0.0005,
    "lr_critic": 0.001,
    "gamma": 0.99,
    "tau": 0.01,
    "batch_size": 64,
    "update_freq": 100,
    "reward_detect": 200
}
```
**使用场景**: 任务更重视目标检测准确率

### 奖励优先配置  
```python
config = {
    "lr_actor": 0.001,
    "lr_critic": 0.001,
    "gamma": 0.95,
    "tau": 0.01,
    "batch_size": 128,
    "update_freq": 200,
    "reward_detect": 300
}
```
**使用场景**: 追求最高累积奖励（与通用最优相同）

## 📋 稳定性分析

### 综合表现稳定的配置TOP5
1. **exp_77** & **exp_7** (并列第一) - 综合排名平均值: 7.0
2. **exp_19** & **exp_75** (并列第三) - 综合排名平均值: 9.7  
3. **exp_11** - 综合排名平均值: 11.7

这些配置在多个性能指标上都表现出色，具有良好的稳定性。

## 🔧 使用建议

### 快速开始
1. **直接使用推荐配置**: 采用exp_7的参数配置，性能最优且稳定
2. **根据任务调整**: 如果更重视检测率，使用exp_80配置
3. **资源受限情况**: 可以将batch_size降为64，但会有约6%的性能损失

### 调参指南
1. **优先调整batch_size**: 在计算资源允许的情况下尽量使用128
2. **其次调整reward_detect**: 300比200效果更好
3. **学习率组合**: lr_critic=0.001效果更稳定
4. **细节优化**: gamma=0.95, tau=0.01为较优选择

## 📁 文件结构
```
├── logs/              # 训练日志文件 (.txt)
├── results/           # 实验结果文件 (.json)
│   └── best.txt      # 最佳配置汇总
├── batch_tune.py     # 批量调参脚本
└── main.py          # 主训练脚本
```

## 🚀 运行方式

### 使用最优配置训练
```bash
python main.py --lr_actor=0.001 --lr_critic=0.001 --gamma=0.95 --tau=0.01 --batch_size=128 --update_freq=200 --reward_detect=300
```

### 批量调参
```bash
python batch_tune.py
```

---
*实验完成时间: 2025-08-08*  
*总实验次数: 128组配置*  
*最佳综合得分: 66899.7 (exp_7)*