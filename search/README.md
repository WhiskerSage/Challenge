# 无人机艇协同搜索强化学习基础框架 (Baseline)

## 1. 框架定位

这是一个为“无人机艇协同搜索”竞赛场景设计的、稳定且经过优化的**强化学习基础框架 (Baseline Framework)**。

本框架的核心目标是为算法研究者和开发者提供一个可靠的起点。它包含一个功能完备的仿真环境和一个经过验证的IPPO（独立近端策略优化）算法实现。开发者可以轻松地在此框架上：
-   **测试新算法**: 替换`ippo.py`为自己的算法实现。
-   **对比性能**: 将新算法的表现与本框架提供的IPPO基线进行量化比较。
-   **实验新策略**: 调整奖励函数、观测空间或环境配置，研究不同策略对协同行为的影响。

## 2. 快速入门

### 2.1 安装依赖
```bash
pip install -r requirements.txt
```

### 2.2 运行基线训练
```bash
python main.py
```
默认配置已设为无渲染的高速训练模式。若需可视化，请修改`config.py`。

## 3. 项目文件结构解析

| 文件名             | 主要职责                                                     |
| ------------------ | ------------------------------------------------------------ |
| `main.py`          | **主程序入口**。负责组织训练循环，调用环境和算法。       |
| `environment.py`   | **核心仿真环境**。基于Gymnasium构建，定义了状态、动作和奖励。 |
| `agents.py`        | **智能体物理模型**。定义了UAV、USV和Target的属性和基础移动逻辑。 |
| `ippo.py`          | **基线算法实现**。提供了一个可运行的IPPO算法作为性能参考。 |
| `config.py`        | **中央配置文件**。所有超参数、奖励权重和物理参数都在此定义。 |
| `evaluator.py`     | **性能评估器**。用于计算和输出标准化的性能指标（如成功率、覆盖率）。 |
| `README.md`        | **开发文档** (本文)。                                        |
| `requirements.txt` | **项目依赖库**。                                               |


## 4. 如何扩展：实现并测试你的新算法

本框架的核心设计思想是易于扩展。只需遵循以下两步，即可将你的算法集成到框架中。

### 步骤1: 实现算法接口

你需要创建一个新的Python文件（例如 `my_maddpg.py`），并在其中实现一个与`PPOAgent`有相同接口的类。

**算法接口规范**:
```python
class YourAlgorithm:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_type):
        """
        构造函数。即使你的算法不需要所有参数，也请保持签名一致以便于替换。
        - state_dim: 状态空间的维度 (当前为44)
        - action_dim: 动作空间的维度 (离散:11, 连续:2)
        - action_type: 'discrete' 或 'continuous'
        """
        # ... 你的初始化代码 ...
        pass

    def select_action(self, state):
        """
        根据当前状态选择一个动作。
        - state: 一个np.array，代表单个智能体的观测值。
        - return: 返回一个动作。对于离散空间，返回一个int；对于连续空间，返回一个np.array。
        """
        # ... 你的动作选择逻辑 ...
        pass

    def update(self):
        """
        执行一次策略更新（训练）。
        框架会在每个 UPDATE_TIMESTEPS 调用此方法。
        你需要在此方法内管理你的经验回放缓冲区。
        """
        # ... 你的训练逻辑 ...
        pass
    
    # (可选) 保存和加载模型的方法
    def save(self, checkpoint_path):
        pass

    def load(self, checkpoint_path):
        pass
```

### 步骤2: 在`main.py`中替换算法

1.  **导入你的算法**:
    ```python
    # 在 main.py 的开头
    # from ippo import PPOAgent  <-- 注释或删除这行
    from my_maddpg import YourAlgorithm  # <-- 导入你的算法类
    ```

2.  **实例化你的算法**:
    在`main`函数中，找到智能体初始化的代码块，将其替换为你的算法类。
    ```python
    # 将:
    ppo_agents = {agent.id: PPOAgent(...) for agent in env.agents}
    # 替换为:
    ppo_agents = {agent.id: YourAlgorithm(...) for agent in env.agents}
    ```

完成以上两步后，运行 `main.py` 即可开始使用你的新算法进行训练。

## 5. 环境详解

### 5.1 观测空间 (44维向量)

| 组成部分         | 维度 | 细节                                            |
| ---------------- | ---- | ----------------------------------------------- |
| **自身状态**     | 6    | `[x, y, vx, vy, cos(h), sin(h)]` (均已归一化)    |
| **最近目标信息** | 9    | 3个最近目标的 `[rel_x, rel_y, is_detected]`   |
| **其他智能体状态** | 20   | 5个其他智能体的 `[rel_x, rel_y, vx, vy]`      |
| **局部探索状态** | 9    | 周围3×3网格的探索情况 `[0或1]`                    |

### 5.2 动作空间

-   **离散空间**: 11个动作，包括不同程度的转向、加减速和停止。
-   **连续空间**: 2维向量 `[速度比例, 转向比例]`。

### 5.3 奖励函数设计哲学

`config.py`中的奖励权重都经过了精心设计，以解决多智能体训练中的特定问题。理解这些奖励的意图，可以帮助你更好地进行实验。

-   `REWARD_DETECT`: **稀疏的核心奖励**。完成最终任务时给予。
-   `REWARD_APPROACHING_TARGET`: **密集的目标引导奖励**。解决奖励稀疏问题，引导智能体朝正确方向移动。
-   `REWARD_MOVEMENT`: **移动激励**。解决智能体因“风险规避”而倾向于静止不动的问题。
-   `REWARD_COLLISION` / `REWARD_OUT_OF_BOUNDS`: **安全惩罚**。权重较高，强制智能体学习避障和区域控制。
-   `REWARD_EXPLORE` / `REWARD_EFFICIENT_SEARCH`: **探索激励**。鼓励智能体探索未知区域，提高搜索覆盖率。
-   `REWARD_COORDINATION`: **协同奖励**。当多个智能体共同完成检测时给予，鼓励协作行为。

## 6. 配置指南 (`config.py`)

`config.py`是所有可调参数的中央枢纽。

-   **`ENABLE_RENDERING`**: 是否开启Pygame可视化。**训练时务必设为`False`**以获得数十倍的速度提升。
-   **`NUM_UAVS`, `NUM_USVS`**: 控制编队中的智能体数量。
-   **IPPO强化学习超参数**:
    -   `UPDATE_TIMESTEPS`: 控制智能体进行一次学习前要收集多少步的经验。
    -   `K_EPOCHS`: PPO算法在一次学习中迭代训练的次数。
    -   `LR_ACTOR`, `LR_CRITIC`: Actor和Critic网络的学习率。
    > **注意**: 这些超参数是为基线IPPO算法调优的，不同的算法可能需要不同的设置。
-   **奖励函数权重**: 你可以在这里调整上一节中所有奖励组件的权重，以测试不同的策略引导效果。



