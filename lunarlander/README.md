好的，这是一个重构和优化后的 README 文件。

我整合并简化了安装步骤，使其更加可靠和易于理解（特别是借鉴了我们之前解决 `box2d` 安装问题的经验）。同时，我也优化了整体结构和措辞，让它看起来更专业、更清晰。

-----

# 强化学习项目：PPO 算法训练登月舱

本项目使用 [PyTorch](https://pytorch.org/) 和 [Gymnasium](https://gymnasium.farama.org/) 环境，通过 **PPO (Proximal Policy Optimization)** 算法来训练一个能够成功在月球上着陆的智能体。

## 🚀 环境配置 (Environment Setup)

请确保你的系统已安装 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。推荐使用以下方法一进行配置。

### 方法一：使用 `environment.yml` (强烈推荐)

这是最可靠、最便捷的安装方式，可以一键创建包含所有依赖的 Conda 环境。

1.  在项目根目录下，创建一个名为 `environment.yml` 的文件，内容如下：

    ```yml
    name: lunarlander
    channels:
      - pytorch
      - conda-forge
      - defaults
    dependencies:
      - python=3.9
      - pytorch
      - torchvision
      - torchaudio
      - cpuonly
      - matplotlib
      - numpy
      - swig # 关键：预先安装 SWIG 以编译 Box2D
      - pip
      - pip:
        - gymnasium[box2d]
        - pygame
    ```

2.  打开终端，进入项目根目录，然后运行以下命令：

    ```bash
    conda env create -f environment.yml
    ```

    Conda 会自动创建名为 `lunarlander` 的环境并安装所有必需的包。

### 方法二：手动分步安装

如果你不想使用 `.yml` 文件，可以按照以下步骤手动安装。

1.  **创建并激活 Conda 环境**

    ```bash
    conda create -n lunarlander python=3.9 -y
    conda activate lunarlander
    ```

2.  **安装 PyTorch (CPU 版本)**

    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    ```

3.  **安装编译工具和其他依赖**
    这是解决 `Box2D` 安装问题的关键。我们使用 Conda 来安装 `swig`。

    ```bash
    conda install numpy matplotlib swig -c conda-forge -y
    ```

4.  **安装 Gymnasium 和 Box2D**
    由于 `swig` 已经存在，现在 `pip` 可以顺利完成编译和安装。

    ```bash
    pip install gymnasium[box2d] pygame
    ```

5.  **验证安装**

    ```bash
    python -c "import torch, gymnasium; print(f'✓ PyTorch 版本: {torch.__version__}'); print(f'✓ Gymnasium 版本: {gymnasium.__version__}'); print('\n🎉 环境配置成功!')"
    ```

## 🎮 如何运行 (How to Run)

1.  **激活环境**

    ```bash
    conda activate lunarlander
    ```

2.  **开始训练**

    ```bash
    python Main.py
    ```

    训练过程中，控制台会输出每个回合的奖励和平均奖励。训练完成后，模型将自动保存为 `PPO_LunarLander-v2.pth`。

## 📝 项目概览 (Project Overview)

  * **`Main.py`**: 项目主入口。负责初始化环境、智能体，并执行训练和评估循环。
  * **`PPO.py`**: PPO 算法的核心实现。包含策略网络、价值网络以及更新逻辑。

#### 主要超参数配置:

  - **学习率 (Learning Rate)**: `0.002`
  - **Gamma (折扣因子)**: `0.99`
  - **K Epochs (策略更新轮数)**: `80`
  - **更新步数 (Update Timestep)**: `2000`
  - **目标奖励 (Target Reward)**: `230` (当最近的平均奖励达到此值时，训练提前结束)

## 🔧 故障排查与提示 (Troubleshooting & Tips)

1.  **Conda 镜像源问题**: 如果下载速度慢或出现 HTTP 错误，可以考虑配置国内镜像源（如清华大学开源软件镜像站），或通过 `conda clean --all` 清理缓存后重试。

2.  **`gymnasium[box2d]` 安装失败**: 这个错误通常是因为缺少 `SWIG` 编译工具。请务必遵循推荐的安装方法，使用 Conda 提前安装 `swig`。

3.  **渲染问题 (Render Error)**:

      - 确保 `pygame` 已正确安装。
      - 在没有图形界面的服务器上运行时，请注释掉或删除代码中的 `env.render()` 相关行。

4.  **训练速度慢**: 本项目配置为 CPU 训练，对于强化学习任务来说，训练数小时是正常的。请耐心等待。

5.  **内存不足 (Out of Memory)**: 如果在内存较小的机器上运行，可以适当减小 `PPO.py` 中的 `update_timestep` 或 `max_timesteps` 参数。

---

## 解决方法

### 1. 修改环境创建方式

将环境初始化改为如下（推荐）：

```python
<code_block_to_apply_changes_from>
```

### 2. 代码修改示例

找到`Main.py`中如下部分：
```python
env_name = "LunarLander-v3"  # 游戏名字
env = gym.make(env_name)
```
改为：
```python
env_name = "LunarLander-v3"  # 游戏名字
env = gym.make(env_name, render_mode="human")
```

### 3. 说明

- `render_mode="human"` 会弹出可视化窗口。
- 只用`env.render()`在新版本gymnasium中不会自动弹窗，必须在`make`时指定。

---

## 总结

请将`Main.py`中的环境创建方式改为：
```python
env = gym.make(env_name, render_mode="human")
```
保存后重新运行，您就会看到图形化界面了！

如需我帮您自动修改代码，请告知！