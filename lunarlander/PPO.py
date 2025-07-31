import torch
import torch.nn as nn
from torch.distributions import Categorical

# 是否使用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Memory:
    def __init__(self):
        """初始化"""
        self.actions = []  # 行动(共4种)
        self.states = []  # 状态, 由8个数字组成
        self.logprobs = []  # 概率
        self.rewards = []  # 奖励
        self.is_terminals = []  # 游戏是否结束

    def clear_memory(self):
        """清除memory"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # 行动
        self.action_layer = nn.Sequential(
            # [b, 8] => [b, 64]
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),  # 激活

            # [b, 64] => [b, 64]
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),  # 激活

            # [b, 64] => [b, 4]
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # 评判
        self.value_layer = nn.Sequential(
            # [b, 8] => [8, 64]
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),  # 激活

            # [b, 64] => [b, 64]
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),

            # [b, 64] => [b, 1]
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        """前向传播, 由act替代"""

        raise NotImplementedError

    def act(self, state, memory):
        """计算行动"""

        # 转成张量
        state = torch.from_numpy(state).float().to(device)

        # 计算4个方向概率
        action_probs = self.action_layer(state)

        # 通过最大概率计算最终行动方向
        dist = Categorical(action_probs)
        action = dist.sample()

        # 存入memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        # 返回行动
        return action.item()

    def evaluate(self, state, action):
        """
        评估
        :param state: 状态, 2000个一组, 形状为 [2000, 8]
        :param action: 行动, 2000个一组, 形状为 [2000]
        :return:
        """

        # 计算行动概率
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)  # 转换成类别分布

        # 计算概率密度, log(概率)
        action_logprobs = dist.log_prob(action)

        # 计算熵
        dist_entropy = dist.entropy()

        # 评判
        state_value = self.value_layer(state)
        state_value = torch.squeeze(state_value)  # [2000, 1] => [2000]

        # 返回行动概率密度, 评判值, 行动概率熵
        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr  # 学习率
        self.betas = betas  # betas
        self.gamma = gamma  # gamma
        self.eps_clip = eps_clip  # 裁剪, 限制值范围
        self.K_epochs = K_epochs  # 迭代次数

        # 初始化policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)  # 优化器
        self.MseLoss = nn.MSELoss()  # 损失函数

    def update(self, memory):
        """更新梯度"""

        # 蒙特卡罗预测状态回报
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            # 回合结束
            if is_terminal:
                discounted_reward = 0

            # 更新削减奖励(当前状态奖励 + 0.99*上一状态奖励
            discounted_reward = reward + (self.gamma * discounted_reward)

            # 首插入
            rewards.insert(0, discounted_reward)

        # 标准化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 张量转换
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # 迭代优化 K 次:
        for _ in range(self.K_epochs):
            # 评估
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算损失
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 梯度清零
            self.optimizer.zero_grad()

            # 反向传播
            loss.mean().backward()

            # 更新梯度
            self.optimizer.step()

        # 将新的权重赋值给旧policy
        self.policy_old.load_state_dict(self.policy.state_dict())
