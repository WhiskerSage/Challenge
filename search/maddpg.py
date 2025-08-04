# maddpg.py - Multi-Agent Deep Deterministic Policy Gradient Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

class ReplayBuffer:
    """经验回放缓冲区 - 适用于MADDPG的集中训练"""
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
        
    def push(self, state, action, reward, next_state, done):
        """
        存储一个多智能体经验
        state: 所有智能体的状态 [num_agents, state_dim]
        action: 所有智能体的动作 [num_agents, action_dim]  
        reward: 所有智能体的奖励 [num_agents]
        next_state: 所有智能体的下一状态 [num_agents, state_dim]
        done: 是否结束 [num_agents]
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """采样批次数据"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor网络 - 策略网络"""
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # 针对无人机艇搜索任务优化的网络结构
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        
        # 网络权重初始化
        self.init_weights()
        
    def init_weights(self):
        """初始化网络权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x * self.max_action

class Critic(nn.Module):
    """Critic网络 - 集中式Q网络，输入所有智能体的状态和动作"""
    def __init__(self, state_dim, action_dim, num_agents):
        super(Critic, self).__init__()
        
        # 输入维度：所有智能体的状态和动作
        total_input_dim = (state_dim + action_dim) * num_agents
        
        self.fc1 = nn.Linear(total_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        # 网络权重初始化
        self.init_weights()
        
    def init_weights(self):
        """初始化网络权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        
    def forward(self, states, actions):
        """
        前向传播
        states: [batch_size, num_agents, state_dim]
        actions: [batch_size, num_agents, action_dim]
        """
        # 将所有智能体的状态和动作展平
        batch_size = states.shape[0]
        states_flat = states.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        
        x = torch.cat([states_flat, actions_flat], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        
        return q_value

class OUNoise:
    """Ornstein-Uhlenbeck噪声 - 用于连续动作空间的探索"""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class MADDPGAgent:
    """MADDPG智能体"""
    def __init__(self, agent_id, state_dim, action_dim, num_agents, 
                 lr_actor=5e-5, lr_critic=2e-4, gamma=0.95, tau=0.005, 
                 max_action=1.0, device='cpu', weight_decay=1e-4):
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.device = device
        
        # 创建Actor网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
        
        # 创建Critic网络
        self.critic = Critic(state_dim, action_dim, num_agents).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # 优化的噪声生成器参数
        self.noise = OUNoise(action_dim, mu=0, theta=0.2, sigma=0.1)  # 调整参数
        
        # 训练统计
        self.actor_loss_history = []
        self.critic_loss_history = []
        
    def select_action(self, state, add_noise=True):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
            
        return action
        
    def update_critic(self, replay_buffer, batch_size, other_agents):
        """更新Critic网络（添加梯度裁剪）"""
        if len(replay_buffer) < batch_size:
            return
            
        # 采样经验
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算目标Q值
        next_actions = torch.zeros_like(actions)
        for i, agent in enumerate([self] + other_agents):
            next_actions[:, i, :] = agent.actor_target(next_states[:, i, :])
            
        target_q = self.critic_target(next_states, next_actions)
        target_q = rewards[:, self.agent_id].unsqueeze(1) + \
                  (self.gamma * target_q * (1 - dones[:, self.agent_id].unsqueeze(1)))
        
        # 计算当前Q值
        current_q = self.critic(states, actions)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        # 更新Critic（添加梯度裁剪）
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 梯度裁剪
        self.critic_optimizer.step()
        
        self.critic_loss_history.append(critic_loss.item())
        
    def update_actor(self, replay_buffer, batch_size, other_agents):
        """更新Actor网络（添加梯度裁剪）"""
        if len(replay_buffer) < batch_size:
            return
            
        # 采样经验
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        # 计算Actor损失
        actions_pred = torch.zeros_like(actions)
        for i, agent in enumerate([self] + other_agents):
            if i == self.agent_id:
                actions_pred[:, i, :] = agent.actor(states[:, i, :])
            else:
                actions_pred[:, i, :] = agent.actor(states[:, i, :]).detach()
                
        actor_loss = -self.critic(states, actions_pred).mean()
        
        # 更新Actor（添加梯度裁剪）
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # 梯度裁剪
        self.actor_optimizer.step()
        
        self.actor_loss_history.append(actor_loss.item())
        
    def soft_update(self):
        """软更新目标网络"""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 更新目标网络
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

class MADDPG:
    """MADDPG算法主类 - 管理所有智能体"""
    def __init__(self, num_agents, state_dim, action_dim, 
                 lr_actor=5e-5, lr_critic=2e-4, gamma=0.95, tau=0.005,
                 max_action=1.0, buffer_size=50000, device='cpu', weight_decay=1e-4):
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 创建所有智能体
        self.agents = []
        for i in range(num_agents):
            agent = MADDPGAgent(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_agents,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                max_action=max_action,
                device=device,
                weight_decay=weight_decay
            )
            self.agents.append(agent)
            
        # 共享经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练参数（优化版）
        self.batch_size = 128
        self.update_freq = 1
        self.steps = 0
        
    def select_actions(self, states, add_noise=True):
        """所有智能体选择动作"""
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(states[i], add_noise)
            actions.append(action)
        return np.array(actions)
        
    def store_transition(self, states, actions, rewards, next_states, dones):
        """存储一个转换到经验回放缓冲区"""
        self.replay_buffer.push(states, actions, rewards, next_states, dones)
        
    def update(self):
        """更新所有智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        self.steps += 1
        
        # 每update_freq步更新一次
        if self.steps % self.update_freq != 0:
            return
            
        for i, agent in enumerate(self.agents):
            other_agents = [self.agents[j] for j in range(self.num_agents) if j != i]
            
            # 更新Critic和Actor
            agent.update_critic(self.replay_buffer, self.batch_size, other_agents)
            agent.update_actor(self.replay_buffer, self.batch_size, other_agents)
            
            # 软更新目标网络
            agent.soft_update()
            
    def save_models(self, filepath_prefix):
        """保存所有智能体的模型"""
        for i, agent in enumerate(self.agents):
            agent.save(f"{filepath_prefix}_agent_{i}.pth")
            
    def load_models(self, filepath_prefix):
        """加载所有智能体的模型"""
        for i, agent in enumerate(self.agents):
            agent.load(f"{filepath_prefix}_agent_{i}.pth")
            
    def reset_noise(self):
        """重置所有智能体的噪声"""
        for agent in self.agents:
            agent.noise.reset()

# 兼容性包装器，保持与原IPPO接口一致
class PPOAgent:
    """MADDPG的兼容性包装器，保持与原始PPOAgent接口一致"""
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_type):
        # 注意：这个类只是为了接口兼容，实际的MADDPG训练在MADDPG类中进行
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.action_type = action_type
        
        # 创建一个简单的缓冲区以保持兼容性
        class SimpleBuffer:
            def __init__(self):
                self.rewards = []
                self.is_terminals = []
        
        self.buffer = SimpleBuffer()
        
        # 这些将在MADDPG实例化时被替换
        self.maddpg_agent = None
        self.agent_id = None
        
    def select_action(self, state):
        """选择动作 - 将被MADDPG重写"""
        if self.maddpg_agent is not None:
            return self.maddpg_agent.select_action(state)
        else:
            # 临时随机动作
            if self.action_type == 'continuous':
                return np.random.uniform(-1, 1, self.action_dim)
            else:
                return np.random.randint(0, self.action_dim)
                
    def update(self):
        """更新 - 在MADDPG中统一处理"""
        pass
        
    def save(self, filepath):
        """保存模型"""
        if self.maddpg_agent is not None:
            self.maddpg_agent.save(filepath)
            
    def load(self, filepath):
        """加载模型"""
        if self.maddpg_agent is not None:
            self.maddpg_agent.load(filepath)