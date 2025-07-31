# ippo.py
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class RolloutBuffer:
    """ 用于存储一个回合中的轨迹数据 """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    """ 支持连续和离散动作的Actor-Critic网络 """
    def __init__(self, state_dim, action_dim, action_type='discrete'):
        super(ActorCritic, self).__init__()
        
        self.action_type = action_type
        
        # 共享特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        if action_type == 'continuous':
            # 连续动作: 输出均值和标准差
            self.actor_mean = nn.Linear(128, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
            
            # 初始化网络权重，避免输出全零
            nn.init.xavier_uniform_(self.actor_mean.weight)
            nn.init.constant_(self.actor_mean.bias, 0)
        else:
            # 离散动作: 输出动作概率
            self.actor = nn.Sequential(
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )
            
            # 初始化网络权重
            nn.init.xavier_uniform_(self.actor[0].weight)
            nn.init.constant_(self.actor[0].bias, 0)
        
        # Critic网络
        self.critic = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0)

    def act(self, state):
        features = self.feature_net(state)
        
        if self.action_type == 'continuous':
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_logstd)
            dist = Normal(action_mean, action_std)
            raw_action = dist.sample()
            
            # 确保动作在正确范围内：[速度比例, 转向角度]
            action = torch.tanh(raw_action)  # 限制到 [-1, 1]
            action[0] = (action[0] + 1) / 2  # 速度比例映射到 [0, 1]
            # action[1] 保持在 [-1, 1] 作为转向比例
            
            action_logprob = dist.log_prob(raw_action).sum(dim=-1)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        features = self.feature_net(state)
        
        if self.action_type == 'continuous':
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_logstd)
            dist = Normal(action_mean, action_std)
            action_logprobs = dist.log_prob(action).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
        
        state_value = self.critic(features)
        return action_logprobs, state_value, dist_entropy

class PPOAgent:
    """ 单个智能体的PPO算法封装 """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_type='discrete'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_type = action_type
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_type)
        
        if action_type == 'continuous':
            self.optimizer = torch.optim.Adam([
                            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
                            {'params': [self.policy.actor_logstd], 'lr': lr_actor},
                            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                            {'params': self.policy.feature_net.parameters(), 'lr': lr_actor} # Added feature_net parameters
                        ])
        else:
            self.optimizer = torch.optim.Adam([
                            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                            {'params': self.policy.feature_net.parameters(), 'lr': lr_actor} # Added feature_net parameters
                        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_type)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        if self.action_type == 'continuous':
            return action.numpy()
        else:
            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))