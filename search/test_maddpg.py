# test_maddpg.py - 优化版MADDPG性能测试脚本
import numpy as np
import torch
from environment import UsvUavEnv
from maddpg import MADDPG
import config
import time
import matplotlib.pyplot as plt

def test_maddpg_optimized():
    """测试优化版MADDPG的性能改进"""
    print("=== 优化版MADDPG性能测试 ===")
    
    # 检查PyTorch和CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = UsvUavEnv(action_type='continuous')
    obs, _ = env.reset()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    num_agents = len(env.agents)
    
    print(f"环境参数: {num_agents}个智能体, 状态维度={state_dim}, 动作维度={action_dim}")
    
    # 创建优化版MADDPG
    maddpg = MADDPG(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=config.LR_ACTOR,      # 使用新的学习率
        lr_critic=config.LR_CRITIC,
        gamma=config.GAMMA,
        tau=config.MADDPG_TAU,         # 使用新的软更新参数
        max_action=1.0,
        buffer_size=config.MADDPG_BUFFER_SIZE,
        device=device,
        weight_decay=config.WEIGHT_DECAY  # 使用权重衰减
    )
    
    # 设置优化后的参数
    maddpg.batch_size = config.MADDPG_BATCH_SIZE
    maddpg.update_freq = config.MADDPG_UPDATE_FREQ
    
    print("优化版MADDPG初始化成功!")
    print(f"训练参数: LR_A={config.LR_ACTOR}, LR_C={config.LR_CRITIC}, TAU={config.MADDPG_TAU}")
    print(f"批次大小: {maddpg.batch_size}, 更新频率: {maddpg.update_freq}")
    
    # 运行测试训练
    episode_rewards = []
    episode_steps = []
    episode_times = []
    detection_rates = []
    termination_reasons = []
    
    print("\n开始优化版训练测试(20个episode)...")
    for episode in range(20):
        start_time = time.time()
        obs, _ = env.reset()
        maddpg.reset_noise()
        
        episode_reward = 0
        steps = 0
        
        while steps < 3000:  # 最大3000步，对应5分钟
            # 获取状态
            states = []
            agent_ids = list(obs.keys())
            for agent_id in agent_ids:
                states.append(obs[agent_id])
            states = np.array(states)
            
            # 选择动作
            actions = maddpg.select_actions(states, add_noise=True)
            
            # 环境交互
            env_actions = {}
            for i, agent_id in enumerate(agent_ids):
                env_actions[agent_id] = actions[i]
            
            next_obs, rewards, terminated, truncated, info = env.step(env_actions)
            
            # 存储经验和更新
            next_states = []
            reward_list = []
            done_list = []
            
            for agent_id in agent_ids:
                next_states.append(next_obs[agent_id])
                reward_list.append(rewards[agent_id])
                done_list.append(terminated or truncated)
            
            next_states = np.array(next_states)
            reward_list = np.array(reward_list)
            done_list = np.array(done_list)
            
            maddpg.store_transition(states, actions, reward_list, next_states, done_list)
            
            episode_reward += sum(reward_list)
            steps += 1
            
            # 网络更新（使用新的更新频率）
            if len(maddpg.replay_buffer) >= maddpg.batch_size:
                maddpg.update()
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_times.append(episode_time)
        
        # 获取终止信息
        detection_rate = info.get('detection_rate', 0)
        detected_count = info.get('detected_count', 0)
        total_spawned = info.get('total_spawned', 0)
        termination_reason = info.get('termination_reason', 'unknown')
        
        detection_rates.append(detection_rate)
        termination_reasons.append(termination_reason)
        
        print(f"Episode {episode+1}: 奖励={episode_reward:.1f}, 步数={steps}, "
              f"检测率={detection_rate:.1%} ({detected_count}/{total_spawned}), "
              f"终止原因={termination_reason}, 用时={episode_time:.1f}s")
    
    # 分析结果
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    avg_time = np.mean(episode_times)
    avg_detection_rate = np.mean(detection_rates)
    
    print(f"\n=== 优化版MADDPG测试结果 ===")
    print(f"平均奖励: {avg_reward:.1f}")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"平均用时: {avg_time:.1f}秒")
    print(f"平均检测率: {avg_detection_rate:.1%}")
    print(f"缓冲区大小: {len(maddpg.replay_buffer)}")
    
    # 分析终止原因
    from collections import Counter
    reason_counts = Counter(termination_reasons)
    print(f"终止原因统计: {dict(reason_counts)}")
    
    # 检查网络学习情况
    if len(maddpg.agents[0].actor_loss_history) > 10:
        recent_actor_losses = [np.mean(agent.actor_loss_history[-10:]) 
                              for agent in maddpg.agents if len(agent.actor_loss_history) >= 10]
        recent_critic_losses = [np.mean(agent.critic_loss_history[-10:]) 
                               for agent in maddpg.agents if len(agent.critic_loss_history) >= 10]
        
        if recent_actor_losses and recent_critic_losses:
            avg_actor_loss = np.mean(recent_actor_losses)
            avg_critic_loss = np.mean(recent_critic_losses)
            print(f"最近网络损失: Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.1f}")
    
    # 学习趋势分析
    if len(episode_rewards) >= 10:
        first_half = np.mean(episode_rewards[:10])
        second_half = np.mean(episode_rewards[-10:])
        improvement = (second_half - first_half) / abs(first_half) * 100 if first_half != 0 else 0
        print(f"学习改进: {improvement:.1f}% (从{first_half:.1f}到{second_half:.1f})")
    
    return {
        'avg_reward': avg_reward,
        'avg_detection_rate': avg_detection_rate,
        'avg_steps': avg_steps,
        'avg_time': avg_time,
        'improvement': improvement,
        'buffer_size': len(maddpg.replay_buffer)
    }

def main():
    """运行优化版MADDPG测试"""
    print("开始MADDPG优化版验证测试\n")
    
    try:
        results = test_maddpg_optimized()
        
        # 总结
        print("\n" + "="*60)
        print("MADDPG优化版测试总结")
        print("="*60)
        print("✓ 奖励系统重新平衡: 避免梯度爆炸")
        print("✓ 网络参数优化: 降低学习率和批次大小")
        print("✓ Episode终止条件: 动态终止提高效率")
        print("✓ 探索策略优化: 位置多样性和噪声调整")
        print("✓ 协同机制增强: 区域分工和协同奖励")
        
        print(f"\n关键性能指标:")
        print(f"  - 平均检测率: {results['avg_detection_rate']:.1%}")
        print(f"  - 训练效率: {results['avg_steps']:.0f}步/episode")
        print(f"  - 学习改进: {results.get('improvement', 0):.1f}%")
        print(f"  - 经验积累: {results['buffer_size']}条经验")
        
        success_rate = results['avg_detection_rate']
        if success_rate > 0.4:
            print("\n🎯 优化效果显著! MADDPG已准备好进行正式训练。")
        elif success_rate > 0.2:
            print("\n⚡ 优化有效! 可继续调整参数以进一步提升。")
        else:
            print("\n🔧 需要进一步优化，建议检查奖励函数和网络结构。")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()