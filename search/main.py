# main.py
import pygame
import numpy as np
from environment import UsvUavEnv
from maddpg import MADDPG, PPOAgent  # 使用MADDPG替换IPPO
import config
import torch

def adjust_difficulty(episode):
    """
    改进的课程学习：根据比赛特点动态调整难度
    重新平衡奖励权重以避免梯度爆炸
    """
    if episode < 100:
        # 初期：边界目标为主，速度较低，便于学习基础探测
        config.REWARD_TIME_STEP = -0.05  # 降低时间惩罚
        config.REWARD_EXPLORE = 1.0      # 增强探索奖励
        config.REWARD_COLLISION = -5.0   # 降低碰撞惩罚
        config.REWARD_OUT_OF_BOUNDS = -2.0 # 降低出界惩罚
        
        # 目标生成参数（新增）
        config.TARGET_BOUNDARY_PROB = 0.8  # 80%概率生成边界目标
        config.TARGET_SPEED_RANGE = (3, 8)  # 3-8节速度范围
        config.TARGET_GENERATION_RATE = 0.01  # 进一步降低生成频率（从0.02降到0.01）
        
        print(f"    [Curriculum] Easy mode: 边界目标主导, 低速目标, 探索奖励增强")
        
    elif episode < 300:
        # 中期：混合目标类型，逐步增加难度
        config.REWARD_TIME_STEP = -0.1   # 标准时间惩罚
        config.REWARD_EXPLORE = 0.5      # 标准探索奖励
        config.REWARD_COLLISION = -8.0   # 中等碰撞惩罚
        config.REWARD_OUT_OF_BOUNDS = -4.0 # 中等出界惩罚
        
        # 目标生成参数
        config.TARGET_BOUNDARY_PROB = 0.5  # 50%概率边界目标
        config.TARGET_SPEED_RANGE = (5, 12)  # 5-12节速度范围
        config.TARGET_GENERATION_RATE = 0.015  # 适中生成频率
        
        if episode == 100:
            print(f"    [Curriculum] Medium mode: 目标类型平衡, 速度提升")
            
    else:
        # 后期：高速目标为主，模拟实际比赛条件
        config.REWARD_TIME_STEP = -0.15  # 提高时间压力
        config.REWARD_EXPLORE = 0.3      # 降低探索奖励，重视效率
        config.REWARD_COLLISION = -10.0  # 恢复标准碰撞惩罚
        config.REWARD_OUT_OF_BOUNDS = -5.0 # 恢复标准出界惩罚
        
        # 目标生成参数
        config.TARGET_BOUNDARY_PROB = 0.6  # 60%概率边界目标（实际比赛特点）
        config.TARGET_SPEED_RANGE = (8, 15)  # 8-15节高速目标
        config.TARGET_GENERATION_RATE = 0.02  # 标准生成频率
        
        if episode == 300:
            print(f"    [Curriculum] Hard mode: 高速目标主导, 效率优先")

def generate_curriculum_target(env):
    """
    基于课程学习参数生成目标
    """
    import random
    
    # 根据课程设置确定目标类型
    is_boundary = random.random() < getattr(config, 'TARGET_BOUNDARY_PROB', 0.5)
    speed_range = getattr(config, 'TARGET_SPEED_RANGE', (5, 15))
    
    if is_boundary:
        # 边界目标生成
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            pos = [random.uniform(0, config.AREA_WIDTH_METERS), config.AREA_HEIGHT_METERS - 50]
            heading = random.uniform(-np.pi * 0.75, -np.pi * 0.25)  # 朝向下方
        elif edge == 'bottom':
            pos = [random.uniform(0, config.AREA_WIDTH_METERS), 50]
            heading = random.uniform(np.pi * 0.25, np.pi * 0.75)  # 朝向上方
        elif edge == 'left':
            pos = [50, random.uniform(0, config.AREA_HEIGHT_METERS)]
            heading = random.uniform(-np.pi * 0.25, np.pi * 0.25)  # 朝向右方
        else:  # right
            pos = [config.AREA_WIDTH_METERS - 50, random.uniform(0, config.AREA_HEIGHT_METERS)]
            heading = random.uniform(np.pi * 0.75, np.pi * 1.25)  # 朝向左方
    else:
        # 中心区域目标
        pos = [
            random.uniform(config.AREA_WIDTH_METERS * 0.2, config.AREA_WIDTH_METERS * 0.8),
            random.uniform(config.AREA_HEIGHT_METERS * 0.2, config.AREA_HEIGHT_METERS * 0.8)
        ]
        heading = random.uniform(0, 2 * np.pi)
    
    # 设置速度
    speed_knots = random.uniform(*speed_range)
    speed_mps = speed_knots * config.KNOTS_TO_MPS
    velocity = [speed_mps * np.cos(heading), speed_mps * np.sin(heading)]
    
    from agents import Target
    target = Target(f"target_{len(env.targets)}", pos, heading)
    target.velocity = np.array(velocity)
    target.spawn_time = env.current_time
    
    return target

def main():
    # 选择动作类型：'discrete' 或 'continuous'  
    action_type = 'continuous'  # MADDPG专为连续动作设计
    env = UsvUavEnv(action_type=action_type)
    
    # 只有启用渲染时才初始化时钟
    if config.ENABLE_RENDERING:
        clock = pygame.time.Clock()
    else:
        clock = None
        print("MADDPG训练模式：已禁用可视化渲染以提高训练速度")

    # --- 1. MADDPG智能体初始化 ---  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    maddpg = None
    is_maddpg_initialized = False

    # --- 2. 主训练循环 ---
    time_step = 0
    num_episodes = 500 # 增加训练回合数
    
    # 训练统计
    episode_rewards = []
    detection_stats = []
    
    for episode in range(num_episodes):
        # 应用课程学习
        adjust_difficulty(episode)
        
        print(f"--- Episode {episode + 1} ---")
        observations, info = env.reset()

        # 仅在第一个回合初始化MADDPG
        if not is_maddpg_initialized:
            state_dim = env.observation_space.shape[0]
            if action_type == 'continuous':
                action_dim = env.action_space.shape[0]
            else:
                action_dim = env.action_space.n
            
            num_agents = len(env.agents)
            
            # 创建MADDPG实例
            maddpg = MADDPG(
                num_agents=num_agents,
                state_dim=state_dim,
                action_dim=action_dim,
                lr_actor=config.LR_ACTOR,
                lr_critic=config.LR_CRITIC,
                gamma=config.GAMMA,
                tau=config.MADDPG_TAU,
                max_action=1.0,
                buffer_size=config.MADDPG_BUFFER_SIZE,
                device=device
            )
            
            # 更新MADDPG的训练参数
            maddpg.batch_size = config.MADDPG_BATCH_SIZE
            maddpg.update_freq = config.MADDPG_UPDATE_FREQ
            
            # 为兼容性创建ppo_agents映射
            ppo_agents = {}
            for i, agent in enumerate(env.agents):
                ppo_agent = PPOAgent(state_dim, action_dim, config.LR_ACTOR, 
                                   config.LR_CRITIC, config.GAMMA, 
                                   config.K_EPOCHS, config.EPS_CLIP, action_type)
                ppo_agent.maddpg_agent = maddpg.agents[i]
                ppo_agent.agent_id = i
                ppo_agents[agent.id] = ppo_agent
                
            is_maddpg_initialized = True
            print(f"MADDPG初始化完成: {num_agents}个智能体")

        # 重置噪声
        maddpg.reset_noise()

        terminated = False
        truncated = False
        running = True
        current_episode_reward = 0
        episode_detected_count = 0
        episode_steps = 0
        
        while running and not terminated and not truncated:
            time_step += 1
            episode_steps += 1
            
            # --- 窗口事件处理（仅在渲染模式下）---
            if config.ENABLE_RENDERING:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            # --- 3. MADDPG算法决策与数据收集 ---
            # 获取所有智能体的观测
            states = []
            agent_ids = []
            for agent_id, obs in observations.items():
                states.append(obs)
                agent_ids.append(agent_id)
            states = np.array(states)
            
            # MADDPG选择动作
            maddpg_actions = maddpg.select_actions(states, add_noise=True)
            
            # 构建环境所需的动作字典
            actions = {}
            for i, agent_id in enumerate(agent_ids):
                actions[agent_id] = maddpg_actions[i]
                
            # 调试：偶尔打印动作信息
            if time_step % 5000 == 0:
                print(f"    DEBUG: MADDPG actions shape = {maddpg_actions.shape}")
                print(f"    DEBUG: Sample action = {maddpg_actions[0]} (agent {agent_ids[0]})")

            # --- 环境交互 ---
            next_observations, rewards, terminated, truncated, info = env.step(actions)

            # --- 4. MADDPG经验存储 ---
            next_states = []
            reward_list = []
            done_list = []
            
            for agent_id in agent_ids:
                next_states.append(next_observations[agent_id])
                reward_list.append(rewards[agent_id])
                done_list.append(terminated or truncated)
                
            next_states = np.array(next_states)
            reward_list = np.array(reward_list)
            done_list = np.array(done_list)
            
            # 存储转换到MADDPG的经验回放缓冲区
            maddpg.store_transition(states, maddpg_actions, reward_list, next_states, done_list)
            
            # 为兼容性维护PPO buffer（实际不使用）
            for agent_id, ppo_agent in ppo_agents.items():
                ppo_agent.buffer.rewards.append(rewards[agent_id])
                ppo_agent.buffer.is_terminals.append(terminated or truncated)

            observations = next_observations
            current_episode_reward += sum(rewards.values())
            
            # 统计检测事件
            if 'detected_events' in info:
                episode_detected_count += len([event for event in info['detected_events'] 
                                             if 'successfully detected' in event])

            # --- 5. MADDPG模型更新 ---
            if time_step % config.UPDATE_TIMESTEPS == 0:
                print(f"    MADDPG UPDATING at time_step {time_step}...")
                maddpg.update()
                print(f"    MADDPG UPDATE finished.")
                
                # 打印训练统计
                if len(maddpg.agents[0].actor_loss_history) > 0:
                    avg_actor_loss = np.mean([agent.actor_loss_history[-10:] for agent in maddpg.agents])
                    avg_critic_loss = np.mean([agent.critic_loss_history[-10:] for agent in maddpg.agents])
                    print(f"    Avg Actor Loss: {avg_actor_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}")
            
            # --- 训练监控（优化版）---
            if config.ENABLE_RENDERING:
                env.render()
                clock.tick(config.TARGET_FPS)
            elif time_step % 1000 == 0:  # 每1000步监控一次
                avg_reward = current_episode_reward / episode_steps if episode_steps > 0 else 0
                print(f"    Step {time_step}, Episode Steps: {episode_steps}, Avg Reward: {avg_reward:.3f}, Episode Reward: {current_episode_reward:.1f}")
                
                # 使用新的info信息
                detection_rate = info.get('detection_rate', 0)
                detected_count = info.get('detected_count', 0)
                total_spawned = info.get('total_spawned', 0)
                print(f"    检测进度: {detected_count}/{total_spawned} ({detection_rate:.1%}), Episode total: {episode_detected_count}")
                
                # 显示智能体状态（包含所有智能体）
                all_agents = [(agent.id, agent.pos) for agent in env.agents]
                print(f"    智能体位置: {[(aid, f'({pos[0]:.0f},{pos[1]:.0f})') for aid, pos in all_agents[:3]]}...")  # 只显示前3个
                
                # 显示当前奖励组成（调试用）
                recent_rewards = {aid: rewards.get(aid, 0) for aid in list(rewards.keys())[:2]}
                print(f"    Recent rewards sample: {recent_rewards}")
                print(f"    Episode time: {env.current_time:.1f}s, Time penalty={config.REWARD_TIME_STEP}")
                print("-" * 60)

        # 回合结束统计（优化版）
        episode_rewards.append(current_episode_reward)
        detection_stats.append(episode_detected_count)
        
        # 获取终止信息
        termination_reason = info.get('termination_reason', 'unknown')
        detection_rate = info.get('detection_rate', 0)
        total_spawned = info.get('total_spawned', 0)
        
        # 回合摘要
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {current_episode_reward:.1f}")
        print(f"  Targets Detected: {episode_detected_count}/{total_spawned} ({detection_rate:.1%})")
        print(f"  Steps: {episode_steps}")
        print(f"  Episode Time: {env.current_time:.1f}s")
        print(f"  Termination Reason: {termination_reason}")
        print(f"  Average Reward/Step: {current_episode_reward/episode_steps:.2f}")
        
        # 每10回合显示训练进度
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_detections = detection_stats[-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_detections = sum(recent_detections) / len(recent_detections)
            
            print(f"\n{'='*50}")
            print(f"MADDPG TRAINING PROGRESS - Episode {episode + 1}")
            print(f"{'='*50}")
            print(f"Last 10 episodes average reward: {avg_reward:.1f}")
            print(f"Last 10 episodes average detections: {avg_detections:.1f}")
            print(f"Best episode reward so far: {max(episode_rewards):.1f}")
            print(f"Best detection count so far: {max(detection_stats)}")
            
            # 学习趋势分析
            if len(episode_rewards) >= 20:
                first_half = sum(episode_rewards[-20:-10]) / 10
                second_half = sum(episode_rewards[-10:]) / 10
                trend = "↗️ Improving" if second_half > first_half else "↘️ Declining" if second_half < first_half else "→ Stable"
                print(f"Learning trend: {trend} ({second_half:.1f} vs {first_half:.1f})")
            
            # 网络训练状态
            if len(maddpg.agents[0].actor_loss_history) > 10:
                recent_actor_loss = np.mean([agent.actor_loss_history[-10:] for agent in maddpg.agents if len(agent.actor_loss_history) >= 10])
                recent_critic_loss = np.mean([agent.critic_loss_history[-10:] for agent in maddpg.agents if len(agent.critic_loss_history) >= 10])
                print(f"Recent training losses: Actor={recent_actor_loss:.4f}, Critic={recent_critic_loss:.1f}")
                
            print(f"Buffer size: {len(maddpg.replay_buffer)}")
            print(f"{'='*50}\n")
    
    # 训练完成摘要
    print("\n" + "🎯 TRAINING COMPLETED 🎯".center(60, "="))
    print(f"Total episodes: {num_episodes}")
    print(f"Best episode reward: {max(episode_rewards):.1f}")
    print(f"Final 10 episodes average: {sum(episode_rewards[-10:]) / 10:.1f}")
    print(f"Best detection performance: {max(detection_stats)} targets")
    print(f"Average detections (final 10): {sum(detection_stats[-10:]) / 10:.1f}")
    print("=" * 60)
    
    env.close()
    print("Simulation finished.")

if __name__ == "__main__":
    main()