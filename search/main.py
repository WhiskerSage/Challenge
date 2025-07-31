# main.py
import pygame
import numpy as np
from environment import UsvUavEnv
from ippo import PPOAgent
import config

def adjust_difficulty(episode):
    """
    课程学习：根据训练进度动态调整难度
    """
    if episode < 100:
        # 初期：更多正奖励，减少惩罚，鼓励探索
        config.REWARD_TIME_STEP = -0.005
        config.REWARD_EXPLORE = 2.0
        config.REWARD_COLLISION = -10.0
        config.REWARD_OUT_OF_BOUNDS = -5.0
        print(f"    [Curriculum] Easy mode: Enhanced exploration rewards")
    elif episode < 300:
        # 中期：标准设置
        config.REWARD_TIME_STEP = -0.01
        config.REWARD_EXPLORE = 1.0
        config.REWARD_COLLISION = -20.0
        config.REWARD_OUT_OF_BOUNDS = -10.0
        if episode == 100:
            print(f"    [Curriculum] Standard mode: Balanced rewards")
    else:
        # 后期：更严格的要求，追求效率
        config.REWARD_TIME_STEP = -0.02
        config.REWARD_EXPLORE = 0.8
        config.REWARD_COLLISION = -30.0
        config.REWARD_OUT_OF_BOUNDS = -15.0
        if episode == 300:
            print(f"    [Curriculum] Hard mode: Efficiency focused")

def main():
    # 选择动作类型：'discrete' 或 'continuous'  
    action_type = 'discrete'  # 暂时切换到离散动作，调试完成后再改回连续
    env = UsvUavEnv(action_type=action_type)
    
    # 只有启用渲染时才初始化时钟
    if config.ENABLE_RENDERING:
        clock = pygame.time.Clock()
    else:
        clock = None
        print("训练模式：已禁用可视化渲染以提高训练速度")

    # --- 1. IPPO智能体初始化 ---
    # 将ppo_agents的初始化延迟到reset之后，以确保能获取到正确的agent列表
    ppo_agents = {}
    is_ppo_agents_initialized = False

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

        # 仅在第一个回合初始化PPO智能体
        if not is_ppo_agents_initialized:
            state_dim = env.observation_space.shape[0]
            if action_type == 'continuous':
                action_dim = env.action_space.shape[0]
            else:
                action_dim = env.action_space.n
            
            ppo_agents = {agent.id: PPOAgent(state_dim, action_dim, config.LR_ACTOR, 
                                             config.LR_CRITIC, config.GAMMA, 
                                             config.K_EPOCHS, config.EPS_CLIP, action_type)
                          for agent in env.agents}
            is_ppo_agents_initialized = True

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

            # --- 3. 算法决策与数据收集 ---
            actions = {}
            for agent_id, obs in observations.items():
                # 从对应的PPOAgent获取动作
                action = ppo_agents[agent_id].select_action(obs)
                actions[agent_id] = action
                
                # 调试：偶尔打印动作信息
                if time_step % 5000 == 0 and agent_id == 'usv_0':
                    print(f"    DEBUG: {agent_id} action = {action} (type: {type(action)})")
                    if hasattr(env.agents[0], 'pos'):
                        print(f"    DEBUG: {agent_id} position = {env.agents[0].pos}")
                        print(f"    DEBUG: {agent_id} velocity = {env.agents[0].velocity}")
                        print(f"    DEBUG: Action type = {action_type}")

            # --- 环境交互 ---
            next_observations, rewards, terminated, truncated, info = env.step(actions)

            # --- 4. 将经验存入每个智能体的Buffer ---
            for agent_id, ppo_agent in ppo_agents.items():
                ppo_agent.buffer.rewards.append(rewards[agent_id])
                # 对IPPO来说，每个智能体都将全局的done信号作为自己的终止信号
                ppo_agent.buffer.is_terminals.append(terminated or truncated)

            observations = next_observations
            current_episode_reward += sum(rewards.values())
            
            # 统计检测事件
            if 'detected_events' in info:
                episode_detected_count += len([event for event in info['detected_events'] 
                                             if 'successfully detected' in event])

            # --- 5. 模型更新 ---
            if time_step % config.UPDATE_TIMESTEPS == 0:
                print(f"    UPDATING at time_step {time_step}...")
                for agent_id, ppo_agent in ppo_agents.items():
                    ppo_agent.update()
                print(f"    UPDATE finished.")
            
            # --- 训练监控（详细版）---
            if config.ENABLE_RENDERING:
                env.render()
                clock.tick(config.TARGET_FPS)
            elif time_step % 1000 == 0:  # 无渲染模式下的详细监控
                avg_reward = current_episode_reward / episode_steps if episode_steps > 0 else 0
                print(f"    Step {time_step}, Episode Steps: {episode_steps}, Avg Reward: {avg_reward:.3f}, Episode Reward: {current_episode_reward:.1f}")
                
                # 检查当前检测状态
                active_detections = sum(1 for target in env.targets 
                                      if hasattr(target, 'detection_completed') and target.detection_completed)
                total_targets = len(env.targets)
                print(f"    Targets: {active_detections}/{total_targets} detected, Episode total: {episode_detected_count}")
                
                # 显示智能体状态（包含所有智能体）
                all_agents = [(agent.id, agent.pos) for agent in env.agents]
                print(f"    All agents: {[(aid, f'({pos[0]:.0f},{pos[1]:.0f})') for aid, pos in all_agents]}")
                print(f"    Total agents: {len(env.agents)} (should be {config.TOTAL_AGENTS})")
                
                # 显示当前奖励组成（调试用）
                recent_rewards = {aid: rewards.get(aid, 0) for aid in list(rewards.keys())[:2]}
                print(f"    Recent rewards sample: {recent_rewards}")
                print(f"    Current curriculum: Time penalty={config.REWARD_TIME_STEP}, Explore reward={config.REWARD_EXPLORE}")
                print(f"    Episode time: {env.current_time:.1f}s")
                print("-" * 60)
                
                # 早停检查：如果步数过多，强制结束回合
                if episode_steps > 50000:  # 50000步后强制结束
                    print(f"    [EARLY STOP] Episode too long, forcing termination")
                    break

        # 回合结束统计
        episode_rewards.append(current_episode_reward)
        detection_stats.append(episode_detected_count)
        
        # 回合摘要
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {current_episode_reward:.1f}")
        print(f"  Targets Detected: {episode_detected_count}")
        print(f"  Steps: {episode_steps}")
        
        # 每10回合显示训练进度
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_detections = detection_stats[-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_detections = sum(recent_detections) / len(recent_detections)
            
            print(f"\n{'='*50}")
            print(f"TRAINING PROGRESS - Episode {episode + 1}")
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