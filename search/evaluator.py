# evaluator.py - 标准化性能评估模块
import numpy as np
import config

class PerformanceEvaluator:
    """
    标准化的性能评估器，为算法开发者提供统一的评估指标
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置评估器状态"""
        self.episode_data = {
            'detected_targets': [],
            'detection_times': [],
            'agent_trajectories': {},
            'collision_count': 0,
            'total_steps': 0,
            'total_reward': 0
        }
        
        for i in range(config.NUM_UAVS):
            self.episode_data['agent_trajectories'][f'uav_{i}'] = []
        for i in range(config.NUM_USVS):
            self.episode_data['agent_trajectories'][f'usv_{i}'] = []
    
    def update_step(self, agents, targets, rewards, detected_events):
        """更新单步数据"""
        self.episode_data['total_steps'] += 1
        self.episode_data['total_reward'] += sum(rewards.values())
        
        # 记录智能体位置
        for agent in agents:
            self.episode_data['agent_trajectories'][agent.id].append(agent.pos.copy())
        
        # 记录检测事件
        for event in detected_events:
            if 'successfully detected' in event:
                target_id = event.split()[-1]
                self.episode_data['detected_targets'].append(target_id)
                self.episode_data['detection_times'].append(self.episode_data['total_steps'])
    
    def update_collision(self):
        """记录碰撞事件"""
        self.episode_data['collision_count'] += 1
    
    def get_episode_metrics(self, total_targets_spawned):
        """计算回合结束时的性能指标"""
        metrics = {}
        
        # 1. 检测成功率
        metrics['detection_success_rate'] = len(self.episode_data['detected_targets']) / max(total_targets_spawned, 1)
        
        # 2. 平均检测时间
        if self.episode_data['detection_times']:
            metrics['average_detection_time'] = np.mean(self.episode_data['detection_times']) * config.SIM_TIME_STEP
        else:
            metrics['average_detection_time'] = float('inf')
        
        # 3. 区域覆盖率（简化计算）
        all_positions = []
        for trajectory in self.episode_data['agent_trajectories'].values():
            all_positions.extend(trajectory)
        
        if all_positions:
            unique_grids = set()
            for pos in all_positions:
                grid_x = int(pos[0] / config.EXPLORATION_GRID_SIZE)
                grid_y = int(pos[1] / config.EXPLORATION_GRID_SIZE)
                unique_grids.add((grid_x, grid_y))
            
            total_grids = (config.AREA_WIDTH_METERS / config.EXPLORATION_GRID_SIZE) * \
                         (config.AREA_HEIGHT_METERS / config.EXPLORATION_GRID_SIZE)
            metrics['area_coverage_rate'] = len(unique_grids) / total_grids
        else:
            metrics['area_coverage_rate'] = 0.0
        
        # 4. 移动效率（总移动距离）
        total_distance = 0
        for trajectory in self.episode_data['agent_trajectories'].values():
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    total_distance += np.linalg.norm(trajectory[i] - trajectory[i-1])
        metrics['total_movement_distance'] = total_distance
        
        # 5. 碰撞次数
        metrics['collision_count'] = self.episode_data['collision_count']
        
        # 6. 总奖励
        metrics['total_reward'] = self.episode_data['total_reward']
        
        # 7. 回合长度
        metrics['episode_length'] = self.episode_data['total_steps']
        
        return metrics
    
    def print_summary(self, metrics):
        """打印性能摘要"""
        print("\n" + "="*50)
        print("EPISODE PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Detection Success Rate: {metrics['detection_success_rate']:.2%}")
        print(f"Average Detection Time: {metrics['average_detection_time']:.1f}s")
        print(f"Area Coverage Rate: {metrics['area_coverage_rate']:.2%}")
        print(f"Total Movement: {metrics['total_movement_distance']:.0f}m")
        print(f"Collisions: {metrics['collision_count']}")
        print(f"Total Reward: {metrics['total_reward']:.1f}")
        print(f"Episode Length: {metrics['episode_length']} steps")
        print("="*50)

class BenchmarkRunner:
    """
    基准测试运行器，为算法开发者提供标准化的测试流程
    """
    def __init__(self, env, agent_class, num_episodes=100):
        self.env = env
        self.agent_class = agent_class
        self.num_episodes = num_episodes
        self.evaluator = PerformanceEvaluator()
    
    def run_benchmark(self):
        """运行基准测试"""
        all_metrics = []
        
        for episode in range(self.num_episodes):
            self.evaluator.reset()
            obs, _ = self.env.reset()
            
            # 初始化智能体
            agents = {}
            for agent_id in obs.keys():
                agents[agent_id] = self.agent_class(
                    state_dim=self.env.observation_space.shape[0],
                    action_dim=self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n,
                    action_type='continuous' if hasattr(self.env.action_space, 'shape') else 'discrete'
                )
            
            done = False
            targets_spawned = 0
            
            while not done:
                # 选择动作
                actions = {}
                for agent_id, observation in obs.items():
                    actions[agent_id] = agents[agent_id].select_action(observation)
                
                # 环境步进
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                # 更新评估器
                if 'detected_events' in info:
                    self.evaluator.update_step(self.env.agents, self.env.targets, rewards, info['detected_events'])
                
                # 计算生成的目标数（简化）
                targets_spawned = max(targets_spawned, len(self.env.targets))
            
            # 计算回合指标
            episode_metrics = self.evaluator.get_episode_metrics(targets_spawned)
            all_metrics.append(episode_metrics)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Success Rate = {episode_metrics['detection_success_rate']:.2%}")
        
        return self._compute_benchmark_stats(all_metrics)
    
    def _compute_benchmark_stats(self, all_metrics):
        """计算基准测试统计数据"""
        stats = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isinf(m[key])]
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
        
        return stats