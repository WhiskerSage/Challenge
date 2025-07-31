# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame # 用于可视化

import config
from agents import UAV, USV, Target
# from strategy import generate_lawnmower_path # 不再需要

class UsvUavEnv(gym.Env):
    """
    机艇协同搜索竞赛的强化学习环境
    """
    # 在 __init__ 方法中更新观测空间
    # 连续动作空间版本
    def __init__(self, action_type='discrete'):
        super(UsvUavEnv, self).__init__()
        
        self.action_type = action_type
        
        # 1. 定义动作空间 (Action Space)
        if action_type == 'continuous':
            # 连续动作: [速度比例, 转向角度]
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0]),   # [速度0-1, 转向-1到1]
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # 扩展的离散动作: 11个动作（增加了更精细的控制）
            self.action_space = spaces.Discrete(11)
    
        # 2. 定义扩展的观测空间 (Observation Space)
        # 自身状态(6) + 最近目标信息(15) + 其他智能体(20) + 探索状态(9) + 目标优先级(3) + 协同信息(4) = 57维
        obs_dim = 6 + 15 + 20 + 9 + 3 + 4  # 增加协同信息
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # 3. 初始化pygame用于可视化（可选）
        if config.ENABLE_RENDERING:
            pygame.init()
            self.screen_size = 800 # 屏幕像素大小
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("USV-UAV Cooperative Search")
            self.font = pygame.font.SysFont(None, 24)
        else:
            self.screen = None
            self.font = None
        
        # 4. 创建智能体和目标列表
        self.uavs = []
        self.usvs = []
        self.agents = [] # 统一管理
        self.targets = []
        self.detection_timers = {} # 用于记录目标被探测的持续时间
        self.next_target_id = 0
        self.current_time = 0  # 添加时间跟踪
        
        # 定义探索网格的形状
        self.grid_shape = (int(config.AREA_HEIGHT_METERS / config.EXPLORATION_GRID_SIZE),
                           int(config.AREA_WIDTH_METERS / config.EXPLORATION_GRID_SIZE))
        # exploration_grids 将在 reset 方法中被完全初始化
        self.exploration_grids = {}
        
        # 新增: 用于存储每个智能体上一步到最近目标的距离
        self.previous_distances = {}

    def _meters_to_pixels(self, pos_meters):
        """将米制坐标转换为屏幕像素坐标"""
        px = pos_meters[0] * (self.screen_size / config.AREA_WIDTH_METERS)
        py = pos_meters[1] * (self.screen_size / config.AREA_HEIGHT_METERS)
        return int(px), int(py)

    def _spawn_targets(self, max_to_spawn=2):
        """
        基于课程学习策略生成目标
        融合传统算法的边界目标优先策略
        """
        # 计算当前可以生成的名额
        num_can_spawn = max_to_spawn - self._count_targets_on_edge()
        if num_can_spawn <= 0:
            return

        for _ in range(np.random.randint(1, num_can_spawn + 1)):
            # 根据课程学习参数确定目标类型
            boundary_prob = getattr(config, 'TARGET_BOUNDARY_PROB', 0.5)
            speed_range = getattr(config, 'TARGET_SPEED_RANGE', (5, 15))
            
            if np.random.random() < boundary_prob:
                # 边界目标生成
                edge = np.random.randint(4)
                if edge == 0: # 上边
                    pos = [np.random.uniform(0, config.AREA_WIDTH_METERS), config.AREA_HEIGHT_METERS - 50]
                    heading = np.random.uniform(-np.pi * 0.75, -np.pi * 0.25) # 朝向下方
                elif edge == 1: # 右边
                    pos = [config.AREA_WIDTH_METERS - 50, np.random.uniform(0, config.AREA_HEIGHT_METERS)]
                    heading = np.random.uniform(np.pi * 0.75, np.pi * 1.25) # 朝向左方
                elif edge == 2: # 下边
                    pos = [np.random.uniform(0, config.AREA_WIDTH_METERS), 50]
                    heading = np.random.uniform(np.pi * 0.25, np.pi * 0.75) # 朝向上方
                else: # 左边
                    pos = [50, np.random.uniform(0, config.AREA_HEIGHT_METERS)]
                    heading = np.random.uniform(-np.pi * 0.25, np.pi * 0.25) # 朝向右方
            else:
                # 中心区域目标
                pos = [
                    np.random.uniform(config.AREA_WIDTH_METERS * 0.2, config.AREA_WIDTH_METERS * 0.8),
                    np.random.uniform(config.AREA_HEIGHT_METERS * 0.2, config.AREA_HEIGHT_METERS * 0.8)
                ]
                heading = np.random.uniform(0, 2 * np.pi)
            
            # 根据课程学习设置速度
            speed_knots = np.random.uniform(*speed_range)
            speed_mps = speed_knots * config.KNOTS_TO_MPS
            velocity = [speed_mps * np.cos(heading), speed_mps * np.sin(heading)]
                
            target = Target(f"tgt_{self.next_target_id}", pos, heading)
            target.velocity = np.array(velocity)
            target.spawn_time = self.current_time  # 记录生成时间
            self.targets.append(target)
            self.next_target_id += 1
    
    def _count_targets_on_edge(self, edge_thickness=10.0):
        """计算位于边界附近的目标数量"""
        count = 0
        for t in self.targets:
            if (t.pos[0] < edge_thickness or 
                t.pos[0] > config.AREA_WIDTH_METERS - edge_thickness or
                t.pos[1] < edge_thickness or
                t.pos[1] > config.AREA_HEIGHT_METERS - edge_thickness):
                count += 1
        return count

    def _get_grid_coords(self, pos_meters):
        """将物理坐标转换为网格坐标"""
        x = int(pos_meters[0] / config.EXPLORATION_GRID_SIZE)
        y = int(pos_meters[1] / config.EXPLORATION_GRID_SIZE)
        return min(x, self.grid_shape[1]-1), min(y, self.grid_shape[0]-1)

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        # 1. 清空智能体和目标
        self.uavs.clear()
        self.usvs.clear()
        self.agents.clear()
        self.targets.clear()
        self.detection_timers.clear()
        self.next_target_id = 0
        self.current_time = 0
        self.previous_distances.clear()
        
        # 重置探索网格 (在agents创建后)
        self.exploration_grids = {agent.id: np.zeros(self.grid_shape) for agent in self.agents}

        # 2. 根据赛题规则重新创建智能体
        # 严格按照赛题要求：AB边中段，中间两个无人艇间距的中点和两个无人机间距的中点为边AB中点
        center_y = config.AREA_HEIGHT_METERS / 2.0
        
        # 创建USVs - 使用配置的间距参数
        # "中间两个无人艇间距的中点为边AB中点"
        usv_positions = []
        for i in range(config.NUM_USVS):
            # 计算Y坐标：以中心为基准，向两侧分布
            y_offset = (i - 1.5) * config.USV_SPACING_METERS
            pos = [0, center_y + y_offset]
            usv_positions.append(pos)
            usv = USV(agent_id=f"usv_{i}", initial_pos=pos)
            self.usvs.append(usv)
            self.agents.append(usv)

        # 创建UAVs - 使用配置的间距参数
        # "两个无人机间距的中点为边AB中点"
        uav_positions = []
        for i in range(config.NUM_UAVS):
            # 计算Y坐标：以中心为基准，向两侧分布
            y_offset = (i - 0.5) * config.UAV_SPACING_METERS
            pos = [0, center_y + y_offset]
            uav_positions.append(pos)
            uav = UAV(agent_id=f"uav_{i}", initial_pos=pos)
            self.uavs.append(uav)
            self.agents.append(uav)
        
        # 在创建完所有agent后，再初始化依赖agent id的字典
        self.exploration_grids = {agent.id: np.zeros(self.grid_shape) for agent in self.agents}
        
        # 2.5. 为智能体分配搜索区域并生成路径点 (移除)
        # 不再需要，将由强化学习自主决定路径
        
        # 3. 创建初始目标（示例）
        self._spawn_targets(max_to_spawn=config.TARGET_MAX_COUNT_AT_ONCE)
        
        # 4. 返回初始观测
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, actions):
        """
        环境步进函数
        actions: 一个字典，key为agent_id, value为离散动作
        """
        # 更新时间
        self.current_time += config.SIM_TIME_STEP
        
        # 1. 更新智能体位置
        for agent in self.agents:
            if agent.id in actions:
                agent.move(actions[agent.id], config.SIM_TIME_STEP)
        
        # 1.5 更新协同信息（新增）
        for agent in self.agents:
            agent.update_coordination_info(self.agents, self.targets, self.current_time)
            agent.get_coordination_role(self.targets, self.agents)
        
        # 2. 更新目标位置，并移除出界的
        for target in self.targets:
            target.move(config.SIM_TIME_STEP)
        
        self.targets = [t for t in self.targets if 
                        0 <= t.pos[0] <= config.AREA_WIDTH_METERS and 
                        0 <= t.pos[1] <= config.AREA_HEIGHT_METERS]
        
        # 随机生成新目标（使用课程学习策略）
        generation_rate = getattr(config, 'TARGET_GENERATION_RATE', 0.002)
        if np.random.rand() < generation_rate:
            self._spawn_targets(max_to_spawn=config.TARGET_MAX_COUNT_AT_ONCE)

        # 3. 计算奖励和执行检测
        rewards, detected_events = self._calculate_rewards_and_detections()
        
        # 4. 获取下一步的观测
        observation = self._get_observation()
        
        # 5. 检查结束条件
        terminated = False
        truncated = False
        
        # 添加回合终止条件
        if self.current_time > 600:  # 10分钟超时
            truncated = True
        
        # 如果所有当前目标都被检测完成，可以选择终止回合
        active_targets = [t for t in self.targets if not (hasattr(t, 'detection_completed') and t.detection_completed)]
        if len(active_targets) == 0 and self.current_time > 60:  # 至少运行1分钟后才能因无目标终止
            terminated = True
        
        info = {'detected_events': detected_events} # 可以把探测事件等信息放进去
        
        return observation, rewards, terminated, truncated, info

    def _calculate_rewards_and_detections(self):
        """
        修正后的检测和奖励计算：检测成功后目标完成任务，无需跟踪
        """
        rewards = {agent.id: config.REWARD_TIME_STEP for agent in self.agents}
        detected_events = []

        # --- 1. 出界检测与惩罚（改进版）---
        for agent in self.agents:
            # 移动激励奖励
            speed_ratio = np.linalg.norm(agent.velocity) / agent.max_speed
            rewards[agent.id] += config.REWARD_MOVEMENT * speed_ratio
            
            x, y = agent.pos
            margin = 200  # 增加边界缓冲区到200米
            out_of_bounds = not (margin <= x <= config.AREA_WIDTH_METERS - margin and 
                               margin <= y <= config.AREA_HEIGHT_METERS - margin)
            
            if out_of_bounds:
                rewards[agent.id] += config.REWARD_OUT_OF_BOUNDS
                # 更温和的边界处理：推向安全区域而不是硬性限制
                safe_margin = margin + 100
                
                if x < margin:
                    # 推向右侧，但保持一定的原有速度分量
                    push_force = (safe_margin - x) / safe_margin  # 0-1的推力
                    agent.pos[0] = margin + 10
                    agent.velocity[0] = abs(agent.velocity[0]) * 0.5 + agent.max_speed * push_force * 0.3
                elif x > config.AREA_WIDTH_METERS - margin:
                    push_force = (x - (config.AREA_WIDTH_METERS - margin)) / margin
                    agent.pos[0] = config.AREA_WIDTH_METERS - margin - 10
                    agent.velocity[0] = -abs(agent.velocity[0]) * 0.5 - agent.max_speed * push_force * 0.3
                    
                if y < margin:
                    push_force = (safe_margin - y) / safe_margin
                    agent.pos[1] = margin + 10
                    agent.velocity[1] = abs(agent.velocity[1]) * 0.5 + agent.max_speed * push_force * 0.3
                elif y > config.AREA_HEIGHT_METERS - margin:
                    push_force = (y - (config.AREA_HEIGHT_METERS - margin)) / margin
                    agent.pos[1] = config.AREA_HEIGHT_METERS - margin - 10
                    agent.velocity[1] = -abs(agent.velocity[1]) * 0.5 - agent.max_speed * push_force * 0.3

        # --- 2. 碰撞检测与惩罚 ---
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1 = self.agents[i]
                agent2 = self.agents[j]
                dist = np.linalg.norm(agent1.pos - agent2.pos)
                if dist < (agent1.collision_radius + agent2.collision_radius):
                    rewards[agent1.id] += config.REWARD_COLLISION
                    rewards[agent2.id] += config.REWARD_COLLISION

        # --- 3. 区域探索奖励（增强版）---
        for agent in self.agents:
            grid_x, grid_y = self._get_grid_coords(agent.pos)
            if self.exploration_grids[agent.id][grid_y, grid_x] == 0:
                self.exploration_grids[agent.id][grid_y, grid_x] = 1
                rewards[agent.id] += config.REWARD_EXPLORE
                
                # 高效搜索奖励：奖励探索未被其他智能体访问过的区域
                is_unexplored_by_others = all(
                    self.exploration_grids[other_agent.id][grid_y, grid_x] == 0 
                    for other_agent in self.agents if other_agent.id != agent.id
                )
                if is_unexplored_by_others:
                    rewards[agent.id] += config.REWARD_EFFICIENT_SEARCH
        
        # --- 3.5 区域覆盖奖励 ---
        # 定期给予区域覆盖奖励
        if int(self.current_time * 10) % 50 == 0:  # 每5秒计算一次
            for agent in self.agents:
                coverage_ratio = np.sum(self.exploration_grids[agent.id]) / (self.grid_shape[0] * self.grid_shape[1])
                coverage_reward = config.REWARD_AREA_COVERAGE * coverage_ratio
                rewards[agent.id] += coverage_reward
        
        # --- 3.6 新增：接近目标的奖励（增强版）---
        active_targets = [t for t in self.targets if not (hasattr(t, 'detection_completed') and t.detection_completed)]
        if active_targets:
            for agent in self.agents:
                # 找到最近的未完成目标
                min_dist = float('inf')
                closest_target = None
                for target in active_targets:
                    dist = np.linalg.norm(agent.pos - target.pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_target = target

                if closest_target:
                    # 获取上一时间步的距离
                    prev_dist = self.previous_distances.get(agent.id, min_dist)
                    
                    # 如果距离变近，给予奖励
                    if min_dist < prev_dist:
                        rewards[agent.id] += config.REWARD_APPROACHING_TARGET
                        
                        # 额外奖励：如果是边界或高速目标
                        boundary_dist = min(
                            closest_target.pos[0] - 0, 
                            config.AREA_WIDTH_METERS - closest_target.pos[0],
                            closest_target.pos[1] - 0, 
                            config.AREA_HEIGHT_METERS - closest_target.pos[1]
                        )
                        target_speed = np.linalg.norm(closest_target.velocity) if hasattr(closest_target, 'velocity') else 0
                        
                        if boundary_dist < 200 or target_speed > 5.14:
                            rewards[agent.id] += config.REWARD_TARGET_TRACKING

        # --- 3.7 协同行为奖励（新增）---
        for agent in self.agents:
            # 角色执行奖励
            if agent.coordination_role == 'tracker':
                # 跟踪高优先级目标的奖励
                high_priority_count = sum(1 for info in agent.shared_target_info.values() 
                                        if info['priority'] > 0.7)
                if high_priority_count > 0:
                    rewards[agent.id] += config.REWARD_TARGET_TRACKING
            
            elif agent.coordination_role == 'interceptor':
                # 拦截模式下接近目标的奖励
                nearby_targets = sum(1 for info in agent.shared_target_info.values()
                                   if np.linalg.norm(agent.pos - info['pos']) < 500)  # 500米内
                if nearby_targets > 0:
                    rewards[agent.id] += config.REWARD_TARGET_TRACKING * 2
            
            # 信息共享奖励
            shared_info_count = len(agent.shared_target_info)
            if shared_info_count > 0:
                rewards[agent.id] += config.REWARD_COVERAGE_EFFICIENCY * min(shared_info_count, 3) / 3

        # --- 4. 目标探测逻辑与奖励（修正版：必须持续10秒在探测范围内）---
        successfully_detected_targets = []
        
        for target in self.targets:
            # 已经检测成功的目标不再处理
            if hasattr(target, 'detection_completed') and target.detection_completed:
                continue
                
            # 检查当前有哪些智能体能探测到此目标
            agents_currently_detecting = []
            for agent in self.agents:
                is_in_range = self._is_target_in_agent_range(agent, target)
                if is_in_range:
                    agents_currently_detecting.append(agent)
            
            # 处理每个智能体的探测计时
            for agent in self.agents:
                timer_key = (agent.id, target.id)
                is_currently_detecting = agent in agents_currently_detecting
                
                if is_currently_detecting:
                    # 在探测范围内，累积时间
                    self.detection_timers[timer_key] = self.detection_timers.get(timer_key, 0) + config.SIM_TIME_STEP
                    
                    # 检查是否达到检测成功条件（持续10秒）
                    if self.detection_timers[timer_key] >= config.DETECTION_SUSTAIN_TIME:
                        if not target.is_detected:  # 首次达到检测条件
                            target.is_detected = True
                            target.detection_completed = True  # 标记为检测完成
                            if target not in successfully_detected_targets:
                                successfully_detected_targets.append(target)
                else:
                    # 不在探测范围内，重置该智能体对此目标的计时器
                    if timer_key in self.detection_timers:
                        self.detection_timers[timer_key] = 0
        
        # --- 5. 为成功检测目标的智能体分配奖励（增强版）---
        if successfully_detected_targets:
            for target in successfully_detected_targets:
                participating_agents = []
                # 找出所有对此目标达到10秒检测时间的智能体
                for agent in self.agents:
                    timer_key = (agent.id, target.id)
                    if self.detection_timers.get(timer_key, 0) >= config.DETECTION_SUSTAIN_TIME:
                        participating_agents.append(agent)
                        
                        # 基础检测奖励
                        rewards[agent.id] += config.REWARD_DETECT
                        
                        # 边界目标额外奖励
                        boundary_dist = min(
                            target.pos[0] - 0, 
                            config.AREA_WIDTH_METERS - target.pos[0],
                            target.pos[1] - 0, 
                            config.AREA_HEIGHT_METERS - target.pos[1]
                        )
                        if boundary_dist < 200:
                            rewards[agent.id] += config.REWARD_BOUNDARY_TARGET
                        
                        # 高速目标额外奖励
                        target_speed = np.linalg.norm(target.velocity) if hasattr(target, 'velocity') else 0
                        if target_speed > 5.14:  # >10节
                            rewards[agent.id] += config.REWARD_HIGH_SPEED_TARGET
                        
                        # 早期发现奖励（5分钟内发现）
                        detection_time = self.current_time - getattr(target, 'spawn_time', self.current_time - 300)
                        if detection_time < 300:  # 5分钟
                            rewards[agent.id] += config.REWARD_EARLY_DETECTION
                
                # 如果有多个智能体同时参与检测，给予协同奖励
                if len(participating_agents) > 1:
                    for agent in participating_agents:
                        rewards[agent.id] += config.REWARD_COORDINATION_DETECT
                
                # 记录检测事件
                agent_names = [agent.id for agent in participating_agents]
                detected_events.append(f"Target {target.id} successfully detected by: {', '.join(agent_names)}")
        
        return rewards, detected_events
    
    def _is_target_in_agent_range(self, agent, target):
        """
        优化的探测范围检查 - 融合传统算法的边界目标增强策略
        关键改进：
        1. 边界目标探测范围增强
        2. 高速目标探测优化
        3. 动态探测参数调整
        """
        # 计算目标到边界的距离
        boundary_dist = min(
            target.pos[0] - 0, 
            config.AREA_WIDTH_METERS - target.pos[0],
            target.pos[1] - 0, 
            config.AREA_HEIGHT_METERS - target.pos[1]
        )
        
        # 边界目标探测范围增强（距离边界<200米时增强50%）
        boundary_boost = 1.0 + max(0, (200 - min(200, boundary_dist)) / 200 * 0.5)
        
        # 计算目标速度
        target_speed = np.linalg.norm(target.velocity) if hasattr(target, 'velocity') else 0
        
        # 高速目标探测范围增强（>10节时增强20%）
        speed_boost = 1.2 if target_speed > 5.14 else 1.0  # 5.14 m/s ≈ 10节
        
        # 计算有效探测范围
        effective_range = agent.sensor_range * boundary_boost * speed_boost
        
        # 基础距离检查
        dist = np.linalg.norm(agent.pos - target.pos)
        if dist > effective_range:
            return False
        
        if isinstance(agent, UAV):
            # 无人机扇形探测，对边界和高速目标扩大角度
            vec_agent_target = target.pos - agent.pos
            angle_to_target = np.arctan2(vec_agent_target[1], vec_agent_target[0])
            angle_diff = (angle_to_target - agent.heading + np.pi) % (2 * np.pi) - np.pi
            
            # 基础FOV角度
            base_fov = config.UAV_SENSOR_FOV_DEG / 2.0 * np.pi / 180.0
            
            # 边界目标或高速目标扩大FOV角度
            if boundary_dist < 200 or target_speed > 5.14:
                enhanced_fov = base_fov * 1.3  # 扩大30%的视场角
            else:
                enhanced_fov = base_fov
                
            return abs(angle_diff) <= enhanced_fov
        else: # USV
            return True # USV为圆形探测，已通过距离检查

    def _calculate_rewards(self): # 这个函数现在不再直接使用，逻辑已合并
        pass

    def render(self):
        """可视化当前环境状态（可选渲染）"""
        if not config.ENABLE_RENDERING:
            return  # 无渲染模式，直接返回
            
        self.screen.fill(config.COLOR_BACKGROUND)
        
        # (可选) 绘制探索网格以供调试
        for y in range(self.grid_shape[0]):
            for x in range(self.grid_shape[1]):
                is_explored_by_any = any(self.exploration_grids[agent.id][y, x] for agent in self.agents)
                if is_explored_by_any:
                    rect = pygame.Rect(x * config.EXPLORATION_GRID_SIZE * (self.screen_size / config.AREA_WIDTH_METERS),
                                       y * config.EXPLORATION_GRID_SIZE * (self.screen_size / config.AREA_HEIGHT_METERS),
                                       config.EXPLORATION_GRID_SIZE * (self.screen_size / config.AREA_WIDTH_METERS),
                                       config.EXPLORATION_GRID_SIZE * (self.screen_size / config.AREA_HEIGHT_METERS))
                    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    s.fill((0, 255, 0, 30)) # 半透明绿色
                    self.screen.blit(s, rect.topleft)

        # 绘制所有智能体和传感器范围
        for agent in self.agents:
            pixel_pos = self._meters_to_pixels(agent.pos)
            color = config.COLOR_UAV if isinstance(agent, UAV) else config.COLOR_USV
            
            # 绘制传感器范围
            if isinstance(agent, UAV):
                # 绘制扇形
                self._draw_fov(self.screen, color, pixel_pos, agent.sensor_range, agent.heading, config.UAV_SENSOR_FOV_DEG)
            else:
                # 绘制圆形
                pixel_range = agent.sensor_range * (self.screen_size / config.AREA_WIDTH_METERS)
                pygame.draw.circle(self.screen, config.COLOR_SENSOR_RANGE, pixel_pos, int(pixel_range), 1)

            # 绘制智能体
            pygame.draw.circle(self.screen, color, pixel_pos, 5)
            heading_end_pos = (pixel_pos[0] + 10 * np.cos(agent.heading), pixel_pos[1] + 10 * np.sin(agent.heading))
            pygame.draw.line(self.screen, color, pixel_pos, heading_end_pos, 2)
            id_text = self.font.render(agent.id, True, config.COLOR_TEXT)
            self.screen.blit(id_text, (pixel_pos[0] + 8, pixel_pos[1]))

            # 移除：绘制当前路径点 (因为已改用RL)
            # waypoint = agent.get_current_waypoint()
            # if waypoint is not None:
            #     pixel_wp = self._meters_to_pixels(waypoint)
            #     pygame.draw.circle(self.screen, color, pixel_wp, 8, 1) # 空心圆
            #     pygame.draw.line(self.screen, color, pixel_pos, pixel_wp, 1) # 连线

        # 绘制所有目标
        for target in self.targets:
            pixel_pos = self._meters_to_pixels(target.pos)
            
            # 根据检测状态选择颜色
            if hasattr(target, 'detection_completed') and target.detection_completed:
                color = (0, 255, 0)  # 绿色表示检测完成的目标
            else:
                # 检查是否有智能体正在检测此目标
                max_detection_time = 0
                detecting_agent = None
                for agent in self.agents:
                    timer_key = (agent.id, target.id)
                    detection_time = self.detection_timers.get(timer_key, 0)
                    if detection_time > max_detection_time:
                        max_detection_time = detection_time
                        detecting_agent = agent.id
                
                if max_detection_time > 0:
                    # 根据检测进度选择颜色（从红色渐变到黄色）
                    progress = min(max_detection_time / config.DETECTION_SUSTAIN_TIME, 1.0)
                    red = int(255 * (1 - progress * 0.5))  # 从255渐变到127
                    green = int(255 * progress)  # 从0渐变到255
                    color = (red, green, 0)
                else:
                    color = config.COLOR_TARGET  # 红色表示未检测的目标
                
            pygame.draw.circle(self.screen, color, pixel_pos, 6)
            pygame.draw.line(self.screen, config.COLOR_BLACK, (pixel_pos[0]-4, pixel_pos[1]-4), (pixel_pos[0]+4, pixel_pos[1]+4), 1)
            pygame.draw.line(self.screen, config.COLOR_BLACK, (pixel_pos[0]-4, pixel_pos[1]+4), (pixel_pos[0]+4, pixel_pos[1]-4), 1)
            
            # 显示检测状态和进度
            if hasattr(target, 'detection_completed') and target.detection_completed:
                status_text = self.font.render(f"DONE", True, (255, 255, 255))
                self.screen.blit(status_text, (pixel_pos[0] + 8, pixel_pos[1] + 8))
            elif max_detection_time > 0:
                progress_percent = int((max_detection_time / config.DETECTION_SUSTAIN_TIME) * 100)
                progress_text = self.font.render(f"{progress_percent}%", True, (255, 255, 255))
                self.screen.blit(progress_text, (pixel_pos[0] + 8, pixel_pos[1] + 8))
                
                # 显示正在检测的智能体ID
                agent_text = self.font.render(f"by {detecting_agent}", True, (200, 200, 200))
                self.screen.blit(agent_text, (pixel_pos[0] + 8, pixel_pos[1] + 25))
            
        pygame.display.flip()

    def _draw_fov(self, surface, color, pos, radius, heading, fov_deg):
        """辅助函数，用于绘制扇形FOV"""
        pixel_radius = radius * (self.screen_size / config.AREA_WIDTH_METERS)
        start_angle = heading - np.deg2rad(fov_deg / 2)
        end_angle = heading + np.deg2rad(fov_deg / 2)
        
        # 创建一个点列表来绘制扇形
        points = [pos]
        for n in range(10): # 用10个点来近似圆弧
            angle = start_angle + (end_angle - start_angle) * n / 9
            points.append((pos[0] + pixel_radius * np.cos(angle), pos[1] + pixel_radius * np.sin(angle)))
        points.append(pos)
        
        # 创建一个带有alpha通道的表面来绘制半透明的扇形
        s = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
        pygame.draw.polygon(s, config.COLOR_SENSOR_RANGE, points)
        surface.blit(s, (0,0))
        pygame.draw.aalines(surface, color, True, points, 1)

    def _get_observation(self):
        """
        获取所有智能体的观测数据（增强版）
        融合传统算法的目标优先级评估
        """
        observations = {}
        for agent in self.agents:
            # 1. 自身状态信息 (6维)
            self_obs = [
                agent.pos[0] / config.AREA_WIDTH_METERS,   # 归一化x位置
                agent.pos[1] / config.AREA_HEIGHT_METERS,  # 归一化y位置
                agent.velocity[0] / agent.max_speed,       # 归一化x速度
                agent.velocity[1] / agent.max_speed,       # 归一化y速度
                np.cos(agent.heading),                     # 航向cos
                np.sin(agent.heading)                      # 航向sin
            ]
            
            # 2. 最近目标信息 (15维: 3个目标 × 5维信息)
            nearest_targets = self._get_nearest_targets(agent, n=3)
            target_obs = []
            target_priorities = []
            
            for target in nearest_targets:
                if target is not None:
                    relative_pos = target.pos - agent.pos
                    target_speed = np.linalg.norm(target.velocity) if hasattr(target, 'velocity') else 0
                    
                    # 计算目标到边界的距离
                    boundary_dist = min(
                        target.pos[0] - 0, 
                        config.AREA_WIDTH_METERS - target.pos[0],
                        target.pos[1] - 0, 
                        config.AREA_HEIGHT_METERS - target.pos[1]
                    )
                    
                    # 目标优先级计算（参考传统算法）
                    speed_priority = target_speed / 15.0  # 归一化到0-1
                    boundary_priority = max(0, (200 - boundary_dist) / 200)  # 边界优先级
                    priority = speed_priority * 0.6 + boundary_priority * 0.4
                    
                    target_obs.extend([
                        relative_pos[0] / 1000,  # 相对x位置(km)
                        relative_pos[1] / 1000,  # 相对y位置(km)
                        1.0 if target.is_detected else 0.0,  # 是否已探测
                        target_speed / 15.0,  # 归一化目标速度
                        boundary_priority  # 边界优先级
                    ])
                    target_priorities.append(priority)
                else:
                    target_obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # 无目标时填充0
                    target_priorities.append(0.0)
            
            # 3. 其他智能体信息 (20维: 5个其他智能体 × 4维[pos_x, pos_y, vx, vy])
            other_agents_obs = []
            other_agents = [a for a in self.agents if a.id != agent.id]
            for i in range(5):  # 最多5个其他智能体
                if i < len(other_agents):
                    other_agent = other_agents[i]
                    relative_pos = other_agent.pos - agent.pos
                    other_agents_obs.extend([
                        relative_pos[0] / 1000,  # 相对x位置(km)
                        relative_pos[1] / 1000,  # 相对y位置(km)
                        other_agent.velocity[0] / other_agent.max_speed, # 相对x速度
                        other_agent.velocity[1] / other_agent.max_speed  # 相对y速度
                    ])
                else:
                    other_agents_obs.extend([0.0, 0.0, 0.0, 0.0])  # 无智能体时填充0
            
            # 4. 探索状态信息 (9维: 3×3邻域网格)
            exploration_obs = self._get_exploration_state(agent)
            
            # 5. 目标优先级信息 (3维: 3个最近目标的优先级)
            priority_obs = target_priorities
            
            # 6. 协同信息 (4维)
            coordination_obs = [
                1.0 if agent.coordination_role == 'searcher' else 0.0,
                1.0 if agent.coordination_role == 'tracker' else 0.0, 
                1.0 if agent.coordination_role == 'interceptor' else 0.0,
                len(agent.shared_target_info) / 10.0  # 共享目标数量，归一化到0-1
            ]
            
            # 组合所有观测 (6+15+20+9+3+4=57维)
            full_obs = self_obs + target_obs + other_agents_obs + exploration_obs + priority_obs + coordination_obs
            observations[agent.id] = np.array(full_obs, dtype=np.float32)
            
        return observations
    
    def _get_nearest_targets(self, agent, n=3):
        """
        获取距离智能体最近的n个目标
        """
        if not self.targets:
            return [None] * n
        
        # 计算所有目标到智能体的距离
        distances = []
        for target in self.targets:
            dist = np.linalg.norm(target.pos - agent.pos)
            distances.append((dist, target))
        
        # 按距离排序
        distances.sort(key=lambda x: x[0])
        
        # 返回最近的n个目标
        nearest = []
        for i in range(n):
            if i < len(distances):
                nearest.append(distances[i][1])
            else:
                nearest.append(None)
        
        return nearest
    
    def _get_exploration_state(self, agent):
        """
        获取智能体周围3×3网格的探索状态
        """
        current_grid_x, current_grid_y = self._get_grid_coords(agent.pos)
        exploration_state = []
        
        # 检查3×3邻域
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                grid_x = current_grid_x + dx
                grid_y = current_grid_y + dy
                
                # 检查边界
                if (0 <= grid_x < self.grid_shape[1] and 
                    0 <= grid_y < self.grid_shape[0]):
                    # 检查是否被探索过
                    explored = self.exploration_grids[agent.id][grid_y, grid_x]
                    exploration_state.append(float(explored))
                else:
                    exploration_state.append(0.0)  # 边界外视为未探索
        
        return exploration_state
    
    def close(self):
        if config.ENABLE_RENDERING:
            pygame.quit()
