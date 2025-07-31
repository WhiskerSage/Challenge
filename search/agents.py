# agents.py
import numpy as np
import config

class Agent:
    """所有智能体的基类（增强协同功能）"""
    def __init__(self, agent_id, initial_pos, max_speed, turn_radius, collision_radius):
        self.id = agent_id
        self.initial_pos = np.array(initial_pos, dtype=float)
        self.pos = np.array(initial_pos, dtype=float)
        self.heading = 0.0  # 航向角，0表示正东方向，单位：弧度
        self.velocity = np.array([0.0, 0.0], dtype=float) # 速度向量 (vx, vy)
        
        self.max_speed = max_speed
        self.min_turn_radius = turn_radius
        self.collision_radius = collision_radius

        # 协同机制增强
        self.coordination_role = 'searcher'  # 'searcher', 'tracker', 'interceptor'
        self.shared_target_info = {}   # 共享目标信息 {target_id: {'pos', 'speed', 'priority', 'last_seen'}}
        self.communication_range = 2000  # 通信范围（米）
        self.last_target_update = 0  # 最后更新目标信息的时间
        
    def update_coordination_info(self, other_agents, targets, current_time):
        """
        更新协同信息 - 参考传统算法的信息共享机制
        """
        # 清理过时的目标信息
        for target_id in list(self.shared_target_info.keys()):
            if current_time - self.shared_target_info[target_id]['last_seen'] > 60:  # 60秒过时
                del self.shared_target_info[target_id]
        
        # 与通信范围内的其他智能体共享信息
        for other_agent in other_agents:
            if other_agent.id != self.id:
                distance = np.linalg.norm(self.pos - other_agent.pos)
                if distance <= self.communication_range:
                    # 共享目标信息
                    for target_id, info in other_agent.shared_target_info.items():
                        if (target_id not in self.shared_target_info or 
                            info['last_seen'] > self.shared_target_info[target_id]['last_seen']):
                            self.shared_target_info[target_id] = info.copy()
        
        # 更新自己看到的目标信息
        for target in targets:
            if self._can_detect_target(target):
                target_speed = np.linalg.norm(target.velocity) if hasattr(target, 'velocity') else 0
                boundary_dist = min(
                    target.pos[0] - 0, 9260 - target.pos[0],  # 假设区域大小
                    target.pos[1] - 0, 9260 - target.pos[1]
                )
                priority = (target_speed / 15.0) * 0.6 + max(0, (200 - boundary_dist) / 200) * 0.4
                
                self.shared_target_info[target.id] = {
                    'pos': target.pos.copy(),
                    'velocity': target.velocity.copy() if hasattr(target, 'velocity') else np.array([0, 0]),
                    'speed': target_speed,
                    'priority': priority,
                    'last_seen': current_time,
                    'detected_by': self.id
                }
    
    def _can_detect_target(self, target):
        """简化的目标检测判断"""
        distance = np.linalg.norm(self.pos - target.pos)
        return distance <= self.sensor_range if hasattr(self, 'sensor_range') else False
    
    def get_coordination_role(self, targets, other_agents):
        """
        根据当前情况动态分配协同角色
        参考传统算法的任务分配策略
        """
        # 如果有高优先级目标，优先分配跟踪角色
        high_priority_targets = [
            tid for tid, info in self.shared_target_info.items() 
            if info['priority'] > 0.7 and info['last_seen'] > self.last_target_update - 30
        ]
        
        if high_priority_targets and self.coordination_role != 'tracker':
            # 检查是否有其他智能体已在跟踪
            other_trackers = [a for a in other_agents if a.coordination_role == 'tracker']
            if len(other_trackers) < len(high_priority_targets):
                self.coordination_role = 'tracker'
                return
                
        # 如果是USV且有检测到的目标，切换到拦截模式
        if isinstance(self, USV):
            detected_targets = [
                tid for tid, info in self.shared_target_info.items()
                if info['last_seen'] > self.last_target_update - 15  # 15秒内的信息
            ]
            if detected_targets:
                self.coordination_role = 'interceptor'
                return
        
        # 默认搜索模式
        self.coordination_role = 'searcher'

    def reset(self):
        self.pos = np.copy(self.initial_pos)
        self.heading = 0.0
        self.velocity = np.array([0.0, 0.0])
        # self.current_waypoint_index = 0

    def move(self, action, dt):
        """
        优化的连续和离散动作移动函数，支持更精确的控制
        """
        if isinstance(action, (list, np.ndarray)) and len(action) == 2:
            # 连续动作: [速度比例, 转向角度]
            speed_ratio, turn_ratio = action
            
            # 限制范围并应用更精细的控制
            speed_ratio = np.clip(speed_ratio, 0.0, 1.0)
            turn_ratio = np.clip(turn_ratio, -1.0, 1.0)
            
            # 计算实际速度（添加加速度约束）
            target_speed = self.max_speed * speed_ratio
            current_speed_magnitude = np.linalg.norm(self.velocity)
            
            # 加速度限制（每秒最多改变20%最大速度）
            max_speed_change = self.max_speed * 0.2 * dt
            if abs(target_speed - current_speed_magnitude) > max_speed_change:
                if target_speed > current_speed_magnitude:
                    current_speed = current_speed_magnitude + max_speed_change
                else:
                    current_speed = max(0, current_speed_magnitude - max_speed_change)
            else:
                current_speed = target_speed
            
            # 转向角速度限制
            if current_speed > 0:
                max_turn_rate = current_speed / self.min_turn_radius
                # 限制最大转向角速度（更真实的物理约束）
                max_angular_velocity = min(max_turn_rate, np.pi/2)  # 最大90度/秒
                delta_heading = max_angular_velocity * dt * turn_ratio
            else:
                # 原地转向时的角速度限制
                delta_heading = np.pi/4 * dt * turn_ratio  # 最大45度/秒
            
        else:
            # 改进的离散动作处理（更多精细控制选项）
            current_speed = self.max_speed
            delta_heading = 0.0
            
            if action == 0:  # 直行
                pass
            elif action == 1:  # 轻微左转
                max_turn_rate = current_speed / self.min_turn_radius
                delta_heading = max_turn_rate * dt * 0.2
            elif action == 2:  # 轻微右转
                max_turn_rate = current_speed / self.min_turn_radius
                delta_heading = -max_turn_rate * dt * 0.2
            elif action == 3:  # 中度左转
                max_turn_rate = current_speed / self.min_turn_radius
                delta_heading = max_turn_rate * dt * 0.5
            elif action == 4:  # 中度右转
                max_turn_rate = current_speed / self.min_turn_radius
                delta_heading = -max_turn_rate * dt * 0.5
            elif action == 5:  # 大幅左转
                max_turn_rate = current_speed / self.min_turn_radius
                delta_heading = max_turn_rate * dt * 0.8
            elif action == 6:  # 大幅右转
                max_turn_rate = current_speed / self.min_turn_radius
                delta_heading = -max_turn_rate * dt * 0.8
            elif action == 7:  # 减速直行
                current_speed = self.max_speed * 0.5
            elif action == 8:  # 停止
                current_speed = 0.0
            elif action == 9:  # 原地左转
                current_speed = 0.0
                delta_heading = np.pi/4 * dt
            elif action == 10:  # 原地右转
                current_speed = 0.0
                delta_heading = -np.pi/4 * dt
            else:
                pass  # 默认直行
        
        # 通用的位置更新逻辑
        self.heading += delta_heading
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        
        # 更新速度向量
        self.velocity[0] = current_speed * np.cos(self.heading)
        self.velocity[1] = current_speed * np.sin(self.heading)
        
        # 更新位置
        self.pos += self.velocity * dt


class UAV(Agent):
    """无人机类"""
    def __init__(self, agent_id, initial_pos):
        super().__init__(
            agent_id=agent_id,
            initial_pos=initial_pos,
            max_speed=config.UAV_MAX_SPEED_MPS,
            turn_radius=config.UAV_MIN_TURN_RADIUS,
            collision_radius=config.UAV_COLLISION_RADIUS
        )
        self.sensor_fov_deg = config.UAV_SENSOR_FOV_DEG
        self.sensor_range = config.UAV_SENSOR_RANGE

class USV(Agent):
    """无人艇类"""
    def __init__(self, agent_id, initial_pos):
        super().__init__(
            agent_id=agent_id,
            initial_pos=initial_pos,
            max_speed=config.USV_MAX_SPEED_MPS,
            turn_radius=config.USV_MIN_TURN_RADIUS,
            collision_radius=config.USV_COLLISION_RADIUS
        )
        self.sensor_range = config.USV_SENSOR_RANGE

class Target:
    """水面目标类"""
    def __init__(self, target_id, initial_pos, initial_heading):
        self.id = target_id
        self.pos = np.array(initial_pos, dtype=float)
        self.heading = initial_heading
        self.speed = np.random.uniform(0.5, config.TARGET_MAX_SPEED_MPS)
        self.is_detected = False # 新增：是否被成功探测
    
    def move(self, dt):
        """根据时间步长dt更新目标位置"""
        # 简单的随机游走
        self.heading += np.random.uniform(-0.1, 0.1) 
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        
        self.pos[0] += self.speed * np.cos(self.heading) * dt
        self.pos[1] += self.speed * np.sin(self.heading) * dt
    
