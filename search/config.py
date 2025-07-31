# config.py

# --- 物理单位换算 ---
KNOTS_TO_MPS = 0.5144  # 1节 ≈ 0.5144米/秒
NM_TO_METERS = 1852  # 1海里 = 1852米
KMH_TO_MPS = 1000 / 3600 # 公里/小时 to 米/秒

# --- 任务区域定义 ---
AREA_WIDTH_NM = 5.0  # 海里
AREA_HEIGHT_NM = 5.0  # 海里
AREA_WIDTH_METERS = AREA_WIDTH_NM * NM_TO_METERS  # 米
AREA_HEIGHT_METERS = AREA_HEIGHT_NM * NM_TO_METERS # 米

# --- 编队定义 ---
NUM_UAVS = 2
NUM_USVS = 4
TOTAL_AGENTS = NUM_UAVS + NUM_USVS

# --- 初始位置参数 ---
UAV_SPACING_METERS = 2000  # 无人机间距（待确认）
USV_SPACING_METERS = 1000  # 无人艇间距（赛题要求）

# --- 智能体物理属性 ---
# 无人机 (UAV)
UAV_MAX_SPEED_KMH = 120.0  # 公里/小时
UAV_MAX_SPEED_MPS = UAV_MAX_SPEED_KMH * KMH_TO_MPS # 米/秒
UAV_MIN_TURN_RADIUS = 100.0  # 米
UAV_COLLISION_RADIUS = 50.0  # 米

# 无人艇 (USV)
USV_MAX_SPEED_KNOTS = 20.0  # 节
USV_MAX_SPEED_MPS = USV_MAX_SPEED_KNOTS * KNOTS_TO_MPS # 米/秒
USV_MIN_TURN_RADIUS = 20.0  # 米
USV_COLLISION_RADIUS = 100.0  # 米

# --- 智能体传感器属性 ---
UAV_SENSOR_FOV_DEG = 60.0  # 扇形视场角（度）
UAV_SENSOR_RANGE = 1500.0  # 无人机探测距离（米）- 待确认
USV_SENSOR_RANGE = 1000.0  # 无人艇探测距离（米）- 待确认

# --- 目标属性 ---
TARGET_MAX_SPEED_KNOTS = 15.0  # 节
TARGET_MAX_SPEED_MPS = TARGET_MAX_SPEED_KNOTS * KNOTS_TO_MPS # 米/秒
TARGET_MAX_COUNT_AT_ONCE = 2 # 同一时刻最多出现的目标数

# --- 规则定义 ---
DETECTION_SUSTAIN_TIME = 10.0  # 秒

# --- 仿真参数 ---
SIM_TIME_STEP = 0.1  # 保持精确控制的时间步长
TARGET_FPS = 5       # 大幅降低FPS，从20降到5，训练时建议设为1或关闭渲染
ENABLE_RENDERING = False  # 新增：是否启用可视化渲染（训练时设为False可大幅提速）

# --- IPPO 强化学习超参数 ---
UPDATE_TIMESTEPS = 4000  # 每4000个时间步更新一次网络 (原2000)
K_EPOCHS = 20          # 每次更新时，用同一批数据训练20次 (原40)
EPS_CLIP = 0.2         # PPO中的裁剪范围
GAMMA = 0.99           # 奖励折扣因子
LR_ACTOR = 0.0001      # Actor网络的学习率 (原0.0003)
LR_CRITIC = 0.0003     # Critic网络的学习率 (原0.001)

# --- 奖励函数权重（基于比赛目标优化）---
# 核心目标：优先检测边界和高速目标
REWARD_DETECT = 200.0      # 基础检测奖励
REWARD_BOUNDARY_TARGET = 300.0    # 边界目标额外奖励（距边界<200米）
REWARD_HIGH_SPEED_TARGET = 250.0  # 高速目标额外奖励（>10节）
REWARD_EARLY_DETECTION = 150.0    # 早期发现奖励（<5分钟）
REWARD_COORDINATION_DETECT = 100.0 # 协同探测奖励（多智能体同时探测）

# 搜索和探索奖励
REWARD_EXPLORE = 1.0       # 探索新网格的奖励
REWARD_EFFICIENT_SEARCH = 20.0  # 高效搜索奖励（探索其他智能体未访问区域）
REWARD_COVERAGE_EFFICIENCY = 15.0  # 覆盖效率奖励
REWARD_AREA_COVERAGE = 10.0  # 区域覆盖奖励
REWARD_DISTANCE_EXPLORATION = 3.0  # 新增：距离起始点探索奖励

# 行为激励奖励
REWARD_APPROACHING_TARGET = 2.0 # 接近目标的奖励（增强）
REWARD_MOVEMENT = 0.1       # 移动激励奖励（增强探索）
REWARD_TARGET_TRACKING = 5.0 # 新增：持续跟踪可疑目标奖励

# 惩罚项（适度调整避免过度惩罚）
REWARD_TIME_STEP = -0.01   # 每个时间步的惩罚
REWARD_COLLISION = -30.0     # 碰撞惩罚（提高安全重要性）
REWARD_OUT_OF_BOUNDS = -15.0 # 出界惩罚
REWARD_INEFFICIENT_SEARCH = -5.0 # 新增：重复搜索已覆盖区域的惩罚

# --- 探索网格参数 ---
EXPLORATION_GRID_SIZE = 200 # 网格大小（米）

# --- 待确认参数（根据最终赛题要求调整）---
# 以下参数可能在正式赛题中有所变化，请注意更新
PARAMETERS_TO_CONFIRM = {
    'UAV_SPACING_METERS': '无人机间距，文档中缺失',
    'UAV_SENSOR_RANGE': '无人机探测距离，当前为估算值',
    'USV_SENSOR_RANGE': '无人艇探测距离，当前为估算值',
    'TARGET_SPAWN_MECHANISM': '目标生成机制需要根据"进入区域"的描述调整'
}

# --- 可视化颜色定义 ---
COLOR_BACKGROUND = (0, 50, 100)  # 深海蓝
COLOR_UAV = (0, 255, 255)      # 青色
COLOR_USV = (255, 255, 0)      # 黄色
COLOR_TARGET = (255, 100, 100) # 红色
COLOR_TARGET_DETECTED = (255, 255, 255) # 白色
COLOR_SENSOR_RANGE = (200, 200, 200, 50) # 半透明灰色
COLOR_TEXT = (255, 255, 255)     # 白色
COLOR_BLACK = (0, 0, 0)          # 黑色