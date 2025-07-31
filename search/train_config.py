# train_config.py - 专门用于高速训练的配置
# 使用方法：在main.py开头添加 import train_config as config

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

# --- 目标属性 ---
TARGET_MAX_SPEED_KNOTS = 15.0  # 节
TARGET_MAX_SPEED_MPS = TARGET_MAX_SPEED_KNOTS * KNOTS_TO_MPS # 米/秒
TARGET_MAX_COUNT_AT_ONCE = 2 # 同一时刻最多出现的目标数

# --- 规则定义 ---
DETECTION_SUSTAIN_TIME = 10.0  # 秒

# --- 仿真参数（训练优化）---
SIM_TIME_STEP = 0.2  # 训练时可以适当增大时间步长到0.2秒，平衡精度和速度
TARGET_FPS = 1       # 训练时极低FPS
ENABLE_RENDERING = False  # 训练时禁用渲染

# --- IPPO 强化学习超参数 ---
UPDATE_TIMESTEPS = 1000  # 减少到1000，更频繁更新
K_EPOCHS = 20          # 减少到20次，加快训练
EPS_CLIP = 0.2         # PPO中的裁剪范围
GAMMA = 0.99           # 奖励折扣因子
LR_ACTOR = 0.0005      # 稍微提高学习率
LR_CRITIC = 0.002      # 稍微提高学习率

# --- 奖励函数权重 ---
REWARD_DETECT = 200.0      # 成功探测到新目标的奖励（持续10秒）
REWARD_EXPLORE = 0.5       # 探索新网格的奖励
REWARD_EFFICIENT_SEARCH = 20.0  # 高效搜索奖励
REWARD_TIME_STEP = -0.05   # 每个时间步的惩罚
REWARD_COLLISION = -50.0     # 碰撞惩罚
REWARD_OUT_OF_BOUNDS = -20.0 # 出界惩罚
REWARD_COORDINATION = 50.0   # 协同探测奖励
REWARD_AREA_COVERAGE = 10.0  # 区域覆盖奖励

# --- 探索网格参数 ---
EXPLORATION_GRID_SIZE = 200 # 网格大小（米）

# --- 可视化颜色定义 ---
COLOR_BACKGROUND = (0, 50, 100)  # 深海蓝
COLOR_UAV = (0, 255, 255)      # 青色
COLOR_USV = (255, 255, 0)      # 黄色
COLOR_TARGET = (255, 100, 100) # 红色
COLOR_TARGET_DETECTED = (255, 255, 255) # 白色
COLOR_SENSOR_RANGE = (200, 200, 200, 50) # 半透明灰色
COLOR_TEXT = (255, 255, 255)     # 白色
COLOR_BLACK = (0, 0, 0)          # 黑色