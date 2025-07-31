import gymnasium as gym
import torch
from PPO import Memory, PPO

############## 超参数 ##############
env_name = "LunarLander-v3"  # 游戏名字
env = gym.make(env_name, render_mode="human")
state_dim = 8  # 状态维度
action_dim = 4  # 行动维度
render = True  # 可视化
solved_reward = 230  # 停止循环条件 (奖励 > 230)
log_interval = 20  # print avg reward in the interval
max_episodes = 50000  # 最大迭代次数
max_timesteps = 300  # 最大单次游戏步数
n_latent_var = 64  # 全连接隐层维度
update_timestep = 2000  # 每2000步policy更新一次
lr = 0.002  # 学习率
betas = (0.9, 0.999)  # betas
gamma = 0.99  # gamma
K_epochs = 4  # policy迭代更新次数
eps_clip = 0.2  # PPO 限幅


#############################################

def main():
    # 实例化
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    # 存放
    total_reward = 0
    total_length = 0
    timestep = 0

    # 训练
    for i_episode in range(1, max_episodes + 1):

        # 环境初始化
        state, _ = env.reset()  # 初始化（重新玩）

        # 迭代
        for t in range(max_timesteps):
            timestep += 1

            # 用旧policy得到行动
            action = ppo.policy_old.act(state, memory)

            # 行动
            state, reward, done, truncated, _ = env.step(action)  # 得到（新的状态，奖励，是否终止，额外的调试信息）
            done = done or truncated  # 合并终止条件

            # 更新memory(奖励/游戏是否结束)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # 更新梯度
            if timestep % update_timestep == 0:
                ppo.update(memory)

                # memory清零
                memory.clear_memory()

                # 累计步数清零
                timestep = 0

            # 累加
            total_reward += reward

            # 可视化
            if render:
                env.render()

            # 如果游戏结束, 退出
            if done:
                break

        # 游戏步长
        total_length += t

        # 如果达到要求(230分), 退出循环
        if total_reward >= (log_interval * solved_reward):
            print("########## Solved! ##########")

            # 保存模型
            torch.save(ppo.policy.state_dict(), './PPO_LunarLander-v3.pth')

            # 退出循环
            break

        # 输出log, 每20次迭代
        if i_episode % log_interval == 0:
            # 求20次迭代平均时长/收益
            avg_length = int(total_length / log_interval)
            running_reward = int(total_reward / log_interval)
            # 调试输出
            print('Episode {} \t avg length: {} \t average_reward: {}'.format(i_episode, avg_length, running_reward))

            # 清零
            total_reward = 0
            total_length = 0


if __name__ == '__main__':
    main()
