import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def extract_log_data(log_file):
    """从日志文件中提取训练数据"""
    data = {
        'episodes': [],
        'steps': [],
        'avg_rewards': [],
        'total_rewards': [],
        'detection_rates': [],
        'detection_counts': [],
        'episode_times': [],
        'termination_reasons': []
    }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取Episode Summary信息
    episode_summaries = re.findall(
        r'Episode (\d+) Summary:\s*\n\s*Total Reward: ([\d.-]+)\s*\n\s*Targets Detected: (\d+)/(\d+) \(([\d.]+)%\)\s*\n\s*Steps: (\d+)\s*\n\s*Episode Time: ([\d.]+)s\s*\n\s*Termination Reason: (\w+)',
        content
    )
    
    for match in episode_summaries:
        episode_num, total_reward, detected, total_targets, detection_rate, steps, episode_time, termination = match
        
        data['episodes'].append(int(episode_num))
        data['total_rewards'].append(float(total_reward))
        data['detection_rates'].append(float(detection_rate))
        data['detection_counts'].append(int(detected))
        data['steps'].append(int(steps))
        data['episode_times'].append(float(episode_time))
        data['avg_rewards'].append(float(total_reward) / int(steps))
        data['termination_reasons'].append(termination)
    
    # 提取训练过程中的reward变化
    step_rewards = re.findall(
        r'Step (\d+), Episode Steps: (\d+), Avg Reward: ([-\d.]+), Episode Reward: ([-\d.]+)',
        content
    )
    
    training_progress = []
    for match in step_rewards:
        step, episode_steps, avg_reward, episode_reward = match
        training_progress.append({
            'step': int(step),
            'episode_steps': int(episode_steps),
            'avg_reward': float(avg_reward),
            'episode_reward': float(episode_reward)
        })
    
    return data, training_progress

def analyze_single_log(log_file, exp_name):
    """分析单个日志文件"""
    try:
        episode_data, training_progress = extract_log_data(log_file)
        
        if not episode_data['episodes']:
            return None
            
        analysis = {
            'exp_name': exp_name,
            'total_episodes': len(episode_data['episodes']),
            'final_avg_reward': np.mean(episode_data['total_rewards'][-10:]) if len(episode_data['total_rewards']) >= 10 else np.mean(episode_data['total_rewards']),
            'best_episode_reward': max(episode_data['total_rewards']) if episode_data['total_rewards'] else 0,
            'worst_episode_reward': min(episode_data['total_rewards']) if episode_data['total_rewards'] else 0,
            'reward_std': np.std(episode_data['total_rewards']) if episode_data['total_rewards'] else 0,
            'avg_detection_rate': np.mean(episode_data['detection_rates']) if episode_data['detection_rates'] else 0,
            'best_detection_rate': max(episode_data['detection_rates']) if episode_data['detection_rates'] else 0,
            'avg_episode_time': np.mean(episode_data['episode_times']) if episode_data['episode_times'] else 0,
            'avg_steps': np.mean(episode_data['steps']) if episode_data['steps'] else 0,
            'convergence_episode': None,  # 收敛的回合数
            'improvement_trend': None,    # 改善趋势
            'stability_score': None       # 稳定性得分
        }
        
        # 分析收敛性
        if len(episode_data['total_rewards']) >= 10:
            # 计算滑动平均
            window_size = min(10, len(episode_data['total_rewards']))
            moving_avg = pd.Series(episode_data['total_rewards']).rolling(window=window_size).mean().tolist()
            
            # 寻找收敛点（连续5个episode的平均奖励变化小于总体标准差的10%）
            threshold = analysis['reward_std'] * 0.1
            for i in range(len(moving_avg) - 5):
                if i >= window_size - 1:  # 确保有足够的数据
                    recent_values = moving_avg[i:i+5]
                    if max(recent_values) - min(recent_values) < threshold:
                        analysis['convergence_episode'] = i + 1
                        break
        
        # 分析改善趋势
        if len(episode_data['total_rewards']) >= 20:
            first_half = episode_data['total_rewards'][:len(episode_data['total_rewards'])//2]
            second_half = episode_data['total_rewards'][len(episode_data['total_rewards'])//2:]
            improvement = (np.mean(second_half) - np.mean(first_half)) / np.mean(first_half) * 100
            analysis['improvement_trend'] = improvement
        
        # 计算稳定性得分 (后30%episode的标准差)
        if len(episode_data['total_rewards']) >= 10:
            stable_episodes = episode_data['total_rewards'][int(len(episode_data['total_rewards']) * 0.7):]
            analysis['stability_score'] = np.std(stable_episodes) / np.mean(stable_episodes) if np.mean(stable_episodes) > 0 else float('inf')
        
        return analysis, episode_data, training_progress
    
    except Exception as e:
        print(f"Error analyzing {log_file}: {e}")
        return None

def analyze_all_logs():
    """分析所有日志文件"""
    logs_dir = "logs"
    all_analysis = []
    detailed_data = {}
    
    print("开始分析训练日志...")
    print("=" * 80)
    
    for filename in sorted(os.listdir(logs_dir)):
        if filename.endswith('.txt') and filename.startswith('exp_'):
            exp_name = filename[:-4]  # 去掉.txt
            log_path = os.path.join(logs_dir, filename)
            
            result = analyze_single_log(log_path, exp_name)
            if result:
                analysis, episode_data, training_progress = result
                all_analysis.append(analysis)
                detailed_data[exp_name] = {
                    'episode_data': episode_data,
                    'training_progress': training_progress
                }
                print(f"✓ 已分析: {exp_name}")
    
    if not all_analysis:
        print("未找到有效的日志数据")
        return
    
    df = pd.DataFrame(all_analysis)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("训练日志分析报告")
    print("=" * 80)
    
    # 1. 基础统计
    print(f"\n📊 基础统计:")
    print(f"  总实验数量: {len(df)}")
    print(f"  平均训练回合数: {df['total_episodes'].mean():.1f}")
    print(f"  最大单回合奖励: {df['best_episode_reward'].max():.1f}")
    print(f"  最小单回合奖励: {df['worst_episode_reward'].min():.1f}")
    
    # 2. 性能排名
    print(f"\n🏆 性能表现TOP10:")
    top_performers = df.nlargest(10, 'final_avg_reward')[['exp_name', 'final_avg_reward', 'best_episode_reward', 'avg_detection_rate', 'reward_std']]
    print(top_performers.to_string(index=False))
    
    # 3. 收敛性分析
    converged = df[df['convergence_episode'].notna()]
    if len(converged) > 0:
        print(f"\n⚡ 收敛性分析:")
        print(f"  收敛实验数量: {len(converged)}/{len(df)} ({len(converged)/len(df)*100:.1f}%)")
        print(f"  平均收敛回合: {converged['convergence_episode'].mean():.1f}")
        print(f"  最快收敛: {converged['convergence_episode'].min():.0f} 回合")
        
        fastest_converge = df.loc[df['convergence_episode'].idxmin()]
        print(f"  最快收敛实验: {fastest_converge['exp_name']} (第{fastest_converge['convergence_episode']:.0f}回合)")
    
    # 4. 稳定性分析
    stable = df[df['stability_score'].notna() & (df['stability_score'] != float('inf'))]
    if len(stable) > 0:
        print(f"\n📈 稳定性分析:")
        stable_top10 = stable.nsmallest(10, 'stability_score')[['exp_name', 'stability_score', 'final_avg_reward', 'reward_std']]
        print(f"  稳定性最好的10个实验:")
        print(stable_top10.to_string(index=False))
    
    # 5. 改善趋势分析
    improved = df[df['improvement_trend'].notna()]
    if len(improved) > 0:
        print(f"\n📊 学习改善分析:")
        print(f"  显示改善的实验: {len(improved[improved['improvement_trend'] > 0])}/{len(improved)} ({len(improved[improved['improvement_trend'] > 0])/len(improved)*100:.1f}%)")
        print(f"  平均改善幅度: {improved['improvement_trend'].mean():.1f}%")
        
        best_improvement = improved.loc[improved['improvement_trend'].idxmax()]
        print(f"  最大改善: {best_improvement['exp_name']} ({best_improvement['improvement_trend']:.1f}%)")
    
    # 6. 检测性能分析
    print(f"\n🎯 检测性能分析:")
    detection_top10 = df.nlargest(10, 'avg_detection_rate')[['exp_name', 'avg_detection_rate', 'best_detection_rate', 'final_avg_reward']]
    print(f"  检测率最高的10个实验:")
    print(detection_top10.to_string(index=False))
    
    # 7. 训练效率分析
    print(f"\n⏱️ 训练效率分析:")
    print(f"  平均每回合用时: {df['avg_episode_time'].mean():.1f}s")
    print(f"  平均每回合步数: {df['avg_steps'].mean():.0f}")
    efficiency = df['final_avg_reward'] / (df['avg_episode_time'] * df['total_episodes'])
    efficiency_analysis = pd.DataFrame({
        'exp_name': df['exp_name'],
        'efficiency_score': efficiency,
        'final_avg_reward': df['final_avg_reward'],
        'total_training_time': df['avg_episode_time'] * df['total_episodes']
    }).nlargest(10, 'efficiency_score')
    
    print(f"  训练效率最高的10个实验 (奖励/训练时间):")
    print(efficiency_analysis.to_string(index=False))
    
    # 8. 综合推荐
    print(f"\n🌟 综合推荐分析:")
    
    # 综合评分：性能 + 稳定性 + 收敛性
    df['comprehensive_score'] = df['final_avg_reward'] * 0.4
    
    # 稳定性加分（稳定性越好分数越高）
    if 'stability_score' in df.columns and df['stability_score'].notna().sum() > 0:
        df.loc[df['stability_score'].notna() & (df['stability_score'] != float('inf')), 'comprehensive_score'] += \
            (1 / df.loc[df['stability_score'].notna() & (df['stability_score'] != float('inf')), 'stability_score']) * 5000
    
    # 收敛性加分
    if 'convergence_episode' in df.columns and df['convergence_episode'].notna().sum() > 0:
        max_convergence = df['convergence_episode'].max()
        df.loc[df['convergence_episode'].notna(), 'comprehensive_score'] += \
            (max_convergence - df.loc[df['convergence_episode'].notna(), 'convergence_episode']) * 100
    
    comprehensive_top10 = df.nlargest(10, 'comprehensive_score')[['exp_name', 'comprehensive_score', 'final_avg_reward', 'stability_score', 'convergence_episode']]
    print(f"  综合表现最佳的10个实验:")
    print(comprehensive_top10.to_string(index=False))
    
    return df, detailed_data

if __name__ == "__main__":
    results = analyze_all_logs()