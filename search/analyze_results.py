import json
import os
import pandas as pd
import numpy as np

def parse_exp_name(exp_name):
    """从实验名称解析参数"""
    parts = exp_name.split('_')
    params = {}
    for part in parts[2:]:  # 跳过 'exp' 和序号
        if part.startswith('lrA'):
            params['lr_actor'] = float(part[3:])
        elif part.startswith('lrC'):
            params['lr_critic'] = float(part[3:])
        elif part.startswith('g'):
            params['gamma'] = float(part[1:])
        elif part.startswith('tau'):
            params['tau'] = float(part[3:])
        elif part.startswith('bs'):
            params['batch_size'] = int(part[2:])
        elif part.startswith('uf'):
            params['update_freq'] = int(part[2:])
        elif part.startswith('rd'):
            params['reward_detect'] = int(part[2:])
    return params

def analyze_results():
    results_dir = "results"
    all_data = []
    
    # 读取所有json结果文件
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('exp_'):
            exp_name = filename[:-5]  # 去掉.json
            file_path = os.path.join(results_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                # 解析参数
                params = parse_exp_name(exp_name)
                
                # 合并数据
                data = {**params, **result}
                data['exp_name'] = exp_name
                
                # 计算综合得分
                detection_rate = result.get('final_detection_rate', 0)
                final_reward = result.get('final_avg_reward', 0)
                data['composite_score'] = detection_rate * 1000 + final_reward
                
                all_data.append(data)
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    # 转换为DataFrame方便分析
    df = pd.DataFrame(all_data)
    
    print("=" * 80)
    print("实验结果分析报告")
    print("=" * 80)
    
    # 1. 按不同指标排序的前10名
    print("\n1. 按最终平均奖励排序的前10名:")
    top_reward = df.nlargest(10, 'final_avg_reward')[['exp_name', 'lr_actor', 'lr_critic', 'gamma', 'tau', 'batch_size', 'update_freq', 'reward_detect', 'final_avg_reward', 'final_detection_rate']]
    print(top_reward.to_string(index=False))
    
    print("\n2. 按检测率排序的前10名:")
    top_detection = df.nlargest(10, 'final_detection_rate')[['exp_name', 'lr_actor', 'lr_critic', 'gamma', 'tau', 'batch_size', 'update_freq', 'reward_detect', 'final_avg_reward', 'final_detection_rate']]
    print(top_detection.to_string(index=False))
    
    print("\n3. 按综合得分排序的前10名:")
    top_composite = df.nlargest(10, 'composite_score')[['exp_name', 'lr_actor', 'lr_critic', 'gamma', 'tau', 'batch_size', 'update_freq', 'reward_detect', 'final_avg_reward', 'final_detection_rate', 'composite_score']]
    print(top_composite.to_string(index=False))
    
    print("\n4. 按最佳单回合奖励排序的前10名:")
    top_best_episode = df.nlargest(10, 'best_episode_reward')[['exp_name', 'lr_actor', 'lr_critic', 'gamma', 'tau', 'batch_size', 'update_freq', 'reward_detect', 'best_episode_reward', 'best_detection_count']]
    print(top_best_episode.to_string(index=False))
    
    # 2. 参数影响分析
    print("\n" + "=" * 80)
    print("参数影响分析")
    print("=" * 80)
    
    params_to_analyze = ['lr_actor', 'lr_critic', 'gamma', 'tau', 'batch_size', 'update_freq', 'reward_detect']
    
    for param in params_to_analyze:
        print(f"\n{param} 对各指标的影响:")
        grouped = df.groupby(param).agg({
            'final_avg_reward': ['mean', 'std'],
            'final_detection_rate': ['mean', 'std'],
            'composite_score': ['mean', 'std'],
            'best_episode_reward': ['mean', 'std']
        }).round(2)
        print(grouped)
    
    # 3. 最佳配置推荐
    print("\n" + "=" * 80)
    print("推荐配置")
    print("=" * 80)
    
    print("\n基于不同目标的推荐:")
    
    # 最高奖励配置
    best_reward_config = df.loc[df['final_avg_reward'].idxmax()]
    print(f"\n最高平均奖励配置 (奖励: {best_reward_config['final_avg_reward']:.1f}):")
    print(f"lr_actor={best_reward_config['lr_actor']}, lr_critic={best_reward_config['lr_critic']}, gamma={best_reward_config['gamma']}, tau={best_reward_config['tau']}, batch_size={int(best_reward_config['batch_size'])}, update_freq={int(best_reward_config['update_freq'])}, reward_detect={int(best_reward_config['reward_detect'])}")
    
    # 最高检测率配置
    best_detection_config = df.loc[df['final_detection_rate'].idxmax()]
    print(f"\n最高检测率配置 (检测率: {best_detection_config['final_detection_rate']:.1f}%):")
    print(f"lr_actor={best_detection_config['lr_actor']}, lr_critic={best_detection_config['lr_critic']}, gamma={best_detection_config['gamma']}, tau={best_detection_config['tau']}, batch_size={int(best_detection_config['batch_size'])}, update_freq={int(best_detection_config['update_freq'])}, reward_detect={int(best_detection_config['reward_detect'])}")
    
    # 最高综合得分配置  
    best_composite_config = df.loc[df['composite_score'].idxmax()]
    print(f"\n最高综合得分配置 (得分: {best_composite_config['composite_score']:.1f}):")
    print(f"lr_actor={best_composite_config['lr_actor']}, lr_critic={best_composite_config['lr_critic']}, gamma={best_composite_config['gamma']}, tau={best_composite_config['tau']}, batch_size={int(best_composite_config['batch_size'])}, update_freq={int(best_composite_config['update_freq'])}, reward_detect={int(best_composite_config['reward_detect'])}")
    
    # 4. 稳定性分析
    print("\n" + "=" * 80)
    print("稳定性分析")
    print("=" * 80)
    
    # 找出在多个指标上都表现较好的配置
    df['reward_rank'] = df['final_avg_reward'].rank(ascending=False)
    df['detection_rank'] = df['final_detection_rate'].rank(ascending=False)
    df['best_episode_rank'] = df['best_episode_reward'].rank(ascending=False)
    df['avg_rank'] = (df['reward_rank'] + df['detection_rank'] + df['best_episode_rank']) / 3
    
    print("\n综合排名前10的配置 (在多个指标上都表现良好):")
    balanced_top = df.nsmallest(10, 'avg_rank')[['exp_name', 'lr_actor', 'lr_critic', 'gamma', 'tau', 'batch_size', 'update_freq', 'reward_detect', 'final_avg_reward', 'final_detection_rate', 'best_episode_reward', 'avg_rank']]
    print(balanced_top.to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = analyze_results()