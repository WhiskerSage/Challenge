
import os
import itertools
import subprocess
import time
import json

# 1. 定义超参数网格（仅智能体/算法相关）
lr_actor_list = [0.001, 0.0005]
lr_critic_list = [0.001, 0.0005]
gamma_list = [0.95, 0.99]
tau_list = [0.01, 0.05]
batch_size_list = [64, 128]
update_freq_list = [100, 200]
reward_detect_list = [200, 300]

param_grid = list(itertools.product(
    lr_actor_list, lr_critic_list, gamma_list, tau_list, batch_size_list, update_freq_list, reward_detect_list
))

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

results_summary = []

for i, (lr_actor, lr_critic, gamma, tau, batch_size, update_freq, reward_detect) in enumerate(param_grid):
    exp_name = f"exp_{i}_lrA{lr_actor}_lrC{lr_critic}_g{gamma}_tau{tau}_bs{batch_size}_uf{update_freq}_rd{reward_detect}"
    log_file_path = f"logs/{exp_name}.txt"
    result_file_path = f"results/{exp_name}.json"

    print(f"[BatchTune] Running {exp_name} ({i+1}/{len(param_grid)})")

    # 构建命令行参数
    args = [
        "python", "main.py",
        f"--lr_actor={lr_actor}",
        f"--lr_critic={lr_critic}",
        f"--gamma={gamma}",
        f"--tau={tau}",
        f"--batch_size={batch_size}",
        f"--update_freq={update_freq}",
        f"--reward_detect={reward_detect}"
    ]

    # 运行 main.py 并捕获输出（兼容中文Windows控制台编码）
    with open(log_file_path, "w", encoding="utf-8") as outfile:
        process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="gbk", errors="replace")
        outfile.write(process.stdout)

    # 解析 main.py 输出中的 JSON 评估结果
    eval_json = None
    for line in process.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                eval_json = json.loads(line)
                break
            except Exception:
                continue

    if eval_json is not None:
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_json, f, ensure_ascii=False, indent=2)
        # 记录参数与结果
        results_summary.append({
            "exp_name": exp_name,
            "params": {
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                "gamma": gamma,
                "tau": tau,
                "batch_size": batch_size,
                "update_freq": update_freq,
                "reward_detect": reward_detect
            },
            "result": eval_json
        })
    else:
        print(f"[BatchTune] Warning: No JSON result found in {exp_name}")

    time.sleep(1)

# 5. 综合评分与最佳参数识别
def score_fn(result):
    # 你可以根据赛题要求自定义综合评分函数
    # 这里假设 result 中有 'final_avg_reward' 和 'final_detection_rate' 字段
    reward = result.get('final_avg_reward', 0)
    detection = result.get('final_detection_rate', 0)
    # 综合分 = 检测率优先 + 奖励
    return detection * 1000 + reward

best = None
best_score = float('-inf')
for item in results_summary:
    score = score_fn(item["result"])
    item["score"] = score
    if score > best_score:
        best = item
        best_score = score

print("\n[BatchTune] All experiments finished. Check logs/ and results/ directory for details.")
if best:
    print("\n[BatchTune] Best parameter combination:")
    print(json.dumps(best["params"], indent=2, ensure_ascii=False))
    print("[BatchTune] Best result summary:")
    print(json.dumps(best["result"], indent=2, ensure_ascii=False))
    print(f"[BatchTune] Best score: {best_score}")
else:
    print("[BatchTune] No valid results found.")