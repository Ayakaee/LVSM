import re
import ast
import wandb
import yaml
import os
from datetime import datetime
from collections import defaultdict

def parse_log_line(line):
    """解析日志行，提取字典数据和时间戳"""
    # 更精确的正则表达式匹配你的日志格式
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| INFO \| train\.py:\d+ \| (.+)'
    match = re.search(pattern, line)
    
    if match:
        timestamp_str = match.group(1)
        dict_str = match.group(2)
        
        try:
            # 使用ast.literal_eval安全地解析字典字符串
            log_dict = ast.literal_eval(dict_str)
            
            # 解析时间戳
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            return log_dict, timestamp
        except (ValueError, SyntaxError) as e:
            print(f"解析字典失败: {e}")
            print(f"问题行: {line.strip()}")
            return None, None
    return None, None

def load_config_and_init_wandb(config_path, api_key_path, run_name=None):
    """加载配置并初始化wandb"""
    # 加载API密钥
    with open(api_key_path, 'r') as f:
        api_keys = yaml.safe_load(f)
    
    os.environ["WANDB_API_KEY"] = api_keys['wandb']
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取项目名称和实验名称
    project_name = config.get('training', {}).get('wandb_project', 'LVSM')
    exp_name = run_name or config.get('training', {}).get('wandb_exp_name', 'default-run')
    
    # 初始化wandb
    wandb.init(
        project=project_name,
        name=exp_name,
        config=config,
        resume="allow"  # 允许恢复现有的运行
    )
    
    return config

def deduplicate_logs(log_entries):
    """
    去重日志条目，对于相同iter的条目只保留最新的
    log_entries: [(log_dict, timestamp), ...]
    """
    # 使用字典存储每个iter的最新条目
    iter_to_entry = {}
    
    for log_dict, timestamp in log_entries:
        iter_num = log_dict.get('iter')
        if iter_num is not None:
            # 如果这个iter还没有记录，或者当前时间戳更新，则更新
            if (iter_num not in iter_to_entry or 
                timestamp > iter_to_entry[iter_num][1]):
                iter_to_entry[iter_num] = (log_dict, timestamp)
        else:
            # 如果没有iter字段，直接添加（使用时间戳作为唯一标识）
            unique_key = f"no_iter_{timestamp.timestamp()}"
            iter_to_entry[unique_key] = (log_dict, timestamp)
    
    # 按iter排序返回
    sorted_entries = []
    for key, (log_dict, timestamp) in iter_to_entry.items():
        if isinstance(key, int):  # 有iter的条目
            sorted_entries.append((key, log_dict, timestamp))
        else:  # 没有iter的条目
            sorted_entries.append((float('inf'), log_dict, timestamp))
    
    sorted_entries.sort(key=lambda x: x[0])
    return [(log_dict, timestamp) for _, log_dict, timestamp in sorted_entries]

def replay_logs_to_wandb(log_file_path, config_path, api_key_path, run_name=None):
    """从日志文件重新播放数据到wandb"""
    
    # 初始化wandb
    config = load_config_and_init_wandb(config_path, api_key_path, run_name)
    
    print(f"开始从日志文件重新播放数据到wandb...")
    print(f"日志文件: {log_file_path}")
    
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析所有日志行
    log_entries = []
    parse_errors = 0
    
    print("正在解析日志文件...")
    for i, line in enumerate(lines):
        log_dict, timestamp = parse_log_line(line.strip())
        if log_dict and timestamp:
            log_entries.append((log_dict, timestamp))
        elif line.strip():  # 忽略空行
            parse_errors += 1
            if parse_errors <= 5:  # 只显示前5个错误
                print(f"解析失败的行 {i+1}: {line.strip()[:100]}...")
    
    print(f"解析完成！成功解析 {len(log_entries)} 条记录，失败 {parse_errors} 条")
    
    # 去重处理
    print("正在去重...")
    original_count = len(log_entries)
    log_entries = deduplicate_logs(log_entries)
    deduplicated_count = len(log_entries)
    
    if original_count != deduplicated_count:
        print(f"去重完成！从 {original_count} 条记录减少到 {deduplicated_count} 条")
    else:
        print("没有发现重复的iter记录")
    
    # 发送到wandb
    print("开始上传到WandB...")
    count = 0
    for log_dict, timestamp in log_entries:
        # 优先使用iter作为step，其次使用total_samples
        step = log_dict.get('iter', log_dict.get('total_samples', count))
        
        # 发送到wandb
        wandb.log(log_dict, step=step)
        count += 1
        
        if count % 100 == 0:
            print(f"已上传 {count}/{deduplicated_count} 条记录...")
    
    print(f"完成！总共上传了 {count} 条日志记录到WandB")
    wandb.finish()

def main():
    """主函数"""
    # 配置路径 - 请根据实际情况修改
    log_file_path = "logs/9.5-dinov3-ex=0-8-3-layer=24-scale"  # 你的日志文件路径
    config_path = "configs/LVSM_ours.yaml"  # 你的配置文件路径  
    api_key_path = "configs/api_keys.yaml"  # API密钥文件路径
    run_name = "0719-baseline-8-replay"  # 可选：自定义运行名称
    
    # 检查文件是否存在
    files_to_check = [
        (log_file_path, "日志文件"),
        (config_path, "配置文件"), 
        (api_key_path, "API密钥文件")
    ]
    
    for file_path, file_type in files_to_check:
        if not os.path.exists(file_path):
            print(f"错误：{file_type}不存在: {file_path}")
            return
    
    try:
        # 重新播放日志到wandb
        replay_logs_to_wandb(log_file_path, config_path, api_key_path, run_name)
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()