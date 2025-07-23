import re
import ast
import wandb
import yaml
import os
from datetime import datetime

def parse_log_line(line):
    """解析日志行，提取字典数据"""
    # 匹配时间戳和字典部分
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO \| train\.py:\d+ \| (.+)'
    match = re.search(pattern, line)
    
    if match:
        dict_str = match.group(1)
        try:
            # 使用ast.literal_eval安全地解析字典字符串
            log_dict = ast.literal_eval(dict_str)
            return log_dict
        except (ValueError, SyntaxError) as e:
            print(f"解析字典失败: {e}")
            return None
    return None

def load_config_and_init_wandb(config_path, api_key_path):
    """加载配置并初始化wandb"""
    # 加载API密钥
    with open(api_key_path, 'r') as f:
        api_keys = yaml.safe_load(f)
    
    os.environ["WANDB_API_KEY"] = api_keys['wandb']
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化wandb
    wandb.init(
        project='LVSM',
        name='0719-baseline-8',
        config=config,
        resume="allow"  # 允许恢复现有的运行
    )
    
    return config

def replay_logs_to_wandb(log_file_path, config_path, api_key_path):
    """从日志文件重新播放数据到wandb"""
    
    # 初始化wandb
    config = load_config_and_init_wandb(config_path, api_key_path)
    
    print(f"开始从日志文件重新播放数据到wandb...")
    print(f"日志文件: {log_file_path}")
    print(f"项目名称: {config['training']['wandb_project']}")
    print(f"实验名称: {config['training']['wandb_exp_name']}")
    
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析并发送数据到wandb
    count = 0
    for line in lines:
        log_dict = parse_log_line(line.strip())
        if log_dict:
            # 使用total_samples作为step
            step = log_dict.get('total_samples', log_dict.get('iter', count))
            
            # 发送到wandb
            wandb.log(log_dict, step=step)
            count += 1
            
            if count % 100 == 0:
                print(f"已处理 {count} 条日志记录...")
    
    print(f"完成！总共处理了 {count} 条日志记录")
    wandb.finish()

def main():
    # 配置路径 - 请根据实际情况修改
    log_file_path = "logs/7.19-baseline-12"  # 你的日志文件路径
    config_path = "configs/LVSM_ours.yaml"  # 你的配置文件路径
    api_key_path = "configs/api_keys.yaml"  # API密钥文件路径
    
    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"错误：日志文件不存在: {log_file_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}")
        return
    
    if not os.path.exists(api_key_path):
        print(f"错误：API密钥文件不存在: {api_key_path}")
        return
    
    # 重新播放日志到wandb
    replay_logs_to_wandb(log_file_path, config_path, api_key_path)

if __name__ == "__main__":
    main()