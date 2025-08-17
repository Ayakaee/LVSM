#!/usr/bin/env python3
"""
手动下载DINOv3预训练模型的脚本
解决HTTP 403错误问题
"""

import os
import requests
from pathlib import Path

def download_file(url, local_path):
    """下载文件到指定路径"""
    print(f"正在下载: {url}")
    print(f"保存到: {local_path}")
    
    # 创建目录
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print(f"\n下载完成: {local_path}")
        return True
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        return False

def main():
    """主函数"""
    # 模型文件URL
    model_urls = {
        "dinov3_vitl16_pretrain_lvd1689m": "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/",
        "dinov3_vitl16_dinotxt_vision_head_and_text_encoder": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    }
    
    # 下载目录
    download_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    
    print("开始下载DINOv3预训练模型...")
    print(f"下载目录: {download_dir}")
    
    success_count = 0
    for model_name, url in model_urls.items():
        local_path = download_dir / f"{model_name}.pth"
        
        if local_path.exists():
            print(f"模型已存在: {local_path}")
            success_count += 1
            continue
            
        if download_file(url, local_path):
            success_count += 1
    
    print(f"\n下载完成: {success_count}/{len(model_urls)} 个模型")
    
    if success_count == len(model_urls):
        print("所有模型下载成功！现在可以运行您的代码了。")
    else:
        print("部分模型下载失败，请检查网络连接或手动下载。")

if __name__ == "__main__":
    main() 