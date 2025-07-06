#!/usr/bin/env python3
"""
修复DINOv2代码中的Python 3.9兼容性问题
将 float | None 替换为 Union[float, None]
"""

import os
import re
from pathlib import Path

def fix_file(file_path):
    """修复单个文件中的类型注解"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加必要的import
    if 'Union' not in content and ('float | None' in content or 'int | None' in content):
        # 在import部分添加Union
        import_pattern = r'(from typing import.*?)(\n)'
        union_import = r'\1, Union\2'
        content = re.sub(import_pattern, union_import, content, flags=re.DOTALL)
        
        # 如果没有找到typing import，在文件开头添加
        if 'from typing import' not in content and 'import typing' not in content:
            content = 'from typing import Union\n' + content
    
    # 替换类型注解
    content = content.replace('float | None', 'Union[float, None]')
    content = content.replace('int | None', 'Union[int, None]')
    content = content.replace('str | None', 'Union[str, None]')
    content = content.replace('bool | None', 'Union[bool, None]')
    content = content.replace('Tensor | None', 'Union[Tensor, None]')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复: {file_path}")

def main():
    # DINOv2缓存路径
    cache_path = Path.home() / '.cache' / 'torch' / 'hub' / 'facebookresearch_dinov2_main'
    
    if not cache_path.exists():
        print(f"DINOv2缓存路径不存在: {cache_path}")
        return
    
    # 需要修复的文件
    files_to_fix = [
        cache_path / 'dinov2' / 'layers' / 'attention.py',
        cache_path / 'dinov2' / 'layers' / 'block.py',
        cache_path / 'dinov2' / 'models' / 'vision_transformer.py',
    ]
    
    for file_path in files_to_fix:
        if file_path.exists():
            fix_file(file_path)
        else:
            print(f"文件不存在: {file_path}")

if __name__ == '__main__':
    main() 