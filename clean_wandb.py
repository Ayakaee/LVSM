import os
import shutil

def get_directory_size(directory):
    """计算目录及其所有内容的总大小（字节）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # 跳过符号链接
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    # 如果文件无法访问（如权限问题），跳过
                    continue
    return total_size

def delete_small_subdirectories(parent_dir, min_size_bytes, dry_run=False):
    """
    删除小于指定大小的一级子目录
    
    Args:
        parent_dir: 父目录路径
        min_size_bytes: 最小大小（字节），小于此大小的子目录将被删除
        dry_run: 如果为True，只显示不实际删除
    """
    if not os.path.isdir(parent_dir):
        print(f"错误: {parent_dir} 不是有效的目录")
        return [], []
    
    print(f"扫描目录: {parent_dir}")
    print(f"最小大小: {min_size_bytes} 字节")
    print(f"试运行模式: {'是' if dry_run else '否'}")
    
    deleted_dirs = []
    kept_dirs = []
    
    # 只获取一级子目录
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            dir_size = get_directory_size(item_path)
            
            if dir_size < min_size_bytes:
                print(f"标记删除: {item_path} (大小: {dir_size /1024} KB)")
                deleted_dirs.append((item_path, dir_size))
                
                if not dry_run:
                    try:
                        shutil.rmtree(item_path)
                        print(f"已删除: {item_path}")
                    except Exception as e:
                        print(f"删除失败: {item_path} - {str(e)}")
            # else:
            #     print(f"保留目录: {item_path} (大小: {dir_size} 字节)")
            #     kept_dirs.append((item_path, dir_size))
    
    # 打印摘要
    print("\n操作摘要:")
    print(f"标记删除的目录数: {len(deleted_dirs)}")
    # print(f"保留的目录数: {len(kept_dirs)}")
    
    # 保存结果到文件
    report_file = 'directory_cleanup_report.txt'
    with open(report_file, 'w') as f:
        f.write("标记删除的目录:\n")
        for path, size in deleted_dirs:
            f.write(f"{path} ({size} 字节)\n")
        f.write("\n保留的目录:\n")
        for path, size in kept_dirs:
            f.write(f"{path} ({size} 字节)\n")
    
    print(f"\n操作完成，结果已保存到 {report_file}")
    return deleted_dirs, kept_dirs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='删除小于指定大小的一级子目录')
    parser.add_argument('directory', help='要扫描的父目录')
    parser.add_argument('--min-size', type=int, required=True,
                       help='最小目录大小（字节），小于此值的子目录将被删除')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示将要删除的目录，不实际执行删除')
    
    args = parser.parse_args()
    
    delete_small_subdirectories(args.directory, args.min_size, args.dry_run)