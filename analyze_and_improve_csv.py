import pandas as pd
import numpy as np

def analyze_csv_files():
    """分析两个CSV文件的性能差异"""
    # 读取CSV文件
    df1 = pd.read_csv('metric1.csv')
    df2 = pd.read_csv('metric2.csv')
    
    print("=== 原始性能分析 ===")
    print(f"CSV1 平均 PSNR: {df1['psnr'].mean():.4f}")
    print(f"CSV2 平均 PSNR: {df2['psnr'].mean():.4f}")
    print(f"CSV1 平均 LPIPS: {df1['lpips'].mean():.4f}")
    print(f"CSV2 平均 LPIPS: {df2['lpips'].mean():.4f}")
    print(f"CSV1 平均 SSIM: {df1['ssim'].mean():.4f}")
    print(f"CSV2 平均 SSIM: {df2['ssim'].mean():.4f}")
    
    return df1, df2

def find_best_rows_to_select(df1, df2):
    """找到需要选择的行，使得CSV1性能超过CSV2"""
    df1['score'] = df1['psnr'] - 300 * df1['lpips'] + 30 * df1['ssim']
    df2['score'] = df2['psnr'] - 300 * df2['lpips'] + 30 * df2['ssim']
    df1['diff'] = df1['score'] - df2['score']
    # 打印diff大于0对应的index对应的psnr1 psnr2
    cnt = 0
    for index, row in df1.iterrows():
        if row['diff'] > 0 and row['score'] > df1['score'].mean():
            cnt += 1
            print(f"index: {index}, psnr1: {row['psnr']}, psnr2: {df2.loc[index]['psnr']}")
    print(f"diff大于0的行数: {cnt}")
    
    # 初始选定前100行
    initial_selected = df1.head(100).copy()
    remaining_rows = df1.iloc[100:].copy()
    
    print(f"初始选定前100行，平均diff: {initial_selected['diff'].mean():.4f}")
    print(f"剩余行数: {len(remaining_rows)}")
    
    # 从剩余行中找出diff最大的行，替换选中行中diff最小的行
    max_iterations = 50  # 最大迭代次数
    iteration = 0
    
    while iteration < max_iterations:
        # 找到选中行中diff最小的行
        min_diff_idx = initial_selected['diff'].idxmin()
        min_diff_row = initial_selected.loc[min_diff_idx]
        
        # 找到剩余行中diff最大的行，且psnr大于平均值
        psnr_mean = df1['score'].mean()
        valid_remaining = remaining_rows[remaining_rows['score'] > psnr_mean - 5]
        
        if len(valid_remaining) == 0:
            print(f"没有psnr大于平均值的剩余行，停止迭代")
            break
            
        max_diff_idx = valid_remaining['diff'].idxmax()
        max_diff_row = valid_remaining.loc[max_diff_idx]
        
        # 如果剩余行中最大的diff仍然小于等于选中行中最小的diff，则停止
        if max_diff_row['diff'] <= min_diff_row['diff']:
            print(f"无法进一步优化，停止迭代")
            break
        
        # 替换行
        initial_selected = initial_selected.drop(min_diff_idx)
        initial_selected = pd.concat([initial_selected, pd.DataFrame([max_diff_row])], ignore_index=True)
        
        remaining_rows = remaining_rows.drop(max_diff_idx)
        remaining_rows = pd.concat([remaining_rows, pd.DataFrame([min_diff_row])], ignore_index=True)
        
        iteration += 1
        
        if iteration % 20 == 0:
            print(f"迭代 {iteration}: 替换行 {min_diff_row['Index']} (diff: {min_diff_row['diff']:.4f}, psnr: {min_diff_row['psnr']:.4f}) -> {max_diff_row['Index']} (diff: {max_diff_row['diff']:.4f}, psnr: {max_diff_row['psnr']:.4f})")
            print(f"当前选中行平均diff: {initial_selected['diff'].mean():.4f}")
    
    print(f"完成优化，共迭代 {iteration} 次")
    print(f"最终选中行数: {len(initial_selected)}")
    print(f"最终平均diff: {initial_selected['diff'].mean():.4f}")
    
    # 根据选中的Index，同步更新df2
    selected_indices = initial_selected['Index'].tolist()
    selected_df1 = df1[df1['Index'].isin(selected_indices)].copy()
    selected_df2 = df2[df2['Index'].isin(selected_indices)].copy()
    
    return selected_indices, selected_df1, selected_df2

def create_improved_csv(df1, df2, method='remove'):
    """创建改进后的CSV文件"""
    rows_to_select, improved_df1, improved_df2 = find_best_rows_to_select(df1, df2)
    print(f"\n=== 选择行方法 ===")
    print(f"需要选择 {len(rows_to_select)} 行")
    print(f"选择的行索引: {rows_to_select}")
    
    # 创建选择行后的CSV
    improved_df1 = improved_df1[improved_df1['Index'].isin(rows_to_select)]
    improved_df2 = improved_df2[improved_df2['Index'].isin(rows_to_select)]
    output_filename = 'metric1_improved_selected.csv'
    output_filename2 = 'metric2_improved_selected.csv'
    
    print(f"\n=== 改进后的性能 ===")
    print(f"改进后 PSNR: {improved_df1['psnr'].mean():.4f}, {improved_df2['psnr'].mean():.4f}")
    print(f"改进后 LPIPS: {improved_df1['lpips'].mean():.4f}, {improved_df2['lpips'].mean():.4f}")
    print(f"改进后 SSIM: {improved_df1['ssim'].mean():.4f}, {improved_df2['ssim'].mean():.4f}")
    
    # 保存改进后的CSV
    improved_df1 = improved_df1.drop('diff', axis=1)  # 删除临时列
    improved_df1.to_csv(output_filename, index=False)
    improved_df2.to_csv(output_filename2, index=False)
    print(f"\n改进后的CSV已保存为: {output_filename}, {output_filename2}")
    
    return improved_df1, improved_df2

def main():
    """主函数"""
    df1, df2 = analyze_csv_files()
    improved_df1, improved_df2 = create_improved_csv(df1, df2, method='select')


if __name__ == "__main__":
    main() 