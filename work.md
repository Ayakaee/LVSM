7.15
1. iamge encoder: dino pe-core none
2. 修复inference bug 原因：模型加载时出错，torch.compile保存的模型state_dict键值多出_orig_mod前缀
3. 确保loss正确 原代码正确，所有输出图片都计算loss
4. normalize 与模型代码预处理保持一致
5. attention dropout 默认为0
6. eval的可视化放到单独文件夹

7.16
改正Attention
优化encoder

7.17
1. 增加patch interpolate选项
2. 参数不冻结选项
3. 层数减少
4. dataloeader time
5. 增加dino-l