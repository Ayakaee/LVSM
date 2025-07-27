实验要有备份:.py+.yaml
训练输出有有日志

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

1. zero init dino->self-attn
2. dino -> self-attn
3. rgb plucker 对齐到448输入给dino target不变

benchmark设置：以时间为标准，显存拉满，batch size有多大看能力
如何解释input=2效果不如LVSM

把self-attn换成FFN -> input特征需要背更新，但是不需要看周围的patch