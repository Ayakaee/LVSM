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

把I;P先经过两层attn编码，再输入到decoder中
有F的时候，I是否有必要
repa的projector换成attention

目前模型效果：
1. 同步数下，效果不如lvsm
2. 同时间下，比优于lvsm
3. scale到模型极限优劣未知

Target view数量对训练有影响，epoch变少，循环变慢，可能让训练变慢
测一下input的泛化性，还有可变input的训练效果


self:layer0-3语义，整体norm大小无明显变化150
layer4-5在norm在几个点上非常非常大1600，这些点似乎呈网格状分布

cross:layer0呈现网格状，似乎只看plucker和语义无关
cross:layer1 norm集中在input view的边缘
cross:layer2 norm集中在target view的边缘
cross:layer3-5都比上一层norm增大60左右
layer3开始有轮廓

没有exenc的情况下，self layer0 有一些网格状， norm尖峰在Self-layer5出现
cross layer0网格更多，layer5的norm更大350， layer4开始有轮廓

lvsm没有norm尖峰

repa enc和self-attn的norm高斯分布，但是pca特征是正确的，确实学到了dino，cos和mse都很好
cross蒸馏output太靠前，学不到，这一层之后重新开始，网格持续到layer2（正常layer1）

none;embed 模型的self-attn有pca横着的条纹，而repa的没有，疑似plucker?但是在none;exenc=2的模型中就弱很多，layer3就看不到了

repa蒸馏input之后，self-attn的语义特征更细腻

提出了一种双向transformer架构，在decoder的每一层同步更新input token，让输入图片解耦，对输入输出数量鲁棒，高效训练和推理的模型

abstract:简单讲，八股，
introduction:：
- 研讨，问题是什么，最近很火，
- 已有方法有问题，引用，（给出一个可视化图或者表）
- 获得启发，提出xxx改进。实验跑了啥，贡献是什么

- related work
- 阐述问题定义，（我们的更应用化）定义好符号，说明全局的setting，输入输出分别是什么，任务是什么
- 为了解决xxx，可能的架构（大的范式而不是小区别），哪个是最好的
- 实验

https://www.overleaf.com/1248324414kgtznzspdqmf#fee15c
https://www.overleaf.com/8873597924gjvmswvdpqwn#a1de62 去年的写作
https://www.overleaf.com/7526643262ghrwsnkmwdjp#8ec810
https://www.overleaf.com/7334783184qthtnrfpxzsq#912dc3
https://www.overleaf.com/read/hgcbjhsjmxjb#87f2cf

exp1
相同训练时间（eg.2卡1天）
点数>lvsm
说明有训练效率的优势

exp2
和lvsm一样的训练时间scale up
（大概率）：如果lvsm训练3天，我们的训练2天就收敛
如果收敛的模型点数不比lvsm差
说明模型本身能力合格