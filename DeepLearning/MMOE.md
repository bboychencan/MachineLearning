# Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

一般情况下，都是单任务模型，一个任务训练一个模型，如果需要完成多个任务的话，就构建多个模型。
使用不同的input训练，不同的模型，不同的label，不同的预测结果。 在广告推荐中，就遇到这个问题。
我们面临不同的业务，每个业务都要建立一个独立的模型，而即使同一业务，有的需要预测点击，有的
需要预测拉端，有的要预测注册，有的是呼单，如果每个任务都独立建立一任务，就很繁琐。有的场景
下我们希望同时优化多个目标，比如推荐。 我们希望提高点击率，同时希望提高视频播放时长，视频
点赞转发等。这些目标并不一定是相辅相成，有的可能是互相竞争。

## shared-bottom 架构
多数的多任务模型，基于shared-bottom架构，不同任务共用统一的底层，这种结构可以减少过拟合的风险
然后到上层，分化成不同的任务，类似于迁移学习，在图像领域很常见。底层的数据很多是图片的基本像素
纹理等特征，跟具体任务不是特别相关，很容易共享。 但如果不同任务之间差别很大，就比较难了。

在这个背景下，为了进行**不相关任务**的多任务学习，很多的工作见效甚微，直到MMoE

## MoE 共享层划分为多个Expert 引入Gate机制
首先理解MoE，是将一个统一的shared底层，分成n个expert network，然后每个expert对应一个gating network
将gate的输出乘以每个expert的输出，加权求和就得到最终的结果。 这个是MoE的概念。
其中gating network就是生成n个experts的概率分布，（应该是softmax）

## MMoE
目的就是相对于shared-bottom结构不明显增加模型参数的要求下捕捉任务的不同
核心思想就是将shared-bottom网络中的函数f替换成MoE层，对于每个任务k都有对应的gating-k输出n个experts
的权重，里面就是softmax方法。 
- gating network通常是轻量级的，而expert network是所有任务共用的，在计算量和参数量上具有优势
- 相对于所有任务公共一个gate network，MMoE每个任务使用单独的gating network，每个任务的gating 
network通过最终输出权重不同实现对experts的选择性利用。不同任务的gating network可以学习到不同
的组合experts的模式，考虑到了捕捉任务的相关性和区别性。

## 总结
MoE与MMoE的共同点是把原先的Hard-Parameter sharing的底层全连接层网络划分成了多个子网络expert，这种
做法模仿了集成学习的思想，即单个网络无法学习到所有任务之间通用的表达但是通过划分得到多个expert子
网络后，每个子网络总能学到某个任务中一些相关独特的表达，再通过Gate输出（soft Max）加权各个expert输出
送入各自多层全连接就能将特定任务学习的较好 
