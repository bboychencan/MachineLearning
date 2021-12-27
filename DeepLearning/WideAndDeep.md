# Wide and Deep

2016年tensorflow发布的用于分类和回归的模型，应用到googleplay的应用推荐中。
核心思想是结合线性模型的记忆能力和dnn模型的泛化能力。训练过程中同时优化2个模型的参数
从而达到整体模型的预测能力最优

1. 记忆memorization即从历史数据中发现item或者特征之间的相关性
2. 泛化generalization即相关性的传递，发现在历史数据中很少或者没有出现的新的特征组合

wide部分就是LR
deep部分是DNN

## cross product 叉乘特征
在wide部分，需要大量的使用原始sparse特征和叉乘特征。很多原始的dense特征通常也会被分桶离散化构造sparse
特征。这种做法的优点是模型可解释性高，实现快速高效，特征重要度易于分析。

不过wide部分的memorization需要很多的人工设计


## 模型结构
wide 部分就是直接线性，特征用的是曝光app和安装app的叉乘(具体手动叉乘是多少阶，看论文前面章节讲的应该是所有的高阶组合都包含了)
deep部分用的continous保留，用CDF归一化到【0，1】之间，类型数据映射到32纬embedding，和原始的continous共1200维作为NN输入

## optimizer
wide部分用了FTRL算法加上L1正则化
deep部分用的是AdaGrad优化

## FTRL

## AdaGrad

## CDF
