## XGBoost: A Scalable Tree Boosting System
# 0 摘要
- 树提升模型有效且被广泛应用
- 本文提出一个可扩展的端到端的提升树xgboost
- 全新的“稀疏感知“算法用于稀疏数据，weighted quantile sketch做树学习（？？）
- 缓存模型，数据压缩，及分区方面提供了见解，使得xgb可以在比现有系统使用少得多的资源扩展到几十亿的样本

# 1 导论
- 机器学习广泛应用，垃圾邮件过滤，广告，银行欺诈检测等
- 以上成功取决于两个因素：有效的模型补货复杂的数据依赖，可扩展学习系统利用大规模数据
- 梯度树提升模型比较好用，如LambdaMART，netflix prize用到的模型
- kaggle 2015年17/29个胜利模型使用xgb，涵盖的比赛有销售预测，高能物理现象分类，用户行为预测，运动监测，点击预估等
- 最重要的因素，扩展性，该系统10倍快于现存的单机系统，可以扩展到10亿样本级
- 扩展性取决于几个重要的系统和算法优化。包括新的处理稀疏数据的新算法； a theoretically justified weighted quantile sketch procedure enables handling instance weights in approximate tree learning？；并行及分布式计算，使得模型探索更快速；最重要，利用了out-of-core计算，可以在淡季处理上亿样本
- 主要贡献
1.设计构建端到端高扩展性提升树系统
2.理论证明的weighted quantile sketch 进行高效率计算
3.提出全新的稀疏感知算法
4.提出有效的缓存感知块结构用于out-of-core树学习

# 2 树提升模型介绍
回顾一下树提升模型，推导过程与现有梯度提升理论推导思路相同。具体的二阶方法起源于Friedman，我们在正则目标上做了一点小小的改进

## 2.1 正则化学习目标
- 定义数据集n个样例，m个特征，
$$
\mathcal{D} = \{(x_i, y_i)\} (|\mathcal{D}| = n, x_i \in  \mathbb{R^m, y_i \in \mathbb{R}})
$$
- 一个集成树模型使用 $K$ 个可加的函数预测输出
$$
\hat{y_i} = \phi(x_i) = \sum_{k=1}^{K}f_k(x_i), \space f_k \in \mathcal{F} \\
\mathcal{F} = \{f(x) = w_{q(x)} \}(q: \mathbb{R^m} \to T, w \in \mathbb{R^T})
$$是回归树空间（CART）$q$表示树的结构，将样本映射到树节点，$T$表示树的叶子个数，每个$f_k$对应一个独立的树结构$q$和叶子权重$w$
与决策树不同，回归树每个叶子结点包含一个连续的数值$w_i$表示第$i-th$个叶子的权重。根据树$q$的规则将样本划分到叶子，然后把所有树预测的值求和得到预测结果。
- 为了学到这些函数，最小化下面的正则化目标函数
$$\mathcal{L}(\phi) = \sum_i l(\hat{y_i}, y_i) + \sum_k \Omega(f_k) \\
where \space \Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2
$$
另一种类似的正则化技术叫做RGF。 这个目标和算法比RGF简单且更容易并行，当正则项设为0时，目标退化成传统的梯度树提升

## 2.2 梯度树提升 Gradient Tree Boosting
上面的集成树模型包含了函数作为参数，无法通过传统的欧式空间方法优化，取而代之，模型通过叠加的方法训练。