# DeepFM
2021-12-30
该模型可认为是Wide & Deep推荐模型的升级版
和wide & deep模型类似，DeepFM同样由浅层模型和深层模型联合训练得到，区别主要一下两点

1. wide模型部分LR替换为FM，FM可以自动学习交叉特征，避免了原始wide部分人工特征工程的工作
2. 共享原始输入特征，DeepFM模型的原始特征将作为FM和Deep部分的共同输入，保证模型特征的
准确与一致。

## 主要贡献
DeepFM模型，可以从原始特征中抽取到各种复杂度特征的端到端模型，没有人工特征工程的困扰。
1. 模型包含FM和DNN两部分，FM可以抽取low-order特征，DNN抽取high-order特征
2. 输入仅为原始特征，FM和DNN共享输入向量，训练速度很快
3. 在benchmark数据集上，DeepFM超过目前所有模型
y = sigmoid(yFM + yDNN)
yFM = <w, x> + sum(<vi, vj>xj1 * xj2)

## 子网络部分
输入原始特征，多数是高纬度，稀疏，连续和类别混合的分域特征，为了更好的发挥DNN学习high-order
特征的能力，文中设计了子网络结构，将原始的稀疏表示映射为稠密的特征向量

<vi, vj>xj1 * xj2
