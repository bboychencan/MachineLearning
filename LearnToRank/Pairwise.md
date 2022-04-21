# Pairwise

pairwise中排序模型需要让正确的回答doc得分明显高于错误的doc。 训练样例是 (qi, ci+, ci-)。c+
为正确答案，c-为错误答案
损失函数是hinge loss 合页损失函数
L = max{0, m - h(qi, c+) + h(qi, c-)}

损失的目标是促使正确答案的得分比错误答案的得分大于m。 预测阶段得分最高的候选答案被当作正确答案

## 缺点
1. doc pair数量级增加到二次。 query与doc数量不平衡的问题将被放大
2. pairwise相对pointwise对噪声更敏感
3. pairwise仅考虑了doc pair的相对位置，没有model预测排序中的位置信息
4. 没有考虑query和doc pair间的内部依赖性，输入空间样本不是IID



