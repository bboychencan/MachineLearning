# Pointwise

单文档方法，用户i的查询query为qi，候选文档集docs为c1, c2, c3。返回topK

## pointwise
排序作为二分类问题，训练样本作为三元组(q, cij, yij) ，yij是0，1表示是否正确选中
训练的目标就是最小化数据集中所有q和c对的交叉熵
长用的方法McRank

## 缺点
1. ranking追求的是排序结果，并不要求精确打分，只要有相对打分即可
2. pointwise类方法并没有考虑同一个query对应的docs间的内部依赖性。 导致输入空间内的样本不是IID
违反了ML的基本假设。 ？
3. 损失函数没有预测排序中位置信息，
4. query与doc间的不平衡，有的query对应很多文档，有的则很少



## 广告ctr
广告场景下，不仅仅需要排序，还需要精准地预估点击概率ctr，因此这个就是一个LTR pointwise的运用。
是一个标准的而分类问题。
