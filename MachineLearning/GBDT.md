# GBDT
GBDT(Gradient Boosting Decision Tree) 又叫 MART（Multiple Additive Regression Tree)，是一种迭代的决策树算法，该算法由多棵决策树组成，所有树的结论累加起来做最终答案。它在被提出之初就和SVM一起被认为是泛化能力较强的算法。

GBDT中的树是回归树，一般用来做回归预测，调整后也可以做分类。

常用做提高CTR预估的准确性，在搜索预测上常发挥重要作用。

拿回归来举例
- 基础分类器用的是CART
- 每次拟合上次结果的梯度（如果损失函数是平方差，则正好是拟合残差）
- 预测结果等与所有树结果的叠加
