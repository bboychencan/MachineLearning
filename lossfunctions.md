# Loss Functions

## Hinge Loss (外形很像hinge，中文里面的合页)

SVM中使用的损失函数

L(y) = max(0, 1 - y * t) 
t是目标值（1,-1), y是预测值 。 
- 当预测结果错误的时候，y * t为负数，L(yi) 返回 1 - y * t，与y值成线性关系
- 当预测正确且|y| >= 1时，损失为0

## Cross Entropy Loss

## Softmax Loss

## MSE

## Exponential Loss