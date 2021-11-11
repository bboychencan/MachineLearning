# LIBSVM 数据格式

2021-11-11

在用spark训练xgb模型的时候遇到这个数据格式，据说很方便。 需要学习一下

其实就是
[label] [index1]:[value1] [index2]:[value2] [index3]:[value3]

label  目标值，就是说class（属于哪一类），就是你要分类的种类，通常是一些整数。
index 是有顺序的索引，通常是连续的整数。就是指特征编号，必须按照升序排列
value 就是特征值，用来train的数据，通常是一堆实数组成。
