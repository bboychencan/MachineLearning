# Thompson Sampling 汤普森采样

## Beta 分布
是一个作为伯努利分布和二项式分布的共轭先验分布的密度函数，在机器学习和数理统计学中有重要应用。在概率论中，贝塔分布，也称Β分布，是指一组定义在(0,1) 区间的连续概率分布。
因为x区间正好在0-1上，所以很适合描述概率分布

beta 分布直观上可以理解为概率的概率。 即对于一个未知的概率时间概率的估计。
最简单的例子就是对于观测到A次正面，B次反面的硬币，对硬币出现正反面的概率p的概率分布就是beta分布。
当beta分布中alpha, beta都取1时，就成了均匀分布。

这里面使用二项分布来估算出现观测到A次正面，B次反面的概率。 然后就可以得到硬币概率p的一个似然函数。就是beta分布的概率密度函数。
beta分布的x，就是我们要估计的p，也就是硬币正面朝上的概率，alpha 就是 A+1，beta 就是B+1，也就是代表我们已经观测到的正面背面
朝上的次数。 

## 共轭先验
对于二项分布，用beta分布作为先验分布，通过贝叶斯推断之后，后验分布依然是beta分布。这种特性成为共轭先验


## Thompson  采样
其实就是假设每个拉杆的奖励服从一个特定的分布，然后计算每个拉杆的期望奖励来选择。但是直接计算每个拉杆的期望奖励计算代价比较高。
汤普森采样算法使用采样的方式，根据每个动作a的奖励分布进行一轮采样，得到一组各个拉杆的奖励样本，选择奖励最大的做操作。 
本质上就是一中计算每个拉杆产生最高奖励概率的蒙特卡罗采样方法。

具体是现实，对每个动作a的奖励分布用beta分布建模，具体来说，如果某拉杆选择了k次 m1次奖励为1，m2次奖励为0，该拉杆奖励服从
(m1 + 1, m2 + 1) 的beta分布。 并且每次实验完之后再更新对应拉杆的参数。

这个看一下代码就知道，其实很简单了
```python
def pull(N, epsilon, P):
    """通过epsilon-greedy来选择物品

    Args:
        N(int) :- 物品总数
        epsilon(float) :- 贪婪系数
        P(iterables) :- 每个物品被转化率
    Returns:
        本次选择的物品
    """
    # 通过一致分布的随机数来确定是搜索还是利用
    exploration_flag = True if np.random.uniform() <= epsilon else False

    # 如果选择探索
    if exploration_flag:
        i = int(min(N-1, np.floor(N*np.random.uniform())))

    # 如果选择利用
    else:
        i = np.argmax(P)
    return i

def trial_thompson(Alpha, Beta, rounds=T):
    """做rounds轮试验

    Args:
        Alpha, Beta(iterables) :- 每个物品被转化率的Beta分布的参数
        rounds(int) :- 一共试验的次数
    Returns:
        一共的转化数

    rewards来记录从头到位的奖励数
    """
    rewards = 0
    for t in range(rounds):
        P_thompson = [np.random.beta(Alpha[i], Beta[i]) for i in range(len(Round))]
        i = pull(N, epsilon, P_thompson)
        Round[i] += 1
        reward = np.random.binomial(1, P[i])
        Alpha[i] += reward
        Beta[i] += 1 - reward
        rewards += reward
    return rewards
```
