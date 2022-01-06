# 深度模型的正则化

## Dropout

## Early Stop
基本作用就是当模型的验证集上的性能不再增加的时候就停止训练，从而达到充分训练的作用，避免过拟合

# 深度模型的正则化

## Dropout

## Early Stop
tf.keras提供了很方便的早停的方法，一个简单的应用场景就是，加入验证集的loss作为指标，然后每个回合
训练完后检验一下val_loss，如果val_loss的变化范围（应该是降低幅度）在min_delta之内，则认为没有改进
否则，就是有改进。 如果连续若干步都被认为没有改进的话就提前停止训练。

## Weight Decay

## Data Augmentation

## Weight Decay

## Data Augmentation
