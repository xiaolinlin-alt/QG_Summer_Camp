# Transformer

## 介绍

实现一个语音识别系统

**一个基于注意力机制的网络主干结构**

序列建模任务：唇语识别、语音识别、手语识别、机器翻译......

## 架构

* Transformer包含编码器、解码器
* 编码器负责特征编码，从原始的，比较底层的输入序列信号提取出抽象的、具有明显语义特征的特征信息

* 解码器负责从编码器得到的原始序列的特征信息中破译出目标序列中的内容

可以直白理解成编码器负责从一个内容中提取信息，而解码器是负责找到提出出来的信息对应的原内容



下图是一个完整的transformer的架构

![image-20250712194659451](C:\Users\linyu\AppData\Roaming\Typora\typora-user-images\image-20250712194659451.png)

##  输入和输出

### 输入：

- X：输入特征                  

   尺寸：**（B,T,d）**          

    **(batch_size(一个训练轮次中模型接收的样本数量)，T(当前batch中各音频序列的最大长度)，d(特征维度))**

* X_lens：B个样本各自的序列长度

​	尺寸：(B,)

由于不同样本的序列长度不同，所以我们为了让模型处理不同长度的音频

将同一个批次中的输入填充（**padding**）到一样的长度

这些填充的部分后续计算损失函数时会被丢掉，所以填充是什么值关系并不大，我们**一般都填充为0**

### 输出：

* Y:输出文本后验概率分布

​	尺寸：（B,T',V）

​	(batch_size(一个训练轮次中模型接收的样本数量)，T'(当前batch中的文本序列的最大长度)，V(可能的候选字个数，词表大小))

* Y_lens:各个文本序列的长度

  尺寸：（B,）

  下图Y_lens=(4,6,8)

![image-20250712202008436](C:\Users\linyu\AppData\Roaming\Typora\typora-user-images\image-20250712202008436.png)

## 注意力机制

### 理解

类比人的注意力，神经网络也同样有记忆力

只不过方式不一样

Transformer，是**直接以注意力机制为主要构成模块的主干网络**。

举个例子;

“早上好！”，我们翻译成“Good Morning！”，其中“Good”和“好”关联性最强，和其他两者关联性比较弱，同理可推其他

### 实现

那transformer中注意力机制是怎么实现的啊？

transformer中包括：

**Q：Query**

**K：Key**

**V：Value**

V是我们手头已经有的所有资料，可以作为一个知识库

Q是我们待查询的东西

我们希望把V中和Q有关的信息都找出来

K是V这个知识库的钥匙

V中每个位置的信息对应于一个K

对于V中每个位置的信息而言，如果Q和对应钥匙的匹配程度越高，那么就可以从该条信息中找到和Q更多的内容

举个例子，

我们现在希望给四不像找妈妈。以四不像作为 ，以[鹿 ，牛 ，羊 ，驴，青蛙 ]同时作为V和K，然后发现四不像和鹿的相似度为1/3、和牛的相似度为1/6，和羊、驴的相似度均为1/4，和青蛙的相似度为0，那么最终的查询结果就是1/3鹿+1/6牛+1/4羊+1/4驴+0青蛙

步骤：

1.计算Q和K的相似度

2.根据计算得到的相似度，取出V每条信息中和Q有关的内容

tranformer中的注意力计算方法：

![image-20250712211227609](C:\Users\linyu\AppData\Roaming\Typora\typora-user-images\image-20250712211227609.png)



Q的序列长度为3，即共有3个元素

K和V的序列长度为4，即分别有四个元素

QKV的特征维度均为2，我们现在希望计算V中和Q有关的信息，QKV的每一个元素都是一个长度为2的行向量

![image-20250712211457742](C:\Users\linyu\AppData\Roaming\Typora\typora-user-images\image-20250712211457742.png)

计算Q和K的相似度：Q1=[0.3,0.3],K3=[0.9,0.9]

则相似度为
$$
Q_1K_3^T=0.3*0.9+0.3*0.9
$$

[^相似度计算方法]: 其实没有特定标准，但是现在主流就是直接两个向量做内积

记Q和K的相似度计算结果为score，score是一个3*4的矩阵，因为Q中有三个元素，Q中有四个元素，所以
$$
score=QK^T
$$
attention本质上是一个概率分布
$$
attention=softmax(QK^T)
$$
取出V中每条信息中和Q中有关的内容

然后运算结果是O=attentionV(它们之间的点积)

## 注意力机制解释

1.注意力机制本质上可以认为是求一个离散概率分布的数学期望

2.多头注意力机制：多头注意力机制假设输入的特征空间可以分为互不相交的几个子空间，然后我们只需要在几个子空间内单独计算注意力，最后将多个子空间的计算结果拼接即可。

举个例子，假设Q、K、V的维度都是512，长度都是 L，现将维度512的特征空间分成8个64维的子空间，每个子空间内单独计算注意力，于是每个子维度的计算结果的尺寸为(L,64) ，然后再将8个子维度的计算结果沿着特征维度拼起来，得到最终的计算结果，尺寸为 (L，512)，和输入的 Q的尺寸保持一致。

多头注意力机制目前已经是Transformer的标配，通常每个注意力头的维度为64，注意力头的个数为输入维度除以64。

~~~~python
q_len, k_len, v_len = ...
batch_size, hdim = ...
head_dim = ...
assert hdim % head_dim = 0
assert k_len == v_len
num_head = hdim // head_dim
q = tensor(batch_size, q_len, hdim)
k = tensor(batch_size, k_len, hdim)
v = tensor(batch_size, v_len, hdim)


def multi_head_split(x):
  # x: (batch_size, len, hdim)
  b, l, hdim = x.size()
  x = x.reshape(b, l, num_head, head_dim).transpose(1, 2)    # (b, num_head, l, dim)
  return x


def multi_head_merge(x, b):
  # x: (batch_ize, num_head, len, head_dim)
  b, num_head, l, head_dim = x.size()
  x = x.transpose(1, 2).reshape(b, l, num_head * head_dim)    #(batch_size, l, hdim)
  return x


q, k, v = map(multi_head_split, [q, k, v])
output = MultiHeadAttention(q, k, v)      # 该函数的具体实现后文给出
output = multi_head_merge(output, batch_size)
~~~~

![image-20250712222153826](C:\Users\linyu\AppData\Roaming\Typora\typora-user-images\image-20250712222153826.png)



## 掩码机制

本节介绍Transformer中的掩码机制，主要包括三部分：

- encoder的self attention的长度mask
- decoder的self attention的causal mask
- encoder和decoder的cross-attention的mask



* encoder的self attention的长度mask

在上文我们提到，序列输入的长度可能是不一样的，我们当时的处理是将同一个batch中的所有样本都padding到最长长度，这样可以解决输入变长的问题，但是这样做的话，attention的计算会不会出问题？举个例子，当前batch的最大长度是10，而当前样本的长度为4，也就是说序列最后6个位置的数据都是padding填充的，应当舍弃，然而在计算自注意力的过程中，由于Q、K 的长度都为10，所有最终计算出的attention长度也是10，其中包含了 Q和K 中padding部分的6个位置的attention计算结果。关于这一点，Transformer中采取的方法为掩码机制 (masking)

我们的目的是为了让attention的计算结果不受padding的影响，那么一个比较简单的方法是，**直接将padding对应位置的attention权重置为0即可。**

![image-20250712221259062](C:\Users\linyu\AppData\Roaming\Typora\typora-user-images\image-20250712221259062.png)