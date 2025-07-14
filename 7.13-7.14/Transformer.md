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

上图中，子图 (a) 表示只要 Q和 K其一为padding，那么我们就将其attention权重置为0；而子图 (b) 表示当 K为padding时将对应attention权重为0。实际模型训练过程中使用(b)而不使用(a)，使用 (b) 不会出错是因为Q 为padding的部分最终计算loss时会被过滤掉，所以 是否mask无影响。而使用(a)时，由于有些行的所有位置都被mask掉了，这部分计算attention时容易出现NaN。举个例子，我们可以将"早上好！"后面的4个位置的文本字符都用一个特殊的token "<ignore>"来填充，然后再计算交叉熵损失的过程中利用`torch.nn.functional.cross_entropy`的`ignore_idx`参数设成 "<ignore>" 将这部分的loss去掉，不纳入计算。

为了得到子图 (b) 所示的mask，我们只需要保留每个输入样本对应的样本长度即可，代码如下：

~~~
import torch

def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)


m = get_len_mask(2, 4, torch.tensor([2, 4]), "cpu")
m=m.int()
print(m)
~~~

在上面的例子中，当前batch中有两个样本，长度分别为2和4，可以看到，长度为2的样本的后两个位置都被mask掉了。

得到mask之后，我们应该怎么用才能将padding部分的attention权重置为0呢？做法是直接将 Q和K 的内积计算结果中被mask掉的部分的值置为-inf，这样的话，过一层softmax之后，padding部分的attention权重就是0了，这一点可以使用PyTorch的masked_fill函数实现。

~~~
print(m)

score=torch.manual(Q,K.transpose(-1,-2))/torch.sqrt(d_k)
if attn_mask is not None:
    score.masked_fill_(attn_mask,-1e9)
~~~

* decoder的self attention的causal mask

~~~~
def get_sbsequent_mask(b:int,max_len:int,device:torch.device)->torch.Tensor:
    return torch.triu(torch.ones((b,max_len,max_len),device=device),diagonal=1)
m=get_sbsequent_mask(2,4,"cpu")
print(m.int())
~~~~

* encoder和decoder的cross-attention的mask

~~~
def get_enc_dec_mask(
    b: int, max_feat_len: int, feat_lens: torch.Tensor, max_label_len: int, device: torch.device
) -> torch.Tensor:
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)       # (b, seq_q, seq_k)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)
~~~

掩码机制补充：

1. 要尤其注意mask是True还是False，这一点非常容易错。通常的用法是将应该被mask掉的部分置为True
2. mask的数据类型最好固定为bool



## 注意力机制完整代码

![image-20250713091607954](C:\Users\linyu\AppData\Roaming\Typora\typora-user-images\image-20250713091607954.png)

~~~~python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)

        # linear projections
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # Normalization
        # References: <<Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification>>
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask, **kwargs):
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # multi_head split
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)

        # pre-process mask
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)    # broadcast
            attn_mask = attn_mask.bool()

        # calculate attention weight
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)        # attention weights
        attns = self.dropout(attns)

        # calculate output
        output = torch.matmul(attns, V)

        # multi_head merge
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output
~~~~

## 位置编码

为了保留原始输入的顺序关系，我们需要在Transformer的输入中加入表征顺序的特征

~~~python
def pos_sinusoid_embedding(seq_len, d_model):
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()
~~~



## 逐位置前馈网络

除了注意力机制以外，为了增强模型的表示能力，作者还提出逐位置的前向神经网络（Position-wise Feed-Forward Networks），其实说白了就是一个两层的MLP

~~~
class PoswiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, p=0.):
        super(PoswiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, 1, 0)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))     # (N, d_model, seq_len) -> (N, d_ff, seq_len)
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)   # (N, d_ff, seq_len) -> (N, d_model, seq_len)
        out = self.dropout(out)
        return out
~~~



代码训练

~~~~python
from math import sqrt
import torch
import torch.nn as nn

class Self_Atteention(nn.Module):
    def __init__(self,input_d,d_k,d_v):
        super(Self_Atteention,self).__init__()
        self.q=nn.Linear(input_d,d_k)
        self.k=nn.Linear(input_d,d_k)
        self.v=nn.Linear(input_d,d_v)
        self._norm_fact=1/sqrt(d_k)

    def forward(self,x):
        Q=self.q(x)
        K=self.k(x)
        V=self.v(x)

        attention=nn.Softmax(dim=-1)(torch.matmul(Q,K.transpose(-1,-2))*self._norm_fact)
        output=torch.matmul(attention,V)
        return output
#%%
X=torch.randn(4,3,2)
print(X)
res=Self_Atteention(2,4,5)
print(res)
~~~~

## 实现

1.多头

~~~~python
#多头注意力机制
from math import sqrt
import torch
import torch.nn as nn

class Multi_Head_Attention(nn.Module):
    def __init__(self,input_d,d_k,d_v,num_heads):
        super(Multi_Head_Attention,self).__init__()
        self.num_heads=num_heads
        self.d_k=d_k
        self.d_v=d_v

        assert d_k%num_heads==0 and d_v%num_heads==0#确保在使用多头注意力机制的时候，维度匹配

        #线性投影层
        self.q=nn.Linear(input_d,d_k)
        self.k=nn.Linear(input_d,d_k)
        self.v=nn.Linear(input_d,d_v)
        #每个头的维度
        self.head_d_k=d_k//num_heads
        self.head_d_v=d_v//num_heads
        #缩放因子
        self.scale=1/sqrt(self.head_d_k)


    def forward(self,x):
        batch_size,seq_len,_=x.shape
        Q=self.q(x).view(batch_size,seq_len,self.num_heads,self.head_d_k).transpose(1,2)
        K=self.k(x).view(batch_size,seq_len,self.num_heads,self.head_d_k).transpose(1,2)
        V=self.v(x).view(batch_size,seq_len,self.num_heads,self.head_d_v).transpose(1,2)

        scores=torch.matmul(Q,K.transpose(-1,-2))*self.scale
        attention=nn.Softmax(dim=-1)(scores)
        output=torch.matmul(attention,V)
        output=output.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_v)
        return output
~~~~

```
#多头注意力机制
from math import sqrt
import torch
import torch.nn as nn

class Multi_Head_Attention(nn.Module):
    def __init__(self,input_d,d_k,d_v,num_heads):
        super(Multi_Head_Attention,self).__init__()
        self.num_heads=num_heads
        self.d_k=d_k
        self.d_v=d_v

        assert d_k%num_heads==0 and d_v%num_heads==0#确保在使用多头注意力机制的时候，维度匹配

        #线性投影层
        self.q=nn.Linear(input_d,d_k)
        self.k=nn.Linear(input_d,d_k)
        self.v=nn.Linear(input_d,d_v)
        #每个头的维度
        self.head_d_k=d_k//num_heads
        self.head_d_v=d_v//num_heads
        #缩放因子
        self.scale=1/sqrt(self.head_d_k)


    def forward(self,x):
        batch_size,seq_len,_=x.shape
        Q=self.q(x).view(batch_size,seq_len,self.num_heads,self.head_d_k).transpose(1,2)
        K=self.k(x).view(batch_size,seq_len,self.num_heads,self.head_d_k).transpose(1,2)
        V=self.v(x).view(batch_size,seq_len,self.num_heads,self.head_d_v).transpose(1,2)

        scores=torch.matmul(Q,K.transpose(-1,-2))*self.scale
        attention=nn.Softmax(dim=-1)(scores)
        output=torch.matmul(attention,V)
        output=output.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_v)
        return output
```

输入: (batch_size, seq_len, input_dim)
↓ 线性投影
Q: (batch_size, seq_len, d_k)
K: (batch_size, seq_len, d_k)
V: (batch_size, seq_len, d_v)
↓ 多头拆分
Q: (batch_size, num_heads, seq_len, head_dim_k)
K: (batch_size, num_heads, seq_len, head_dim_k)
V: (batch_size, num_heads, seq_len, head_dim_v)
↓ 注意力计算
scores: (batch_size, num_heads, seq_len, seq_len)
attn_weights: (batch_size, num_heads, seq_len, seq_len)
output: (batch_size, num_heads, seq_len, head_dim_v)
↓ 多头合并
output: (batch_size, seq_len, d_v)
↓ 输出投影
output: (batch_size, seq_len, d_v)











