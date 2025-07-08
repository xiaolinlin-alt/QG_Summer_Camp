* 深度学习——图像处理

# 卷积神经网络

## 全连接层

神经元共同连接

BP算法包括信号的前向传播和误差的反向传播两个过程

![5ff7e4afd5cc13da14b1a14a00022b6](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070909968.jpg)

![44959ebb4f4236afd3292161fdceb49](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070909733.jpg)实例：利用BP神经网络做车牌数字识别

![504eff8aa7226d864cb268aff15a90a](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070909639.jpg)

## ![3d5410076905fb904606fa93a29103e](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070909377.jpg)

![35cfec0a02f15a09bcbb0fa82be1ca0](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070910090.jpg)

![970a37eb38b9241f91ad4bc2a70bf3b](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070910598.jpg)

![1b7d967caecb704b8535923c525d194](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070910092.jpg)

## 卷积层

特征：拥有局部感知机制

权值共享

![ddfc5f21975fef3ecea654d1103c775](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070911657.jpg)

![580231959418b5c5cc50f322e4ee8e5](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070911867.jpg)

1.怎么加上一个偏移量：卷积核那里加的话体现在输出矩阵那里加



卷积核深度与输入特征层的深度个数相同，比如上面那个RGB,它输入层个数是三，卷积核个数也是对应是三个

![87818e0fc0333efb57ffccc3a7b4e66](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070911809.jpg)

![c624ee097aa8fe597092254b822de99](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070912885.jpg)

目的：进行图像特征提取

卷积核的深度和输入特征层的深度相同

输出的特征矩阵深度与卷积核的个数相同

## 池化层

![60f93424f70d706afc7d17860735700](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070912925.jpg)

平均下采样层：找它们的平均值

池化层特点：

1.没有训练参数，只是在原始特征上求最大值或者求平均值

2.只改变特征矩阵的宽度和高度，并不会改变它的深度

3.一般池化核大小和布局是相同的

这样子就可以将它的大小缩小成原来的一半



## 反向传播过程

误差计算

![20c7e714ed7c89b4dff958a208a204c](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070953150.jpg)

![9e239ca36e2df46ad70cd8ceb620ba0](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070953862.jpg)

![20078c87b0f6cf386162801cee92700](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070953252.jpg)

![](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070953505.jpg)

![03942840cfa5664c1b02472c55b607a](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070953196.jpg)

![7e38e0543011808bc8a466ba44dbc23](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070954081.jpg)

![208e1d2cdbf875fcd89ba2695c10f8a](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070954021.jpg)

![30d13fdde70a498d5746dbd63099bf0](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070954185.jpg)

![f284f780cbfa2f3f87380b1acac79a6](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070954010.jpg)

![69fd5e8bd1b4c6db44f8b7158ea071e](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070954265.jpg)

![b7a964879d38d74edee54940396c5e6](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070954458.jpg)

![8a9c64cbf8ccb5ef26af8ad597ed73c](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507070954714.jpg)



# LeNet

Pytorch tensor的通道排序：[batch,channnel,height,width]

根据这个顺序

![image-20250707101756258](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507071017325.png)

因此在这里

一批图像的个数-channel（图片深度3）-高-宽

项目实战-test1



# ALexNet

![image-20250707145625324](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507071456500.png)

![image-20250707145730145](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507071457257.png)

![](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507071458654.png)

![image-20250707145855657](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507071458788.png)

这里为了简化，先看一个的



花分类数据集

![image-20250707194316967](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507071943135.png)

每个层详细的参数

Sequential用来层数比较多的，用来精简代码

这里激活函数用的是RELU函数

然后根据这个用Sequential串起来，然后用dropout防止过拟合

在展平操作之后与我们的全连接层进行

这里需要注意flatten和view其实并不一样，

![image-20250707203117174](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507072031211.png)

这里用view的话，是

~~~python
x=x.view(x.size(0),-1)
~~~

用flatten的话，是

~~~
x=torch.flatten(x,start_dim=1)
~~~

项目实战-test2-花分类



# VGG

亮点：通过**堆叠多个3×3的卷积核**替代大尺度的卷积核（减少所需的参数）

论文中提到，可以通过堆叠两个3×3的卷积核替代5×5的卷积核，堆叠三个3×3的卷积核替代7×7的卷积核

拥有相同的感受野

![image-20250707215653917](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507072156089.png)

我们用的比较多的是第四个

啥是感受野：

在卷积神经网络中，决定某一层输出结果中一个元素所对应的输入层的区域大小，被称为感受野。

通俗来说，就是

输出feature map上的单元对应的输入层上的区域大小

![image-20250707215907670](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507072159826.png)

![image-20250707220838088](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507072208232.png)

![image-20250707220903955](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507072209093.png)

![image-20250707221710202](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507072217317.png)

项目实战3-VGG

用VGG搭建上图ABCD四个模型

![image-20250708110923087](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507081109301.png)

项目受限于配置，因此训练轮次少，模型跑出来准确率低



#  GoogleLeNet

**亮点：**

引入了inception结构（融合不同尺度的特征信息）

使用1*1的卷积核进行降维以及映射处理

添加两个辅助分类器帮助训练

丢弃全连接层，使用平均池化层（大大减小模型参数）

![image-20250708205022760](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507082050154.png)

相当于以前的神经网络只能用一个放大镜看图片，而这个可以用多个放大镜看图片

Inception模块就是想模拟人眼观察照片时候的这种多尺度处理

![image-20250708211530707](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507082115926.png)

1. **1x1 小刀**：快速扫描基础特征（比如颜色斑点）

2. **3x3 中刀**：识别中等特征（比如眼睛轮廓）

3. **5x5 大刀**：识别大特征（比如整个猫头）

4. **池化工具**：压缩信息保留重点

   ![image-20250708211736959](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507082117010.png)

1. **1-10层**：普通楼层（基础特征提取）
2. **11-21层**：9个 Inception 工作室（核心区域）
3. **22层**：全局观景台（全局平均池化）
4. **决策室**：全连接层（输出结果）

当 GoogleNet 看到猫图：

1. **第一层工作室**：同时发现"毛茸茸纹理"+"尖耳朵轮廓"+"胡须斑点"
2. **中间工作室**：组合成"猫耳+猫脸"特征
3. **顶层工作室**：确认这是"蹲着的橘猫"
4. **决策室**：输出"猫咪！概率 99%"



# git多人协作

![image-20250708221916051](https://yuyingcun.oss-cn-guangzhou.aliyuncs.com/typora/202507082219465.png)





