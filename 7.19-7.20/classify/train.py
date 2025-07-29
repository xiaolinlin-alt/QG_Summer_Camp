from mindspore import dataset as ds
from mindspore.dataset import GeneratorDataset
from transformers import BertTokenizer
from model import SimpleBART
from mindspore import nn, Tensor, grad, context
import mindspore as ms
import numpy as np
from mindspore import value_and_grad
from mindspore import ops
import os
from text_data import text_data

context.set_context(mode=context.GRAPH_MODE)

type_dic = {
    "首次办理身份证": 0,
    "户口迁移": 1,
    "水电费缴纳": 2,
    "医保参保": 3,
    "缴纳个人所得税": 4,
    "其他":5,
}

text_data=text_data

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def generate_data(text_data, label_dict):
    for text, label in text_data:
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        # 确保返回正确的形状 [seq_len]
        yield encoding['input_ids'][0], label_dict[label]


# 创建数据集
train_dataset = GeneratorDataset(
    source=generate_data(text_data, type_dic),
    column_names=["input_ids", "labels"],
    shuffle=True
)
train_dataset = train_dataset.batch(batch_size=2)

# 创建模型 (添加vocab_size参数)
model = SimpleBART(
    num_layers=6,
    hidden_size=768,
    num_heads=12,
    ff_size=2048,
    num_classes=len(type_dic),
    vocab_size=tokenizer.vocab_size
)

# 优化器和损失函数
optimizer = nn.AdamWeightDecay(
    model.trainable_params(),
    learning_rate=1e-5,
    weight_decay=0.01
)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 定义前向函数和梯度函数
def forward_fn(input_ids, labels):
    logits = model(input_ids)
    loss = loss_fn(logits, labels)
    return loss

grad_fn = value_and_grad(forward_fn, None, model.trainable_params())

# 训练循环
for epoch in range(10):
    model.set_train(True)
    total_loss = 0.0
    total_samples = 0

    for data in train_dataset:
        input_ids, labels = data
        # 确保输入是int32类型
        input_ids = input_ids.astype(ms.int32)
        labels = labels.astype(ms.int32)

        # 计算损失和梯度
        loss, grads = grad_fn(input_ids, labels)

        # 参数更新
        optimizer(grads)

        total_loss += loss.asnumpy()
        total_samples += input_ids.shape[0]

    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
# 保存模型
ms.save_checkpoint(model, "bart_classifier.ckpt")


# 分类函数修正
def classify_using_trained_model(text, label_dict):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )
    # 确保输入形状正确 [1, seq_len]
    input_ids = Tensor(encoding['input_ids'], ms.int32)

    model.set_train(False)
    logits = model(input_ids)
    prediction = np.argmax(logits.asnumpy(), axis=-1)

    # 反转字典映射
    reverse_dict = {v: k for k, v in label_dict.items()}
    return reverse_dict[prediction[0]]

# 测试示例
text = ("身份证")
result = classify_using_trained_model(text, type_dic)
print(f"分类结果: {result}")