import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.features = nn.Sequential(  # 定义一个顺序模型，储存卷积、标准化和非线性激活函数层等特征提取层
            nn.Conv2d(3, 64, kernel_size=3, stride=2),  # 2维卷积层，输入通道数为3，输出通道数为64，卷积核大小为3x3，步长为2
            nn.BatchNorm2d(64),  # 2维批量标准化层，输入通道数为64
            nn.ReLU(inplace=True),  # ReLU激活函数层，在原位进行操作（inplace=True）
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，池化核大小为3x3，步长为2
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # 2维卷积层，输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为2
            nn.BatchNorm2d(128),  # 2维批量标准化层，输入通道数为128
            nn.ReLU(inplace=True),  # ReLU激活函数层，在原位进行操作（inplace=True）
            nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 2维卷积层，输入通道数为128，输出通道数为256，卷积核大小为3x3，步长为2
            nn.BatchNorm2d(256),  # 2维批量标准化层，输入通道数为256
            nn.AvgPool2d((5, 8)),  # 平均池化层，池化核大小为(5, 8)
            # nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，池化核大小为3x3，步长为2
        )

        self.affine_layers = nn.ModuleList()  # 定义一个带有模块列表的模块类，储存全连接层
        last_dim = 256  # 初始化最后一个维度为256
        for nh in hidden_size:  # 对于hidden_size中的每个元素
            self.affine_layers.append(nn.Linear(last_dim, nh))  # 将一个线性变换层添加到affine_layers中，输入维度为last_dim，输出维度为nh
            last_dim = nh  # 更新最后一个维度为nh

        self.logic = nn.Linear(last_dim, 1)  # 定义一个线性变换层，输入维度为last_dim，输出维度为1
        self.logic.weight.data.mul_(0.1)  # 对线性变换层的权重进行缩放
        self.logic.bias.data.mul_(0.0)  # 对线性变换层的偏置进行缩放

    def forward(self, x):  # 定义前向传播函数
        x = self.features(x)  # 将输入通过特征提取层处理
        x = x.view(x.size(0), -1)  # 将x的形状重新变为(batch_size, -1)

        for affine in self.affine_layers:  # 对于affine_layers中的每个元素
            x = self.activation(affine(x))  # 将x输入线性变换层后经过激活函数处理

        prob = torch.sigmoid(self.logic(x))  # 将x输入到逻辑层中，经过sigmoid函数处理
        return prob  # 返回逻辑层的输出

if __name__ == "__main__":
    input = torch.randn((1, 3, 100, 150))
    print(input.size())
    policy = Discriminator(num_inputs=32)
    print(policy(input))
