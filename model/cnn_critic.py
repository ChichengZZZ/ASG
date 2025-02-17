import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, ), activation='tanh'):
        super().__init__()

        # 根据激活函数名称给self.activation赋值
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # 定义神经网络的特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),  # 输入通道数为3，输出通道数为64，卷积核大小为3，步长为2
            nn.BatchNorm2d(64),  # 批量归一化层，64个通道归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，池化核大小为3，步长为2
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # 输入通道数为64，输出通道数为128，卷积核大小为3，步长为2
            nn.BatchNorm2d(128),  # 批量归一化层，128个通道归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 输入通道数为128，输出通道数为256，卷积核大小为3，步长为2
            nn.BatchNorm2d(256),  # 批量归一化层，256个通道归一化
            nn.AvgPool2d((5, 8)),  # 平均池化层，池化核大小为(5, 8)
            # nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，池化核大小为3，步长为2
        )

        # 定义全连接层
        self.affine_layers = nn.ModuleList()
        last_dim = 256
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))  # 添加一个全连接层，输入维度为last_dim，输出维度为nh
            last_dim = nh

        # 定义输出层
        self.value_head = nn.Linear(last_dim, 1)  # 全连接层，输入维度为last_dim，输出维度为1
        self.value_head.weight.data.mul_(0.1)  # 权重初始化为0.1
        self.value_head.bias.data.mul_(0.0)  # 偏置初始化为0.0

    def forward(self, x):
        x = self.features(x)  # 通过特征提取部分处理输入
        x = x.view(x.size(0), -1)  # 将特征展平成一维向量

        for affine in self.affine_layers:  # 通过全连接层进行处理
            x = self.activation(affine(x))  # 使用激活函数进行激活

        value = self.value_head(x)  # 输出层输出值
        return value

if __name__ == "__main__":
    input = torch.randn((1, 3, 100, 150))  # 输入数据，维度为(1, 3, 100, 150)
    print(input.size())
    policy = Value(state_dim=32)  # 创建Value对象
    print(policy(input))  # 打印模型的输出值