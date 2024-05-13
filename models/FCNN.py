import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
import torch


class FCNN(nn.Module):
    def __init__(self, input_size=2048, fc_size=1024, num_classes=5, hidden_times=1):
        super(FCNN, self).__init__()

        # 定义FC层
        self.fc1 = nn.Linear(input_size, fc_size)
        self.fc_output = nn.Linear(fc_size, num_classes)  # 注意这里改为num_classes

        # RobustScaler初始化
        self.scaler = RobustScaler(quantile_range=(5, 95))

        # 定义隐藏次数
        self.hidden_times = hidden_times

    def forward(self, x):
        # 使用RobustScaler进行缩放
        x_np = x.detach().cpu().numpy()
        x_scaled = self.scaler.fit_transform(x_np)
        x = torch.tensor(x_scaled).float().cuda()

        # 通过FC层
        for _ in range(self.hidden_times):
            x = self.fc1(x)
            x = F.leaky_relu(x)

        x = self.fc_output(x)


        return x
