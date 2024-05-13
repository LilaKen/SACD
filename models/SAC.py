import torch.nn as nn
import torch
import numpy as np


class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0, initial_estimate=1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = initial_estimate

    def update(self, measurement):
        # 预测
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # 更新
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate


# 假设的卡尔曼滤波参数
# 64 1E-2
process_variance = 1e-2
measurement_variance = 0.1 ** 2

kf = KalmanFilter(process_variance, measurement_variance)


# 对数据应用卡尔曼滤波
def apply_kalman(data):
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered_data[i, j] = kf.update(data[i, j])
    return filtered_data

# 之前用的对比学习，效果不好 acc 59%
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, true_labels):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.temperature

        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        positive_samples = torch.cat([sim_ij, sim_ji], dim=0).reshape(2 * batch_size, 1)
        negative_samples = sim - torch.eye(2 * batch_size, device=sim.device) * 1e5

        labels = torch.arange(2 * batch_size).to(z_i.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


# 对比正样本方法一
def random_jitter(signal, std=0.01):
    noise = torch.randn_like(signal) * std
    return signal + noise

# 对比正样本方法二
def time_warp(data, alpha=1.0):
    """ 时间拉伸或缩放 """
    n = len(data)
    alpha = int(alpha)  # 确保alpha是整数
    warp_amount = int(torch.randint(-alpha, alpha + 1, (1,)).item())  # 需要+1来确保上界是包括的
    src_index = torch.arange(n)
    dest_index = torch.clamp(src_index + warp_amount, 0, n - 1)
    warped_data = data[dest_index]
    return warped_data


def random_slice(signal, slice_length=1000):
    start_idx = np.random.randint(0, signal.size(0) - slice_length)
    return signal[start_idx:start_idx + slice_length]


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# 2. 堆叠自编码器
class StackedAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(StackedAutoencoder, self).__init__()
        self.ae_layers = nn.ModuleList()
        last_size = input_size
        for hidden_size in hidden_sizes:
            self.ae_layers.append(Autoencoder(last_size, hidden_size))
            last_size = hidden_size

    def forward(self, x):
        for ae in self.ae_layers:
            x, _ = ae(x)
        return x


class BLSLayer(nn.Module):
    def __init__(self, input_dim, num_enhance_nodes=500):
        super(BLSLayer, self).__init__()
        self.input_dim = input_dim
        self.num_enhance_nodes = num_enhance_nodes

        # 初始化增强节点的权重和偏置
        self.weights = nn.Parameter(torch.randn(input_dim, num_enhance_nodes), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(num_enhance_nodes), requires_grad=False)

        self.beta = None

    def forward(self, x):
        # 计算增强节点的输出
        V = x @ self.weights + self.bias
        H_enhanced = torch.sigmoid(V)

        # 将增强节点的输出与原始数据并起来
        H = torch.cat([x, H_enhanced], dim=1)

        # 如果beta已经计算，使用它进行预测
        if self.beta is not None:
            H = H @ self.beta
        return H

    def compute_beta(self, H, Y):
        # 使用正规方程计算beta，这部分使用NumPy
        H_np = H.detach().numpy()
        Y_np = Y.detach().numpy()
        pseudo_inverse = np.linalg.pinv(H_np)
        self.beta = torch.tensor(np.matmul(pseudo_inverse, Y_np), dtype=torch.float32)
        self.beta = nn.Parameter(self.beta, requires_grad=False)


# 3. 堆叠自编码器 + 分类器
class SAC(nn.Module):
    def __init__(self, input_size, sae_hidden_sizes, classifier_hidden_size=128, num_enhance_nodes=500, num_classes=5):
        super(SAC, self).__init__()

        self.sae = StackedAutoencoder(input_size, sae_hidden_sizes)

        # 建立分类器
        self.classifier = nn.Sequential(
            nn.Linear(sae_hidden_sizes[-1], classifier_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(classifier_hidden_size, num_classes),
        )
        # 使用BLSLayer代替原始的分类器
        # self.classifier = BLSLayer(sae_hidden_sizes[-1], num_enhance_nodes)

    def forward(self, x):
        x = self.sae(x)
        x = self.classifier(x)
        return x

# 使用例子:
# model = SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64], classifier_hidden_size=128, num_classes=5)
