import pandas as pd
import numpy as np
from scipy.special import gamma as gamma_fn, factorial
from tqdm import tqdm


def save_10_percent(data, labels):
    # 计算10%的数量
    n = int(len(data) * 0.10)
    np.random.seed(2023)
    # 随机选择10%的索引
    indices = np.random.choice(len(data), n, replace=False)

    # 使用选择的索引获取数据和标签
    subset_data = data[indices]
    subset_labels = labels[indices]

    return subset_data, subset_labels


class FHG(object):
    def __init__(self, num_classes=5, train_suffix="D", test_suffix="E", file_extension=".csv"):
        self.train_files = [f"{i}{train_suffix}" for i in range(num_classes)]
        self.test_files = [f"{i}{test_suffix}" for i in range(num_classes)]
        self.file_extension = file_extension
        self.chunk_size = 4096

    def _read_and_chunk_file(self, file_prefix):
        data_chunks = []
        file_name = file_prefix + self.file_extension
        data = pd.read_csv(file_name)["Vibration_1"].values
        num_chunks = len(data) // self.chunk_size
        for _ in range(num_chunks):
            chunk = data[self.chunk_size * _: self.chunk_size * (_ + 1)]
            data_chunks.append(chunk)
        return data_chunks

    def extract_data(self):
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for idx, file in enumerate(tqdm(self.train_files, desc="Processing Training Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file)
            train_data.extend(chunks)
            train_labels.extend([idx] * len(chunks))

        for idx, file in enumerate(tqdm(self.test_files, desc="Processing Testing Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file)
            test_data.extend(chunks)
            test_labels.extend([idx] * len(chunks))

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        return (train_data, train_labels), (test_data, test_labels)


# 1. Generalized Laguerre polynomial
def generalized_laguerre(n, c, x):
    poly = 0
    for m in range(n + 1):
        poly += ((-1) ** m * factorial(n + c) * x ** m) / (factorial(m) * factorial(c + m + 1) * factorial(c + m + 1))
    return poly


# 2. Generalized Morse wavelet
def morse_wavelet(t, k, beta, gamma):
    omega = 2.0 * np.pi * t
    r = 2 * beta + 1 / gamma
    A = np.sqrt(gamma * np.exp(1) / (2 ** beta * gamma_fn(gamma * k + 1) * r))  # use gamma_fn here
    wavelet = A * np.sqrt(2) * (omega ** gamma) * np.exp(-omega) * generalized_laguerre(k, r - 1, 2 * omega)
    return wavelet


# 3. Continuous Wavelet Transform (simplified for demonstration)
def cwt(data, wavelet_function, widths, k, beta, gamma):
    output = np.zeros([len(widths), len(data)])
    for ind, width in enumerate(widths):
        wavelet_data = wavelet_function(data / width, k, beta, gamma)
        output[ind, :] = np.convolve(data, wavelet_data, mode='same')
    return output


class CWTTransform:
    """Apply Continuous Wavelet Transform."""

    def __init__(self, widths, k, beta, gamma):
        self.widths = widths
        self.k = k
        self.beta = beta
        self.gamma = gamma

    def __call__(self, chunk):
        return cwt(chunk, morse_wavelet, self.widths, self.k, self.beta, self.gamma)


fhg = FHG()
(train_data, train_labels), (test_data, test_labels) = fhg.extract_data()

# 保存10%的训练数据和标签
train_data_10, train_labels_10 = save_10_percent(train_data, train_labels)

# 保存10%的测试数据和标签
test_data_10, test_labels_10 = save_10_percent(test_data, test_labels)

# 设定widths参数，这里假设从1到64，可以根据实际需求进行修改
widths = np.arange(1, 65)

# 初始化一个空的数组来存储CWT的结果
train_transformed_data = np.zeros((len(train_data_10), len(widths) * 4096))

# 对每个样本进行CWT变换
for i in tqdm(range(train_data_10.shape[0])):
    sample = train_data_10[i, :]
    cwt_result = cwt(sample, morse_wavelet, widths, k=2, beta=3.0, gamma=4.0)  # 假设使用之前定义的morse_wavelet
    train_transformed_data[i] = cwt_result.flatten()

# 保存结果为train.csv
df_train = pd.DataFrame(train_transformed_data)
df_train.to_csv("../dataset/FH/train_demo.csv", index=False)

# 初始化一个空的数组来存储CWT的结果
test_transformed_data = np.zeros((len(test_data_10), len(widths) * 4096))

# 对每个样本进行CWT变换
for i in tqdm(range(test_data_10.shape[0])):
    sample = test_data_10[i, :]
    cwt_result = cwt(sample, morse_wavelet, widths, k=2, beta=3.0, gamma=4.0)  # 假设使用之前定义的morse_wavelet
    test_transformed_data[i] = cwt_result.flatten()

# 保存结果为test.csv
df_test = pd.DataFrame(test_transformed_data)
df_test.to_csv("../dataset/FH/test_demo.csv", index=False)
