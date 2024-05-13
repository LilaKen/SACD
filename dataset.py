import pandas as pd
import numpy as np
from scipy.special import gamma as gamma_fn, factorial
from tqdm import tqdm


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


class Normalize:
    def __call__(self, data):
        # epsilon = 1e-10
        return (data - np.mean(data)) / np.std(data)


class Reshape:
    def __call__(self, data):
        return data.reshape(-1, 1)


class Retype:
    def __call__(self, data):
        return data.astype(np.float32)


class FFTTransform():
    def __call__(self, chunk):
        # 进行FFT
        fft_result = np.fft.fft(chunk)
        # 只保留前2048个系数
        meaningful_coefficients = fft_result[:2048]
        # 这里只保留了幅度，如果你还需要相位，可以另外处理
        return np.abs(meaningful_coefficients)


class FHG(object):
    def __init__(self, num_classes=5, train_suffix="D", test_suffix="E", file_extension=".csv"):
        self.train_files = [f"{i}{train_suffix}" for i in range(num_classes)]
        self.test_files = [f"{i}{test_suffix}" for i in range(num_classes)]
        self.file_extension = file_extension
        self.chunk_size = 4096

        self.data_transforms = {
            'train': [Reshape(), Normalize(), Retype()],
            'val': [Reshape(), Normalize(), Retype()]
        }

    def _read_and_chunk_file(self, file_prefix, transforms):
        data_chunks = []
        file_name = file_prefix + self.file_extension
        data = pd.read_csv(file_name)["Vibration_1"].values
        num_chunks = len(data) // self.chunk_size
        for _ in range(num_chunks):
            chunk = data[self.chunk_size * _: self.chunk_size * (_ + 1)]
            for transform in transforms:
                chunk = transform(chunk)
            data_chunks.append(chunk)
        return data_chunks

    def extract_data(self):
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for idx, file in enumerate(tqdm(self.train_files, desc="Processing Training Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['train'])
            train_data.extend(chunks)
            train_labels.extend([idx] * len(chunks))

        for idx, file in enumerate(tqdm(self.test_files, desc="Processing Testing Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['val'])
            test_data.extend(chunks)
            test_labels.extend([idx] * len(chunks))

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        return (train_data, train_labels), (test_data, test_labels)


class FHGFFT(object):
    def __init__(self, num_classes=5, train_suffix="D", test_suffix="E", file_extension=".csv"):
        self.train_files = [f"{i}{train_suffix}" for i in range(num_classes)]
        self.test_files = [f"{i}{test_suffix}" for i in range(num_classes)]
        self.file_extension = file_extension
        self.chunk_size = 4096

        self.data_transforms = {
            'train': [FFTTransform(), Reshape(), Normalize(), Retype()],
            'val': [FFTTransform(), Reshape(), Normalize(), Retype()]
        }

    def _read_and_chunk_file(self, file_prefix, transforms):
        data_chunks = []
        file_name = file_prefix + self.file_extension
        data = pd.read_csv(file_name)["Vibration_1"].values
        num_chunks = len(data) // self.chunk_size
        for _ in range(num_chunks):
            chunk = data[self.chunk_size * _: self.chunk_size * (_ + 1)]
            for transform in transforms:
                chunk = transform(chunk)
            data_chunks.append(chunk)
        return data_chunks

    def extract_data(self):
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for idx, file in enumerate(tqdm(self.train_files, desc="Processing Training Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['train'])
            train_data.extend(chunks)
            train_labels.extend([idx] * len(chunks))

        for idx, file in enumerate(tqdm(self.test_files, desc="Processing Testing Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['val'])
            test_data.extend(chunks)
            test_labels.extend([idx] * len(chunks))

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        return (train_data, train_labels), (test_data, test_labels)


class FHGFFTDOU(object):
    def __init__(self, num_classes=2, train_suffix="D", test_suffix="E", second_train_suffix="D",
                 second_test_suffix="E", file_extension=".csv"):
        self.train_files = [train_suffix, second_train_suffix]
        self.test_files = [test_suffix, second_test_suffix]
        self.file_extension = file_extension
        self.chunk_size = 4096

        self.data_transforms = {
            'train': [FFTTransform(), Reshape(), Normalize(), Retype()],
            'val': [FFTTransform(), Reshape(), Normalize(), Retype()]
        }

    def _read_and_chunk_file(self, file_prefix, transforms):
        data_chunks = []
        file_name = file_prefix + self.file_extension
        data = pd.read_csv(file_name)["Vibration_1"].values
        num_chunks = len(data) // self.chunk_size
        for _ in range(num_chunks):
            chunk = data[self.chunk_size * _: self.chunk_size * (_ + 1)]
            for transform in transforms:
                chunk = transform(chunk)
            data_chunks.append(chunk)
        return data_chunks

    def extract_data(self):
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for idx, file in enumerate(tqdm(self.train_files, desc="Processing Training Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['train'])
            train_data.extend(chunks)
            train_labels.extend([idx] * len(chunks))

        for idx, file in enumerate(tqdm(self.test_files, desc="Processing Testing Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['val'])
            test_data.extend(chunks)
            test_labels.extend([idx] * len(chunks))

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        return (train_data, train_labels), (test_data, test_labels)

class FHGDOU(object):
    def __init__(self, num_classes=2, train_suffix="D", test_suffix="E", second_train_suffix="D",
                 second_test_suffix="E", file_extension=".csv"):
        self.train_files = [train_suffix, second_train_suffix]
        self.test_files = [test_suffix, second_test_suffix]
        self.file_extension = file_extension
        self.chunk_size = 4096

        self.data_transforms = {
            'train': [Reshape(), Normalize(), Retype()],
            'val': [Reshape(), Normalize(), Retype()]
        }

    def _read_and_chunk_file(self, file_prefix, transforms):
        data_chunks = []
        file_name = file_prefix + self.file_extension
        data = pd.read_csv(file_name)["Vibration_1"].values
        num_chunks = len(data) // self.chunk_size
        for _ in range(num_chunks):
            chunk = data[self.chunk_size * _: self.chunk_size * (_ + 1)]
            for transform in transforms:
                chunk = transform(chunk)
            data_chunks.append(chunk)
        return data_chunks

    def extract_data(self):
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for idx, file in enumerate(tqdm(self.train_files, desc="Processing Training Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['train'])
            train_data.extend(chunks)
            train_labels.extend([idx] * len(chunks))

        for idx, file in enumerate(tqdm(self.test_files, desc="Processing Testing Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['val'])
            test_data.extend(chunks)
            test_labels.extend([idx] * len(chunks))

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        return (train_data, train_labels), (test_data, test_labels)


class FHGCWT(object):
    def __init__(self, num_classes=5, train_suffix="D", test_suffix="E", file_extension=".csv",
                 widths=np.arange(301, 401), k=3, beta=8.5, gamma=3):
        self.train_files = [f"{i}{train_suffix}" for i in range(num_classes)]
        self.test_files = [f"{i}{test_suffix}" for i in range(num_classes)]
        self.file_extension = file_extension
        self.chunk_size = 4096

        self.data_transforms = {
            'train': [CWTTransform(widths, k, beta, gamma), Reshape(), Retype()],
            'val': [CWTTransform(widths, k, beta, gamma), Reshape(), Retype()]
        }

    def _read_and_chunk_file(self, file_prefix, transforms):
        data_chunks = []
        file_name = file_prefix + self.file_extension
        data = pd.read_csv(file_name)["Vibration_1"].values
        num_chunks = len(data) // self.chunk_size
        for _ in range(num_chunks):
            chunk = data[self.chunk_size * _: self.chunk_size * (_ + 1)]
            for transform in transforms:
                chunk = transform(chunk)
            data_chunks.append(chunk)
        return data_chunks

    def extract_data(self):
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

        for idx, file in enumerate(tqdm(self.train_files, desc="Processing Training Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['train'])
            train_data.extend(chunks)
            train_labels.extend([idx] * len(chunks))

        for idx, file in enumerate(tqdm(self.test_files, desc="Processing Testing Files")):
            chunks = self._read_and_chunk_file('../dataset/FH/' + file, self.data_transforms['val'])
            test_data.extend(chunks)
            test_labels.extend([idx] * len(chunks))

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        return (train_data, train_labels), (test_data, test_labels)


# Usage:
# fhg = FHG()
# (train_data, train_labels), (test_data, test_labels) = fhg.extract_data()


