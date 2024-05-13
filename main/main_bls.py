from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from models.BLS import *
from utils.dataset import *

# 数据提取
fhg_fft = FHGFFT()
(train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_fft.extract_data()

# 假设你的train_labels_fft和test_labels_fft是整数类标签，我们需要将其转换为one-hot编码
encoder = OneHotEncoder(sparse=False, categories='auto')
train_labels_onehot = encoder.fit_transform(train_labels_fft.reshape(-1, 1))
test_labels_onehot = encoder.transform(test_labels_fft.reshape(-1, 1))

# 初始化BLS模型
input_dim = train_data_fft.shape[1]  # 获取输入数据的维度
bls = BLS_MultiClass(input_dim=2048, num_enhance_nodes=500, num_classes=5)

# 使用训练数据进行训练
bls.fit(train_data_fft.squeeze(-1), train_labels_onehot)

# 使用模型进行预测
predicted_labels_onehot = bls.predict(test_data_fft.squeeze(-1))
predicted_labels = np.argmax(predicted_labels_onehot, axis=1)

# 打印分类报告
report = classification_report(test_labels_fft, predicted_labels)
print("Classification Report BLS :")
print(report)
