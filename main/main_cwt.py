from utils.dataset import *
from utils.seed import *
from models.CNN import *
from models.SAC import *
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 对于每个4096值的向量，选择前2048个最小的值
def select_lower_half_values(arr):
    sorted_indices = np.argsort(arr)
    half_length = len(sorted_indices) // 2
    indices_to_keep = sorted_indices[:half_length]
    new_arr = np.zeros_like(arr)
    new_arr[indices_to_keep] = arr[indices_to_keep]
    return new_arr


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


# CNN_CWT + FCNN_CWT
logging.basicConfig(filename='Double_combine_CWT.log', filemode='w', level=logging.INFO, format='%(message)s')


# RNN + LSTM
# logging.basicConfig(filename='Double_combine_second.log', filemode='w', level=logging.INFO, format='%(message)s')
# AdversarialNet
# logging.basicConfig(filename='Double_combine_third.log', filemode='w', level=logging.INFO, format='%(message)s')
# SAC
# logging.basicConfig(filename='Double_combine_fourth.log', filemode='w', level=logging.INFO, format='%(message)s')
# SAC + facol
# logging.basicConfig(filename='Double_combine_fifth.log', filemode='w', level=logging.INFO, format='%(message)s')

# use sigmoid to classification
def to_one_hot(labels, num_classes):
    one_hot_labels = torch.zeros(labels.size(0), num_classes).cuda()
    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1.)
    return one_hot_labels


# Function to evaluate model accuracy
def evaluate_model(loader, model, name):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in loader:
            # input_data = inputs.squeeze(-1)
            input_data = inputs
            if name == 'FCNN_CWT' or name == 'AdversarialNet' or name == 'SAC':
                outputs = model(input_data)
            if name == 'CNN_CWT':
                outputs = model(input_data)
            if name == 'RNN' or name == 'LSTM':
                outputs = model(input_data.unsqueeze(2))
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    # 计算精度、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)
    accuracy = accuracy_score(true_labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(true_labels, predictions)
    }
    return metrics


def All_Unbalance(model, name):
    # Initialize TensorBoard writer

    writer = SummaryWriter(f'runs/{name}_experiment_1')
    # Usage:
    fhg_cwt = FHG()
    (train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_cwt.extract_data()

    # 保存10%的训练数据和标签
    train_data_10, train_labels_fft_new = save_10_percent(train_data_fft, train_labels_fft)

    # 保存10%的测试数据和标签
    test_data_10, test_labels_fft_new = save_10_percent(test_data_fft, test_labels_fft)

    # 读取CSV文件
    train_data_fft = pd.read_csv("../dataset/FH/train_demo.csv").values.reshape(len(train_data_10), 64, 4096)
    test_data_fft = pd.read_csv("../dataset/FH/test_demo.csv").values.reshape(len(test_data_10), 64, 4096)

    train_data_fft = np.apply_along_axis(select_lower_half_values, 2, train_data_fft)
    test_data_fft = np.apply_along_axis(select_lower_half_values, 2, test_data_fft)

    # 1. 数据加载器
    batch_size = 128

    # Splitting train dataset into train and validation sets
    train_dataset_full = TensorDataset(torch.tensor(train_data_fft).float().to(device),
                                       torch.tensor(train_labels_fft_new).long().to(device))
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_data_fft).float().to(device),
                                 torch.tensor(test_labels_fft_new).long().to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. 损失函数和优化器
    model = model  # Move model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1)

    best_acc = 0
    # 3. 训练循环
    num_epochs = 100
    global outputs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        i = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad()
            # test cnncwt close it
            # input_data = inputs.squeeze(-1)
            # open it for cnncwt
            input_data = inputs
            if name == 'FCNN_CWT' or name == 'AdversarialNet' or name == 'SAC':
                outputs = model(input_data)
            if name == 'CNN_CWT':
                outputs = model(input_data)
            if name == 'RNN' or name == 'LSTM':
                outputs = model(input_data.unsqueeze(2))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Record the learning rate
            # current_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + i)

        # Assuming you have already defined a model called 'model'
        # if name == 'FCNN_CWT' or name == 'AdversarialNet' or name == 'SAC':
        #     dummy_input = torch.randn(batch_size, 100, 1000)  # Adjust the size based on your model's expected input
        #     writer.add_graph(model, dummy_input.cuda())
        # if name == 'CNN_CWT':
        #     dummy_input = torch.randn(batch_size, 64, 2048)  # Adjust the size based on your model's expected input
        #     writer.add_graph(model, dummy_input.cuda())
        # if name == 'RNN' or name == 'LSTM':
        #     dummy_input = torch.randn(batch_size, 2048, 1)  # Adjust the size based on your model's expected input
        #     writer.add_graph(model, dummy_input.cuda())

        metrics = evaluate_model(val_loader, model, name)
        scheduler.step(metrics['accuracy'])
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        avg_loss = total_loss / (i + 1)
        writer.add_scalar('Training Loss', avg_loss, epoch)
        writer.add_scalar('Validation Accuracy', metrics['accuracy'], epoch)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / (i + 1)}, Validation Accuracy: {metrics['accuracy']}%")

        # Save the best model
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            torch.save(model.state_dict(), f'outputs/{name}_best_model.pth')

    # Close the writer after training
    writer.close()
    print("Training complete!")

    # 4. Test the model using the best model
    model.load_state_dict(
        torch.load(f'outputs/{name}_best_model.pth', map_location=device))  # Ensure model loads on correct device
    metrics = evaluate_model(test_loader, model, name)
    # 使用logging.info代替print
    logging.info(f"\nClassification Report {name} :\n" + metrics['classification_report'])
    print(f"\nClassification Report {name} :\n", metrics['classification_report'])


if __name__ == '__main__':
    # 0. Set up device and seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seeds(seed_value=2023)

    # 0D 1D 两两组合
    prefixes = ["0D", "1D", "2D", "3D", "4D"]

    #  ALL 5 num_classes CNN_CWT
    All_Unbalance(model=CNN_CWT().to(device), name='CNN_CWT')

    # # CNN_CWT
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E', model=CNN(num_classes=2).to(device), name='CNN_CWT')
    #
    # # ALL 5 num_classes FCNN_CWT
    # All_Unbalance(model=FCNN().to(device), name='FCNN_CWT')
    #
    # # FCNN_CWT
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E', model=FCNN(num_classes=2).to(device), name='FCNN_CWT')

    # RNN
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E', model=RNN(num_classes=2).to(device), name='RNN')

    # LSTM
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E', model=LSTM(num_classes=2).to(device), name='LSTM')

    # AdversarialNet
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E', model=AdversarialNet(num_classes=2).to(device),
    #                      name='AdversarialNet')

    # # ALL 5 num_classes SAC
    # All_Unbalance(
    #     model=SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64], classifier_hidden_size=128).to(device),
    #     name='SAC')
    #
    # # SAC
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E',
    #                      model=SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64],
    #                                classifier_hidden_size=128, num_classes=2).to(device),
    #                      name='SAC')

    # # ALL 5 num_classes SAC
    # All_Unbalance(
    #     model=SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64], classifier_hidden_size=128).to(device),
    #     name='SAC')
    #
    # # SAC
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E',
    #                      model=SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64],
    #                                classifier_hidden_size=128, num_classes=2).to(device),
    #                      name='SAC')
