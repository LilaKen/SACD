from utils.dataset import *
from utils.seed import *
from models.CNN import *
from models.FCNN import *
from models.RNN import *
from models.LSTM import *
from models.QCNN import *
from models.WDCNN import *
from models.SAC import *
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import logging

from torch.utils.data import TensorDataset, DataLoader, random_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CNN + FCNN
logging.basicConfig(filename='Double_combine.log', filemode='w', level=logging.INFO, format='%(message)s')
# RNN + LSTM
logging.basicConfig(filename='Double_combine_second.log', filemode='w', level=logging.INFO, format='%(message)s')
# SAC
logging.basicConfig(filename='Double_combine_fourth.log', filemode='w', level=logging.INFO, format='%(message)s')
# SAC + facol
logging.basicConfig(filename='Double_combine_fifth.log', filemode='w', level=logging.INFO, format='%(message)s')
# QCNN
logging.basicConfig(filename='Double_combine_qcnn.log', filemode='w', level=logging.INFO, format='%(message)s')
# WDCNN
logging.basicConfig(filename='Double_combine_wdcnn.log', filemode='w', level=logging.INFO, format='%(message)s')
# CWT
logging.basicConfig(filename='Double_combine_cnn_cwt.log', filemode='w', level=logging.INFO, format='%(message)s')
# SACD
logging.basicConfig(filename='Double_combine_SACD.log', filemode='w', level=logging.INFO, format='%(message)s')
# SACD 10%
logging.basicConfig(filename='Double_combine_SACD10.log', filemode='w', level=logging.INFO, format='%(message)s')
# SACD
logging.basicConfig(filename='Double_combine_SACD_FHG.log', filemode='w', level=logging.INFO, format='%(message)s')
# SACD
logging.basicConfig(filename='Double_combine_SACD_Hyper.log', filemode='w', level=logging.INFO, format='%(message)s')



class DifferenceTensorDataset(TensorDataset):
    def __init__(self, original_data, difference_data, labels):
        assert original_data.shape == difference_data.shape
        super().__init__(original_data, difference_data, labels)


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
        # for original_inputs, difference_inputs, labels in loader:
        for (inputs, labels) in loader:
            # original_input_data = original_inputs.squeeze(-1)
            # difference_input_data = difference_inputs.squeeze(-1)

            if name == 'FCNN' or name == 'AdversarialNet' or name == 'SACD':
                # 获取原始数据的编码表示
                # original_encoded = model.sae(original_input_data)
                #
                # # 获取滤波后的数据差异的编码表示
                # difference_encoded = model.sae(difference_input_data)
                #
                # # 将两部分的编码相加
                # combined_encoded = original_encoded + difference_encoded

                # # 将加和后的编码数据传递给分类器进行预测
                # outputs = model.classifier(combined_encoded)
                outputs = model(inputs.squeeze(-1))
            # if name == 'RNN' or name == 'LSTM':
            #     outputs = model(original_inputs.unsqueeze(2))
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
    fhg_fft = FHGFFT()
    (train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_fft.extract_data()


    # 1. 数据加载器
    batch_size = 128

    # Splitting train dataset into train and validation sets
    train_dataset_full = TensorDataset(torch.tensor(train_data_fft).float().to(device),
                                       torch.tensor(train_labels_fft).long().to(device))
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_data_fft).float().to(device),
                                 torch.tensor(test_labels_fft).long().to(device))

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
        # for i, (original_inputs, difference_inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad()
            # 对比学习数据增强
            # inputs_aug1 = random_jitter(inputs.squeeze(-1))
            # inputs_aug2 = time_warp(inputs.squeeze(-1))

            # 根据模型名称处理输入数据
            if name in ['FCNN', 'AdversarialNet', 'SACD', 'StackedAutoencoder']:  # 如果你的对比学习模型名是其他的，请修改这里
                # z_i = model(inputs_aug1)
                # z_j = model(inputs_aug2)
                # # 初始化NTXentLoss对象
                # ntxent_criterion = NTXentLoss()
                # contra_labels = torch.cat([labels, labels], dim=0)
                #
                # # 在训练循环中计算损失
                # contrastive_loss = ntxent_criterion(z_i, z_j, contra_labels)
                #
                # # 分类输出和损失
                # class_outputs = model(inputs.squeeze(-1))
                # classification_loss = criterion(class_outputs, labels)
                #
                # # 这里你可能需要权衡两者的损失
                # loss = contrastive_loss + classification_loss
                input_data = inputs.squeeze(-1)
                outputs = model(input_data)
                # 获取原始数据的编码表示
                # original_encoded = model.sae(original_inputs.squeeze(-1))
                #
                # # 获取滤波后的数据差异的编码表示
                # difference_encoded = model.sae(difference_inputs.squeeze(-1))
                #
                # # 将两部分的编码相加
                # combined_encoded = original_encoded + difference_encoded
                #
                # # 将加和后的编码数据传递给分类器进行预测
                # outputs = model.classifier(combined_encoded)

                # loss = criterion(outputs, labels)

            # elif name in ['CNN', 'QCNN', 'WDCNN']:
            #     outputs = model(original_inputs.unsqueeze(1))
            #     loss = criterion(outputs, labels)
            # elif name in ['RNN', 'LSTM']:
            #     outputs = model(original_inputs.unsqueeze(2))
            #     loss = criterion(outputs, labels)
            else:
                raise ValueError(f"Model name {name} not recognized.")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Record the learning rate
            # current_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + i)

        # Assuming you have already defined a model called 'model'
        # if name == 'FCNN' or name == 'AdversarialNet' or name == 'SACD':
        #     dummy_input = torch.randn(batch_size, 2048)  # Adjust the size based on your model's expected input
        #     writer.add_graph(model, dummy_input.cuda())
        # if name == 'CNN' or name == 'QCNN' or name == 'WDCNN':
        #     dummy_input = torch.randn(batch_size, 1, 2048)  # Adjust the size based on your model's expected input
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

def All_Unbalance_D(model, name):
    # Initialize TensorBoard writer

    writer = SummaryWriter(f'runs/{name}_experiment_1')
    # Usage:
    fhg_fft = FHGFFT()
    (train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_fft.extract_data()

    train_data_filtered = apply_kalman(train_data_fft)
    test_data_filtered = apply_kalman(test_data_fft)

    train_data_difference = train_data_fft - train_data_filtered
    test_data_difference = test_data_fft - test_data_filtered

    # 在这里存储train_data_difference和test_data_difference，如果需要

    # 1. 数据加载器
    batch_size = 128

    # Splitting train dataset into train and validation sets
    # train_dataset_full = TensorDataset(torch.tensor(train_data_fft).float().to(device),
    #                                    torch.tensor(train_labels_fft).long().to(device))
    # train_size = int(0.9 * len(train_dataset_full))
    # val_size = len(train_dataset_full) - train_size
    # train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #
    # test_dataset = TensorDataset(torch.tensor(test_data_fft).float().to(device),
    #                              torch.tensor(test_labels_fft).long().to(device))
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_data_fft_tensor = torch.tensor(train_data_fft).float().to(device)
    train_data_difference_tensor = torch.tensor(train_data_difference).float().to(device)
    train_labels_fft_tensor = torch.tensor(train_labels_fft).long().to(device)

    test_data_fft_tensor = torch.tensor(test_data_fft).float().to(device)
    test_data_difference_tensor = torch.tensor(test_data_difference).float().to(device)
    test_labels_fft_tensor = torch.tensor(test_labels_fft).long().to(device)

    # 使用DifferenceTensorDataset
    train_dataset_full = DifferenceTensorDataset(train_data_fft_tensor, train_data_difference_tensor,
                                                 train_labels_fft_tensor)

    # 与您之前的代码相同，随机划分训练集和验证集
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 测试数据集和加载器
    test_dataset = DifferenceTensorDataset(test_data_fft_tensor, test_data_difference_tensor, test_labels_fft_tensor)
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
        # for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        for i, (original_inputs, difference_inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad()
            # 对比学习数据增强
            # inputs_aug1 = random_jitter(inputs.squeeze(-1))
            # inputs_aug2 = time_warp(inputs.squeeze(-1))

            # 根据模型名称处理输入数据
            if name in ['FCNN', 'AdversarialNet', 'SACD', 'StackedAutoencoder']:  # 如果你的对比学习模型名是其他的，请修改这里
                # z_i = model(inputs_aug1)
                # z_j = model(inputs_aug2)
                # # 初始化NTXentLoss对象
                # ntxent_criterion = NTXentLoss()
                # contra_labels = torch.cat([labels, labels], dim=0)
                #
                # # 在训练循环中计算损失
                # contrastive_loss = ntxent_criterion(z_i, z_j, contra_labels)
                #
                # # 分类输出和损失
                # class_outputs = model(inputs.squeeze(-1))
                # classification_loss = criterion(class_outputs, labels)
                #
                # # 这里你可能需要权衡两者的损失
                # loss = contrastive_loss + classification_loss
                # input_data = inputs.squeeze(-1)
                # outputs = model(input_data)
                # 获取原始数据的编码表示
                original_encoded = model.sae(original_inputs.squeeze(-1))

                # 获取滤波后的数据差异的编码表示
                difference_encoded = model.sae(difference_inputs.squeeze(-1))

                # 将两部分的编码相加
                combined_encoded = original_encoded + difference_encoded

                # 将加和后的编码数据传递给分类器进行预测
                outputs = model.classifier(combined_encoded)

                loss = criterion(outputs, labels)

            elif name in ['CNN', 'QCNN', 'WDCNN']:
                outputs = model(original_inputs.unsqueeze(1))
                loss = criterion(outputs, labels)
            elif name in ['RNN', 'LSTM']:
                outputs = model(original_inputs.unsqueeze(2))
                loss = criterion(outputs, labels)
            else:
                raise ValueError(f"Model name {name} not recognized.")
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Record the learning rate
            # current_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + i)

        # Assuming you have already defined a model called 'model'
        # if name == 'FCNN' or name == 'AdversarialNet' or name == 'SACD':
        #     dummy_input = torch.randn(batch_size, 2048)  # Adjust the size based on your model's expected input
        #     writer.add_graph(model, dummy_input.cuda())
        # if name == 'CNN' or name == 'QCNN' or name == 'WDCNN':
        #     dummy_input = torch.randn(batch_size, 1, 2048)  # Adjust the size based on your model's expected input
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


def Enum_Esecnum_D(train_suffix, test_suffix, second_train_suffix, second_test_suffix, model, name):
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/{name}_experiment_' + str(test_suffix) + '_' + str(second_test_suffix))
    # Usage:
    fhg_fft = FHGFFTDOU(train_suffix=train_suffix, test_suffix=test_suffix, second_train_suffix=second_train_suffix,
                        second_test_suffix=second_test_suffix)
    (train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_fft.extract_data()

    train_data_filtered = apply_kalman(train_data_fft)
    test_data_filtered = apply_kalman(test_data_fft)

    train_data_difference = train_data_fft - train_data_filtered
    test_data_difference = test_data_fft - test_data_filtered


    # 1. 数据加载器
    batch_size = 128

    train_data_fft_tensor = torch.tensor(train_data_fft).float().to(device)
    train_data_difference_tensor = torch.tensor(train_data_difference).float().to(device)
    train_labels_fft_tensor = torch.tensor(train_labels_fft).long().to(device)

    test_data_fft_tensor = torch.tensor(test_data_fft).float().to(device)
    test_data_difference_tensor = torch.tensor(test_data_difference).float().to(device)
    test_labels_fft_tensor = torch.tensor(test_labels_fft).long().to(device)

    # 使用DifferenceTensorDataset
    train_dataset_full = DifferenceTensorDataset(train_data_fft_tensor, train_data_difference_tensor,
                                                 train_labels_fft_tensor)

    # 与您之前的代码相同，随机划分训练集和验证集
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 测试数据集和加载器
    test_dataset = DifferenceTensorDataset(test_data_fft_tensor, test_data_difference_tensor, test_labels_fft_tensor)
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
        for i, (original_inputs, difference_inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad()
            if name == 'FCNN' or name == 'AdversarialNet' or name == 'SACD':
                original_encoded = model.sae(original_inputs.squeeze(-1))
                # 获取滤波后的数据差异的编码表示
                difference_encoded = model.sae(difference_inputs.squeeze(-1))

                # 将两部分的编码相加
                combined_encoded = original_encoded + difference_encoded

                # 将加和后的编码数据传递给分类器进行预测
                outputs = model.classifier(combined_encoded)

                # loss = criterion(outputs, labels)

            # if name == 'CNN' or name == 'QCNN' or name == 'WDCNN':
            #         outputs = model(input_data.unsqueeze(1))
            # if name == 'RNN' or name == 'LSTM':
            #         outputs = model(input_data.unsqueeze(2))
            # use sigmoid function
            # one_hot_labels = to_one_hot(labels, 2)
            # loss = criterion(outputs, one_hot_labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # # Assuming you have already defined a model called 'model'
        # dummy_input = torch.randn(batch_size, 1, 4096)  # Adjust the size based on your model's expected input
        # writer.add_graph(model, dummy_input.cuda())

        # 在您的训练循环或模型评估部分，调用evaluate_model
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
            torch.save(model.state_dict(), f"outputs/{test_suffix}_{second_test_suffix}_{name}_best_model.pth")

    # Close the writer after training
    writer.close()
    print("Training complete!")

    # 4. Test the model using the best model
    model.load_state_dict(
        torch.load(f"outputs/{test_suffix}_{second_test_suffix}_{name}_best_model.pth",
                   map_location=device))  # Ensure model loads on correct device
    metrics = evaluate_model(test_loader, model, name)
    # 使用logging.info代替print
    logging.info(f"\nClassification Report {name} :\n" + metrics['classification_report'])
    print(f"\nClassification Report {name} :\n", metrics['classification_report'])


def Enum_Esecnum(train_suffix, test_suffix, second_train_suffix, second_test_suffix, model, name):
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/{name}_experiment_' + str(test_suffix) + '_' + str(second_test_suffix))
    # Usage:
    fhg_fft = FHGDOU(train_suffix=train_suffix, test_suffix=test_suffix, second_train_suffix=second_train_suffix,
                        second_test_suffix=second_test_suffix)
    (train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_fft.extract_data()

    # 1. 数据加载器
    batch_size = 128

    # Splitting train dataset into train and validation sets
    train_dataset_full = TensorDataset(torch.tensor(train_data_fft).float().to(device),
                                       torch.tensor(train_labels_fft).long().to(device))
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_data_fft).float().to(device),
                                 torch.tensor(test_labels_fft).long().to(device))
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
            input_data = inputs.squeeze(-1)
            if name == 'FCNN' or name == 'AdversarialNet' or name == 'SACD':
                outputs = model(input_data)
            if name == 'CNN' or name == 'QCNN' or name == 'WDCNN':
                outputs = model(input_data.unsqueeze(1))
            if name == 'RNN' or name == 'LSTM':
                outputs = model(input_data.unsqueeze(2))
            # use sigmoid function
            # one_hot_labels = to_one_hot(labels, 2)
            # loss = criterion(outputs, one_hot_labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # # Assuming you have already defined a model called 'model'
        # dummy_input = torch.randn(batch_size, 1, 4096)  # Adjust the size based on your model's expected input
        # writer.add_graph(model, dummy_input.cuda())

        # 在您的训练循环或模型评估部分，调用evaluate_model
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
            torch.save(model.state_dict(), f"outputs/{test_suffix}_{second_test_suffix}_{name}_best_model.pth")

    # Close the writer after training
    writer.close()
    print("Training complete!")

    # 4. Test the model using the best model
    model.load_state_dict(
        torch.load(f"outputs/{test_suffix}_{second_test_suffix}_{name}_best_model.pth",
                   map_location=device))  # Ensure model loads on correct device
    metrics = evaluate_model(test_loader, model, name)
    # 使用logging.info代替print
    logging.info(f"\nClassification Report {name} :\n" + metrics['classification_report'])
    print(f"\nClassification Report {name} :\n", metrics['classification_report'])
    return metrics['classification_report']


def All_Unbalance_CWT(model, name):
    # Initialize TensorBoard writer

    writer = SummaryWriter(f'runs/{name}_experiment_1')
    # Usage:
    fhg_fft = FHGFFT()
    (train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_fft.extract_data()

    # 1. 数据加载器
    batch_size = 128

    # Splitting train dataset into train and validation sets
    train_dataset_full = TensorDataset(torch.tensor(train_data_fft).float().to(device),
                                       torch.tensor(train_labels_fft).long().to(device))
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_data_fft).float().to(device),
                                 torch.tensor(test_labels_fft).long().to(device))
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
            input_data = inputs.squeeze(-1)
            if name == 'FCNN' or name == 'AdversarialNet' or name == 'SAC':
                outputs = model(input_data)
            if name == 'CNN' or name == 'QCNN' or name == 'WDCNN':
                outputs = model(input_data.unsqueeze(1))
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
        if name == 'FCNN' or name == 'AdversarialNet' or name == 'SAC':
            dummy_input = torch.randn(batch_size, 2048)  # Adjust the size based on your model's expected input
            writer.add_graph(model, dummy_input.cuda())
        if name == 'CNN_CWT' or name == 'QCNN' or name == 'WDCNN':
            dummy_input = torch.randn(batch_size, 100, 2048)  # Adjust the size based on your model's expected input
            writer.add_graph(model, dummy_input.cuda())
        if name == 'RNN' or name == 'LSTM':
            dummy_input = torch.randn(batch_size, 2048, 1)  # Adjust the size based on your model's expected input
            writer.add_graph(model, dummy_input.cuda())

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

    # ALL 5 num_classes FCNN
    # All_Unbalance(model=FCNN().to(device), name='FCNN')
    # #
    # # # ALL 5 num_classes CNN
    # All_Unbalance(model=CNN().to(device), name='CNN')

    # # ALL 5 num_classes RNN
    # All_Unbalance(model=RNN().to(device), name='RNN')

    # # ALL 5 num_classes LSTM
    # All_Unbalance(model=LSTM().to(device), name='LSTM')

    # # ALL 5 num_classes AdversarialNet
    # All_Unbalance(model=AdversarialNet().to(device), name='AdversarialNet')

    # # # ALL 5 num_classes QCNN
    # All_Unbalance(model=QCNN().to(device), name='QCNN')

    # # ALL 5 num_classes WDCNN
    # All_Unbalance(model=WDCNN().to(device), name='WDCNN')

    # # ALL 5 num_classes WDCNN
    # All_Unbalance_CWT(model=CNN_CWT().to(device), name='CNN_CWT')

    # 0D 1D 两两组合
    prefixes = ["0D", "1D", "2D", "3D", "4D"]

    # FCNN
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E', model=FCNN(num_classes=2).to(device), name='FCNN')
    #
    # # CNN
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E', model=CNN(num_classes=2).to(device), name='CNN')

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
    # SAC
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E',
    #                      model=SAC(input_size=4096, sae_hidden_sizes=[1024, 512, 256, 128, 64],
    #                                classifier_hidden_size=128, num_classes=2).to(device),
    #                      name='SACD')

    # ALL 5 num_classes SAC
    All_Unbalance(
        model=SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64], classifier_hidden_size=128).to(device),
        name='SACD')

    # All_Unbalance_D(
    #     model=SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64], classifier_hidden_size=128).to(device),
    #     name='SACD')
    #
    # #
    # # # SAC
    # for i in range(len(prefixes)):
    #     for j in range(i + 1, len(prefixes)):
    #         # print(f"# {prefixes[i]} {prefixes[j]}")
    #         Enum_Esecnum_D(train_suffix=prefixes[i], test_suffix=f'{prefixes[i][0]}E', second_train_suffix=prefixes[j],
    #                      second_test_suffix=f'{prefixes[j][0]}E',
    #                      model=SAC(input_size=2048, sae_hidden_sizes=[1024, 512, 256, 128, 64],
    #                                classifier_hidden_size=128, num_classes=2).to(device),
    #                      name='SACD')
