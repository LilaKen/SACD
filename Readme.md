## dataset

dataset: 存放数据集的文件夹，打开dataset目录下的Readme文件可以下载本论文公开数据集。



## main

main: 存放主要运行.py文件，outputs目录存放各模型的最好的权重参数文件， main目录下存放log文件为论文实验日志文件（运行代码生成的log）， main.py包含本实验中大部分模型的代码，main_cwt.py存放的是经过CWT变换的代码以及模型实验(仅仅CNN)， main_bls.py存放的是宽度学习网络的代码，main_ml.py存放的是机器学习方法在该数据集上的实验代码。



## models

models：包含FCNN,CNN,BLS,LSTM,QCNN,WDCNN,RNN,SAC,SACD的模型架构代码。



## utils

utils: utils目录下的data_process_cwt.py为连续小波变换的代码，dataset.py存放加载数据集的各个实现类，seed.py为固定种子文件，种子取2023，即进行实验的年份。





## 疑惑

如果对于代码文件有什么疑问，可以联系作者邮箱:lil_ken@163.com。

ps:第一份公开代码实验，可能有很多不周到的地方，还请各位研究者多多谅解。

