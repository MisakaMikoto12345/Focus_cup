import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


    # 模型读取

def model_reading(path):
    model = torch.load(path)

    return model

    #文件读取

def read_data(filepath):
    file = filepath  # 文件路径
    data = pd.read_csv(file, encoding='gbk', index_col=0)  # 读入文件
    data.drop(columns=['股票代码', '名称'], inplace=True)
    data.replace("None", np.nan, inplace=True)  # 将‘None’替换为Numpy空值‘nan’
    data.dropna(axis=0, inplace=True)  # 删除空值行
    data.sort_index(inplace=True)
    data = data.iloc[-19:, :]

    return data

    # 归一化

def data_normalization(data):
    price = data.copy()
    scaler = {}  # 创建一个字典，存储每一列不同归一化参数的函数
    for i in range(12):
        scaler[i] = MinMaxScaler(feature_range=(-1, 1))
        price.iloc[:, i] = scaler[i].fit_transform(price.iloc[:, i].values.reshape(-1, 1))

    return price, scaler

    # 预测未来一天股价

def stock_pre(stock,lookback,model):
    data_raw = stock.to_numpy()
    data_set = []

    data_set.append(data_raw[0:0 + lookback])
    data_set = np.array(data_set)

    x = data_set[-1:,:,:]
    x = data_transform(x)

    y = model(x)
    return x, y

    # 将数据转化为张量（一个可以运行在GPU上的多维数据，加快计算效率）

def data_transform(x):
    x = torch.Tensor(x)

    return x



if __name__ == '__main__':

    model = model_reading(r'..\..\Data\Model\LSTM_A.pt')

    filepath = r'..\..\Data\Stock\600519.csv'
    data = read_data(filepath)

    price, scaler = data_normalization(data)

    lookback = 19
    x, y = stock_pre(price,lookback,model)

    y = scaler[0].inverse_transform(y.detach().numpy())
    print(y)




