import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import time
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']   # 设置图中字的字体

    # 模型读取

def model_reading(path):
    model = torch.load(path)

    return model

    #文件读取

def read_data(filepath):
    file = filepath  # 文件路径
    data = pd.read_csv(file, encoding='gbk', index_col=0)  # 读入文件
    name = data.iloc[0, 1]
    data = data[['收盘价']]
    data.replace("None", np.nan, inplace=True)  # 将‘None’替换为Numpy空值‘nan’
    data.dropna(axis=0, inplace=True)  # 删除空值行
    data.sort_index(inplace=True)
    data = data.iloc[-19:, :]

    return data,name

    # 归一化

def data_normalization(data):
    price = data.copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price.iloc[:, 0] = scaler.fit_transform(price.iloc[:, 0].values.reshape(-1, 1))

    return price, scaler

    # 预测未来pre_days天股价

def stock_pre(stock,lookback,pre_days,model):
    data_raw = stock.to_numpy()
    data_set = []
    data_pre = []

    data_set.append(data_raw[0:0 + lookback])
    data_set_array = np.array(data_set)

    for i in range(pre_days):
        i += 1
        x = data_set_array[-1:, :, :]
        x = data_transform(x)

        y = model(x)
        data_pre.append(y)

        y = y.detach().numpy()

        data_raw = data_raw.tolist()
        data_raw.append(y[0])       #将预测的y加入data_raw以组成新的x
        data_raw = np.array(data_raw)
        data_set.append(data_raw[i:i + lookback])
        data_set_array = np.array(data_set)

    return data_pre

    # 将数据转化为张量（一个可以运行在GPU上的多维数据，加快计算效率）

def data_transform(x):
    x = torch.Tensor(x)

    return x


if __name__ == '__main__':

    model = model_reading(r'..\..\Data\Model\LSTM_B.pt')

    filepath = r'..\..\Data\Stock\600519.csv'
    data,name = read_data(filepath)
    print(name)

    price, scaler = data_normalization(data)

    lookback = 19
    pre_days = 30
    total_days = lookback+pre_days
    data_pre = stock_pre(price,lookback,pre_days,model)

    for i in range(pre_days):
        data_pre[i] = scaler.inverse_transform(data_pre[i].detach().numpy())
        data_pre[i] = data_pre[i].tolist()
        data_pre[i] = data_pre[i][0][0]

    print(data_pre)

    # 结果可视化

    fin=str(data.index[-1])

    timeArray = time.strptime(data.index[-1],"%Y-%m-%d")
    timeStamp = int(time.mktime(timeArray))

    last_date_st = timeStamp
    next_date_st = last_date_st + 86400

    pt=[]
    for i in range(pre_days):
        next_date_st = last_date_st + 86400
        last_date_st=next_date_st
        x=time.strftime('%Y-%m-%d',time.localtime(next_date_st))
        pt.append(x)        # 获得从data最后一天开始往后pre_days天的日期
    print(pt)

    data_new = data
    data_new['预测值'] = np.nan

    for i in range(pre_days):
        data_new.loc[pt[i]] = [np.nan for _ in range(len(data_new.columns) - 1)] + [data_pre[i]]

    plt.figure(figsize=(20, 10))
    plt.plot(data_new.iloc[-total_days:-pre_days, 0], color='red')
    plt.plot(data_new.iloc[-pre_days:, 1], color='green')
    for x, y in zip(data_new.index[-total_days:-pre_days], data_new.iloc[-total_days:-pre_days, 0]):
        plt.text(x, y + 10, '%.0f' % y, ha='center', va='bottom', fontsize=10)
    for x, y in zip(data_new.index[-pre_days:], data_new.iloc[-pre_days:, 1]):
        plt.text(x, y + 10, '%.0f' % y, ha='center', va='bottom', fontsize=10)
    plt.title(name+'股价预测', fontsize=18, fontweight='bold')
    plt.xlabel('日期', fontsize=18)
    plt.ylabel('收盘价', fontsize=18)
    plt.legend(['往日价格', '预测价格'])
    plt.xticks(rotation=45)
    plt.savefig(r'..\..\Data\Stock_Prediction'+'\\'+name+'股价预测', dpi=300)
    plt.show()

