import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
import torch

from LSTM_B_Module import LSTM_B

import time

import math
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
import plotly as py

    # 数据预处理

def read_data(filepath):
    file = filepath  # 文件路径
    data = pd.read_csv(file, encoding='gbk', index_col=0)  # 读入文件
    name = data.iloc[0, 1]
    data = data[['收盘价']]
    data.replace("None", np.nan, inplace=True)  # 将‘None’替换为Numpy空值‘nan’
    data.dropna(axis=0, inplace=True)  # 删除空值行
    data.sort_index(inplace=True)

    return data,name

    # 作图观察一下收盘价走势

def plot_recent_stock_prices(data, i):  # i 为要观察的天数
    sns.set_style('darkgrid')
    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 设置图中字的字体
    plt.figure(figsize=(15, 10))
    plt.plot(data['收盘价'][-i:])
    plt.xticks(rotation=45)
    plt.title('Stock Price', fontsize=18, fontweight='bold')
    plt.xlabel('日期', fontsize=18)
    plt.ylabel('收盘价', fontsize=18)
    plt.show()

    # 归一化

def data_normalization(data):
    price = data.copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price.iloc[:, 0] = scaler.fit_transform(price.iloc[:, 0].values.reshape(-1, 1))

    return price, scaler

    # 切分数据集

def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data_set = []

    for index in range(len(data_raw) - lookback):
        data_set.append(data_raw[index:index + lookback])

    data = np.array(data_set)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, 0:1]

    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, 0:1]

    return [x_train, y_train, x_test, y_test]

    # 将数据集转化为张量（一个可以运行在GPU上的多维数据，加快计算效率）

def data_transform(x_train, y_train, x_test, y_test):
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train_lstm = torch.Tensor(y_train)
    y_test_lstm = torch.Tensor(y_test)

    return [x_train, x_test, y_train_lstm, y_test_lstm]

    # 模型训练

def model_training(input_dim,hidden_dim,num_layers,output_dim,num_epochs,x_train,y_train_lstm):
    model = LSTM_B(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
    criterion = torch.nn.MSELoss()  # 定义均方差损失函数
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器

    hist = np.zeros(num_epochs)
    start_time = time.time()

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print('Epoch', t, 'MSE:', loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()     # loss反向传播
        optimiser.step()    # 梯度更新

    training_time = time.time() - start_time
    print('Training time :{}'.format(training_time))

    return model

    # 模型保存

def model_save(model, path):
    torch.save(model, path)

    # 模型可视化

def model_visualization(y_train_pred, y_train):
    predict = pd.DataFrame(y_train_pred)
    original = pd.DataFrame(y_train)

    plt.figure(figsize=(15, 10))
    sns.lineplot(x=original.index[-1000:], y=original[0][-1000:], label='Data', color='royalblue')
    sns.lineplot(x=predict.index[-1000:], y=predict[0][-1000:], label='Training Prediction(LSTM)', color='tomato')
    plt.title('Stock Price', size=14)
    plt.xlabel('Days', size=14)
    plt.ylabel('Price', size=14)
    plt.show()

    # 模型验证

def model_validation(y_train,y_train_pred,y_test,y_test_pred):
    trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


if __name__ == '__main__':
    filepath = r'..\..\Data\Stock\600519.csv'
    data,name = read_data(filepath)

    plot_recent_stock_prices(data, 30)

    price, scaler = data_normalization(data)

    lookback = 20
    x_train, y_train, x_test, y_test = split_data(price, lookback)

    x_train, x_test, y_train_lstm, y_test_lstm, = data_transform(x_train,y_train,x_test,y_test)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 1000
    model = model_training(input_dim,hidden_dim,num_layers,output_dim,num_epochs,x_train,y_train_lstm)

    y_train_pred = model(x_train)
    y_test_pred = model(x_test)

    #反归一化
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())

    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

    path = r'../../Data/Model/LSTM_B.pt'
    model_save(model,path)

    model_visualization(y_train_pred, y_train)

    model_validation(y_train,y_train_pred,y_test,y_test_pred)


    new_price = price[['收盘价']]

    trainPredictPlot = np.empty_like(new_price)
    trainPredictPlot[:, 0] = np.nan
    trainPredictPlot[lookback:len(y_train_pred) + lookback, :] = y_train_pred

    testPredictPlot = np.empty_like(new_price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred) + lookback - 1:len(price) - 1, :] = y_test_pred

    original = scaler.inverse_transform(price['收盘价'].values.reshape(-1, 1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                                        mode='lines',
                                        name='Train prediction')))
    fig.add_trace(go.Scatter(x=result.index, y=result[1],
                             mode='lines',
                             name='Test prediction'))
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                                        mode='lines',
                                        name='Actual Value')))
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white'
            ),
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white'
            ),
        ),
        showlegend=True,
        template='plotly_dark'
    )

    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='Result(LSTM)',
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)

    py.offline.plot(fig, filename=r'..\..\Data\Model\result_B.html', auto_open=False)

    print(name + '股票预测模型训练完毕')