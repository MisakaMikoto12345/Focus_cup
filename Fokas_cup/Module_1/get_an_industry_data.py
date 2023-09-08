import pandas as pd
from selenium import webdriver
import requests
import csv
import time
import os

def download_code(url,filepath):
    resp = requests.get(url)
    all_data = resp.json()
    with open(filepath,'w') as f:
        writer = csv.writer(f,dialect = 'unix')
        for data in all_data:
            code = data['code']
            # print(code)
            name = data['name']
            writer.writerow([code,name])


def get_data_url(filepath):
    df_code = pd.read_csv(filepath,encoding = 'gbk',header = None)
    historical_transaction_data_url = []
    for i in range(len(df_code)):
        code = str(df_code.iloc[i,0]).zfill(6)
        data_url = "http://quotes.money.163.com/trade/lsjysj_" + str(code) + ".html#01b07"
        historical_transaction_data_url.append(data_url)
    # print(historical_transaction_data_url)
    return historical_transaction_data_url


def download_historical_transaction_data(url_list,download_path):
    options = webdriver.ChromeOptions()
    prefs = {'download.default_directory':download_path }
    options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options=options)

    driver.implicitly_wait(30)  # 隐式等待
    for i in range(len(url_list)):
        driver.get(url_list[i])
        driver.find_element_by_id("downloadData").click() #找到“下载数据”并点击
        driver.find_element_by_name("date_start_value").clear() #清除“起始日期”输入框
        driver.find_element_by_name("date_start_value").send_keys('2020-10-01') #“起始日期”输入框重新赋值
        driver.find_element_by_name("date_end_value").clear() #清除“截止日期”输入框
        driver.find_element_by_name("date_end_value").send_keys('2022-11-10') #“截止日期”输入框重新赋值
        driver.find_element_by_xpath('//div[@class = "align_c"]/a').click() #找到“下载”并点击
    time.sleep(5)
    driver.quit()



if __name__ == '__main__':
    code_url = \
        'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=80&sort=symbol&asc=1&node=new_ljhy&symbol=&_s_r_a=init'
    filepath = r'..\Data\Stock\stock_code.csv'

    download_code(code_url,filepath)

    historical_transaction_data_url = get_data_url(filepath)

    path = os.path.abspath(os.path.dirname(os.getcwd()))
    download_path = path + '\Data\Stock'

    download_historical_transaction_data(historical_transaction_data_url,download_path)

    print('爬取完毕')


