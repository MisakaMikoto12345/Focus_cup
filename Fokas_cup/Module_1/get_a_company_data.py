from selenium import webdriver
import os
import time

def download_historical_transaction_data(company_code,start_time,end_time,download_path):
    options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': download_path}
    options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options = options)

    driver.implicitly_wait(30)  # 隐式等待
    data_url = "http://quotes.money.163.com/trade/lsjysj_" + str(company_code) + ".html#01b07"
    driver.get(data_url)
    driver.find_element_by_id("downloadData").click() #找到“下载数据”并点击
    driver.find_element_by_name("date_start_value").clear() #清除“起始日期”输入框
    driver.find_element_by_name("date_start_value").send_keys(start_time) #“起始日期”输入框重新赋值
    driver.find_element_by_name("date_end_value").clear() #清除“截止日期”输入框
    driver.find_element_by_name("date_end_value").send_keys(end_time) #“截止日期”输入框重新赋值
    driver.find_element_by_xpath('//div[@class = "align_c"]/a').click() #找到“下载”并点击
    time.sleep(5)
    driver.quit()

if __name__ == '__main__':
    company_code = input('请输入股票代码（A股）(例如：000001)：')
    start_time = input('请输入起始日期(例如：2020-01-01)：')
    end_time = input('请输入结束日期(例如：2020-01-01)：')
    path = os.path.abspath(os.path.dirname(os.getcwd()))
    download_path = path+'\Data\Stock'
    download_historical_transaction_data(company_code,start_time,end_time,download_path)

    print('爬取完毕')