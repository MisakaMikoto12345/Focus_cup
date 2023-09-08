# coding=GB18030

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

httpheaders={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36'}
urlbase="http://guba.eastmoney.com/list,600519_"

textlst=[]

for j in range(6):
    url=urlbase+str(j+1)+".html"
    print(url)
    p=requests.get(url, headers=httpheaders)
    p.encoding = 'utf-8'
    webhtml=p.text
    soup=BeautifulSoup(webhtml,'html.parser')
    x=soup.find_all('div',attrs={'class':'articleh normal_post'})
    for item in x:
        textlst.append(item.select('span a')[0].string)
    print(len(textlst))
    time.sleep(3)

data_dict = {'title':textlst}
data_dict_df = pd.DataFrame(data_dict)
data_dict_df.to_csv(r'..\Data\sentiment_analysis\title_2.csv',encoding = 'GB18030',index = False)

print('爬取完毕')