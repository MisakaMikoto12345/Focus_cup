# coding=GB18030

import pandas as pd
import snownlp as sn
import os

data_path = os.path.join(os.path.dirname(os.path.abspath("__file__")),r'..\..\Data\Model\sentiment_marshal')#ָ����ѵ����ģ��

def sentiment(text):
    s = sn.SnowNLP(text)
    return s.sentiments

title = pd.read_csv(r'..\..\Data\sentiment_analysis\title.csv',encoding='GB18030')
title.loc[:,'emotion'] = 0
emotion_index = 0
for i in range(len(title)):
    print(sentiment(title.iloc[i,0]))
    if sentiment(title.iloc[i,0]) > 0.5:
        title.iloc[i, 1] = 1
        emotion_index += title.iloc[i, 1]

if (emotion_index/len(title)) > 0.5:
    print('���ڹ������Ϊ����')
else:
    print('���ڹ������Ϊ����')

title.to_csv(r'..\..\Data\sentiment_analysis\title_emotion.csv',encoding = 'GB18030',index = False)