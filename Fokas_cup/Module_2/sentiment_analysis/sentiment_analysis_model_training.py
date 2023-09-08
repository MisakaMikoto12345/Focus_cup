
# coding=GB18030

#利用SnowNLP建模分类
from snownlp import sentiment
import pandas as pd
import os

from sklearn import metrics
import numpy as np

# 数据预处理

#读取数据
reviews = pd.read_csv(r'..\..\Data\sentiment_analysis\Basis.csv',encoding='gbk')
#分割数据集
pos=reviews[reviews['emotion'].isin([1])]#积极语料数据集
neg=reviews[reviews['emotion'].isin([0])]#消极语料数据集

pos_training=pos.iloc[:int(pos.shape[0]*4/5),:]
pos_test=pd.concat([pos,pos_training,pos_training]).drop_duplicates(keep=False)
neg_training=neg.iloc[:int(pos.shape[0]*4/5),:]
neg_test=pd.concat([neg,neg_training,neg_training]).drop_duplicates(keep=False)
test_set=pos_test.append(neg_test)

pos_2=""
neg_2=""

#生成积极语料文本
for i in pos_training['title']:
    i="\n"+i
    pos_2=pos_2+i
with open(r'..\..\Data\sentiment_analysis\pos.txt','w',encoding='utf-8') as f:
   f.write(pos_2)
#生成消极语料文本
for i in neg_training['title']:
    i="\n"+i
    neg_2=neg_2+i
with open(r'..\..\Data\sentiment_analysis\neg.txt','w',encoding='utf-8') as f:
   f.write(neg_2)

# 模型训练

#情感分析中Sentiment源代码解析
'''
class Sentiment(object):

    def __init__(self):
        self.classifier = Bayes() # 使用的是Bayes的模型

    def save(self, fname, iszip=True):
        self.classifier.save(fname, iszip) # 保存最终的模型

    def load(self, fname=data_path, iszip=True):
        self.classifier.load(fname, iszip) # 加载贝叶斯模型

    # 分词以及去停用词的操作    
    def handle(self, doc):
        words = seg.seg(doc) # 分词
        words = normal.filter_stop(words) # 去停用词
        return words # 返回分词后的结果

    def train(self, neg_docs, pos_docs):
        data = []
        # 读入负样本
        for sent in neg_docs:
            data.append([self.handle(sent), 'neg'])
        # 读入正样本
        for sent in pos_docs:
            data.append([self.handle(sent), 'pos'])
        # 调用的是Bayes模型的训练方法
        self.classifier.train(data)

    def classify(self, sent):
        # 1、调用sentiment类中的handle方法
        # 2、调用Bayes类中的classify方法
        ret, prob = self.classifier.classify(self.handle(sent)) # 调用贝叶斯中的classify方法
        if ret == 'pos':
            return prob
        return 1-prob

'''

sentiment.train(r'..\..\Data\sentiment_analysis\neg.txt', r'..\..\Data\sentiment_analysis\pos.txt')
sentiment.save(r'..\..\Data\Model\sentiment_marshal')

data_path = os.path.join(os.path.dirname(os.path.abspath("__file__")),r'..\..\Data\Model\sentiment_marshal')#指向新训练的模型

import snownlp as sn
 
def sentiment(text):
    s = sn.SnowNLP(text)
    return s.sentiments

pre_metrics=[]
for i in test_set['title']:
    x=sentiment(i)
    if x >0.5:
        x=1
        pre_metrics.append(x)
    if x<=0.5:
        x=0
        pre_metrics.append(x)

test_metrics=np.array(test_set['emotion'])
test_metrics=test_metrics.tolist()

print(metrics.confusion_matrix(test_metrics,pre_metrics))
print(metrics.classification_report(test_metrics,pre_metrics ))