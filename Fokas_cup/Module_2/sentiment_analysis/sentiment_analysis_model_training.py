
# coding=GB18030

#����SnowNLP��ģ����
from snownlp import sentiment
import pandas as pd
import os

from sklearn import metrics
import numpy as np

# ����Ԥ����

#��ȡ����
reviews = pd.read_csv(r'..\..\Data\sentiment_analysis\Basis.csv',encoding='gbk')
#�ָ����ݼ�
pos=reviews[reviews['emotion'].isin([1])]#�����������ݼ�
neg=reviews[reviews['emotion'].isin([0])]#�����������ݼ�

pos_training=pos.iloc[:int(pos.shape[0]*4/5),:]
pos_test=pd.concat([pos,pos_training,pos_training]).drop_duplicates(keep=False)
neg_training=neg.iloc[:int(pos.shape[0]*4/5),:]
neg_test=pd.concat([neg,neg_training,neg_training]).drop_duplicates(keep=False)
test_set=pos_test.append(neg_test)

pos_2=""
neg_2=""

#���ɻ��������ı�
for i in pos_training['title']:
    i="\n"+i
    pos_2=pos_2+i
with open(r'..\..\Data\sentiment_analysis\pos.txt','w',encoding='utf-8') as f:
   f.write(pos_2)
#�������������ı�
for i in neg_training['title']:
    i="\n"+i
    neg_2=neg_2+i
with open(r'..\..\Data\sentiment_analysis\neg.txt','w',encoding='utf-8') as f:
   f.write(neg_2)

# ģ��ѵ��

#��з�����SentimentԴ�������
'''
class Sentiment(object):

    def __init__(self):
        self.classifier = Bayes() # ʹ�õ���Bayes��ģ��

    def save(self, fname, iszip=True):
        self.classifier.save(fname, iszip) # �������յ�ģ��

    def load(self, fname=data_path, iszip=True):
        self.classifier.load(fname, iszip) # ���ر�Ҷ˹ģ��

    # �ִ��Լ�ȥͣ�ôʵĲ���    
    def handle(self, doc):
        words = seg.seg(doc) # �ִ�
        words = normal.filter_stop(words) # ȥͣ�ô�
        return words # ���طִʺ�Ľ��

    def train(self, neg_docs, pos_docs):
        data = []
        # ���븺����
        for sent in neg_docs:
            data.append([self.handle(sent), 'neg'])
        # ����������
        for sent in pos_docs:
            data.append([self.handle(sent), 'pos'])
        # ���õ���Bayesģ�͵�ѵ������
        self.classifier.train(data)

    def classify(self, sent):
        # 1������sentiment���е�handle����
        # 2������Bayes���е�classify����
        ret, prob = self.classifier.classify(self.handle(sent)) # ���ñ�Ҷ˹�е�classify����
        if ret == 'pos':
            return prob
        return 1-prob

'''

sentiment.train(r'..\..\Data\sentiment_analysis\neg.txt', r'..\..\Data\sentiment_analysis\pos.txt')
sentiment.save(r'..\..\Data\Model\sentiment_marshal')

data_path = os.path.join(os.path.dirname(os.path.abspath("__file__")),r'..\..\Data\Model\sentiment_marshal')#ָ����ѵ����ģ��

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