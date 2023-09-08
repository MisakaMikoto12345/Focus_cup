
# 数据预处理

import pandas as pd
import re
import jieba.posseg as psg


#利用pandas导入爬取的评论
reviews = pd.read_csv(r'..\..\Data\sentiment_analysis\title_emotion.csv',encoding='GB18030')

reviews['title'].duplicated().sum()


#数据清洗
content = reviews['title']
pattern = re.compile('[a-zA-Z0-9]|贵州|茅台|白酒|中国')
content = content.apply(lambda x : pattern.sub('',x))


#自定义分词函数
worker = lambda s : [[x.word,x.flag] for x in psg.cut(s)] 
seg_word = content.apply(worker)


# 将词语转化为数据框形式，一列是词，一列是词语所在的句子id，最后一列是词语在该句子中的位置
 # 每一评论中词的个数
n_word = seg_word.apply(lambda x: len(x)) 


# 构造词语所在的句子id
n_content = [[x+1]*y for x,y in zip(list(seg_word.index), list(n_word))]
# 将嵌套的列表展开，作为词所在评论的id
index_content = sum(n_content, [])  

seg_word = sum(seg_word,[])
# 词
word = [x[0] for x in seg_word]
# 词性
nature = [x[1] for x in seg_word]


# content_type评论类型
content_type = [[x]*y for x,y in zip(list(reviews['emotion']),list(n_word))]


content_type = sum(content_type,[])

# 构造数据框
result = pd.DataFrame({'index_content': index_content,
                      'word' : word,
                      'nature': nature,
                      'emotion':content_type})


#观察nature列得，x表示标点符号
result = result[result['nature'] != 'x']


stop_path = open(r'..\..\Data\sentiment_analysis\stopwords.txt','r',encoding='utf-8')# 加载停用词
stop = [x.replace('\n','') for x in stop_path.readlines()]# 删除停用词
# 得到非停用词序列
word = list(set(word) - set(stop))
# 判断表格中的单词列是否在非停用词列中
result = result[result['word'].isin(word)]
result


pos=result[result['emotion'].isin([1])]#积极语料数据集
neg=result[result['emotion'].isin([0])]#消极语料数据集


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 按word分组统计数目
frequencies = result.groupby(by = ['word'])['word'].count()
# 按数目降序排序
frequencies = frequencies.sort_values(ascending = False)



# 从文件中将图像读取为数组
backgroud_Image=plt.imread(r'..\..\Data\sentiment_analysis\pl.jpg')


wordcloud = WordCloud(width=1000,height=700,font_path=r'..\..\Data\sentiment_analysis\msyh.ttc',# 这里的字体要与自己电脑中的对应
                      max_words=100,            # 选择前100词
                      background_color='white',  # 背景颜色为白色
                      mask=backgroud_Image)

my_wordcloud = wordcloud.fit_words(frequencies)

# 将数据展示到二维图像上
plt.imshow(my_wordcloud)
# 关掉x,y轴
plt.axis('off')
plt.show()


# 将结果写出
result.to_csv(r'..\..\Data\sentiment_analysis\word.csv', index = False, encoding = 'GB18030')


# 按word分组统计数目
frequencies = pos.groupby(by = ['word'])['word'].count()
# 按数目降序排序
frequencies = frequencies.sort_values(ascending = False)
# 从文件中将图像读取为数组
backgroud_Image=plt.imread(r'..\..\Data\sentiment_analysis\pl.jpg')
wordcloud = WordCloud(width=1000,height=700,font_path=r'..\..\Data\sentiment_analysis\msyh.ttc',# 这里的字体要与自己电脑中的对应
                      max_words=100,            # 选择前100词
                      background_color='white',  # 背景颜色为白色
                      mask=backgroud_Image)
my_wordcloud = wordcloud.fit_words(frequencies)
# 将数据展示到二维图像上
plt.imshow(my_wordcloud)
# 关掉x,y轴
plt.axis('off')
plt.show()


# 将结果写出
result.to_csv(r'..\..\Data\sentiment_analysis\word_pos.csv', index = False, encoding = 'GB18030')

# 按word分组统计数目
frequencies = neg.groupby(by = ['word'])['word'].count()
# 按数目降序排序
frequencies = frequencies.sort_values(ascending = False)
# 从文件中将图像读取为数组
backgroud_Image=plt.imread(r'..\..\Data\sentiment_analysis\pl.jpg')
wordcloud = WordCloud(width=1000,height=700,font_path=r'..\..\Data\sentiment_analysis\msyh.ttc',# 这里的字体要与自己电脑中的对应
                      max_words=100,            # 选择前100词
                      background_color='white',  # 背景颜色为白色
                      mask=backgroud_Image)
my_wordcloud = wordcloud.fit_words(frequencies)
# 将数据展示到二维图像上
plt.imshow(my_wordcloud)
# 关掉x,y轴
plt.axis('off')
plt.show()

# 将结果写出
result.to_csv(r'..\..\Data\sentiment_analysis\word_neg.csv', index = False, encoding = 'GB18030')

