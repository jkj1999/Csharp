import nltk
import math
import string
import nltk.stem
import os
import os.path
import re
import sys
import codecs
import datetime as dt
from wordcloud import WordCloud
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')

#设置文本
text_2 = "Variations of the tf–idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query. tf–idf can be successfully used for stop-words filtering in various subject fields, including text summarization and classification."

def stem_count(text):
    l_text = text.lower()  # 全部转化为小写以方便处理
    without_punctuation = l_text.translate(punctuation_map)  # 去除文章标点符号
    tokens = nltk.word_tokenize(without_punctuation)  # 将文章进行分词处理,将一段话转变成一个list
    without_stopwords = [w for w in tokens if not w in stopwords.words('english')]  # 去除文章的停用词
    cleaned_text = []
    for i in range(len(without_stopwords)):
        cleaned_text.append(s.stem(without_stopwords[i]))  # 提取词干
    count = Counter(cleaned_text)  # 实现计数功能
    return count

# 定义TF-IDF的计算过程
def D_con(word, count_list):
    D_con = 0
    for count in count_list:
        if word in count:
            D_con += 1
    return D_con

def tf(word, count):
    return count[word] / sum(count.values())

def idf(word, count_list):
    return math.log(len(count_list)) / (1 + D_con(word, count_list))

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


months = [1,2,3,4,5,6,7,8,9,10,11,12]
years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

text_sum = ''

for year in years:
    for month in months:
        text_4 = ''
        path = '../data/'+str(year)+'/'+str(month)+'/'
        # 把目录下的文件名全部获取保存在files中
        files = os.listdir(path)
        for file in files:
            # 准确获取一个txt的位置，利用字符串的拼接
            txt_path = path + file
            data = open(txt_path, 'r', encoding='utf-8')
            line = data.readline()
            line = data.readline()
            line = data.readline()
            text_4 += line
        # print(text_4)

        punctuation_map = dict((ord(char), None) for char in string.punctuation)  # 引入标点符号，为下步去除标点做准备
        s = nltk.stem.SnowballStemmer('english')  # 在提取词干时,语言使用英语,使用的语言是英语

        fo = open(path + "result.txt", "a")

        dict1 = dict()

        texts = [text_4, text_2]
        count_list = []
        for text in texts:
            count_list.append(stem_count(text))  # 填入清洗好后的文本
        for i in range(len(count_list)):
            if (i == 0):
                # print('For document {}'.format(i + 1))
                # 获取当前时间
                now_time = dt.datetime.now().strftime('%F %T')
                # 输出时间
                print('Time: ' + now_time)

                tf_idf = {}
                for word in count_list[i]:
                    tf_idf[word] = tfidf(word, count_list[i], count_list)
                sort = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)  # 将集合按照TF-IDF值从大到小排列
                for word, tf_idf in sort[:100]:
                    # print("\tWord: {} : {}".format(word, round(tf_idf, 6)))
                    fo.write(word + ' ' + str(round(tf_idf, 6)) + '\n')
                    if(word != '±'):
                        dict1[word] = tf_idf
                word_cloud = WordCloud(scale=4, background_color='white', max_font_size=70)
                word_cloud.fit_words(dict1)
                plt.imshow(word_cloud)
                plt.xticks([])  # 去掉横坐标
                plt.yticks([])  # 去掉纵坐标
                plt.rcParams['savefig.dpi'] = 300  # 图片像素
                plt.rcParams['figure.dpi'] = 300  # 分辨率
                #imgstr = str(year)+"_"+str(month)+".png"
                plt.savefig("../img/2013_9.png")
        # 关闭打开的文件
        fo.close()

