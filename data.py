import csv
import re

import pandas as pd
import numpy as np
import jieba
from sklearn.preprocessing import LabelEncoder

train_number=6000
#训练数据
DATA='data/data.csv'
TRAIN_DATA='data/train/train_data.csv'#原始训练数据
TEST_DATA='data/test/test_data.csv'
TRAIN_CUTTED='data/train/train_cutted.csv'#分词结果
TEST_CUTTED='data/test/test_cutted.csv'#分词结果

#读取酒店评论数据集
dataset = pd.read_csv(DATA, delimiter=',')
data=np.array(dataset)

reviews=np.array(dataset['review'])
labels = np.array(dataset['label'])
labels = LabelEncoder().fit_transform(labels)

pos_sum = 0  # 正向评价
for i in range(len(labels)):
    if labels[i] == 1:
        pos_sum += 1

#划分训练集和测试集（中间6000条为训练数据，其余的为测试数据）
with open(TRAIN_DATA,'a') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['label','review'])
    f_csv.writerows(data[pos_sum-4800:pos_sum+1199])
with open(TEST_DATA,'a') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['label', 'review'])
    f_csv.writerows(data[:pos_sum-4800])
    f_csv.writerows(data[pos_sum+1199:])

#数据预处理（去除无用字符）
list1 = pd.read_csv('data/hit_stopwords.txt',encoding='utf-8',delimiter="\t",names=['t'])['t'].tolist()
def stopwordslist(sentence):
    #stopwords=[row.strip() for row in open('data/hit_stopwords.txt',encoding='UTF-8').readline()]
    #stopwords = [row.strip() for row in open('data/stopwords.txt', encoding='UTF-8').readline()]
    return [value for value in jieba.lcut(sentence) if value not in list1]
   # return stopwords

#分词
# def seg_department(sentence):
#     sentence_department=jieba.cut(sentence.strip(),cut_all=False)
#     stop_words=stopwordslist()
#     cut_result=''
#     for word in sentence_department:
#         if word!='\t':
#             if word not in stop_words:
#                 cut_result+=word
#                 cut_result+=" "
#     return cut_result

def sen_cut(filename,out_filename):
    sen_out = open(out_filename, 'w', encoding='UTF-8')
    writer=csv.writer(sen_out)
    writer.writerow(['label','seg_cut'])
    # label = []
    review = []
    with open(filename,'r',encoding='UTF-8') as f:
        reader=csv.DictReader(f)
        for row in reader:
            # label.append(row['label'])
            cut_rev=row['review']
            #row_seg=seg_department(cut_rev)
            row_seg = stopwordslist(cut_rev)
            out= ' '.join(list(row_seg))
            review.append(out)

            cut_list=[]
            cut_list.append(row['label'])
            cut_list.append(out)
            writer.writerow(cut_list)

    sen_out.close()

sen_cut(TRAIN_DATA,TRAIN_CUTTED)
sen_cut(TEST_DATA, TEST_CUTTED)