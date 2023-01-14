import csv
import pickle
import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#训练数据
TRAIN_DATA='data/train/train_data.csv'#原始训练数据
TEST_DATA='data/test/test_data.csv'

TRAIN_CUTTED='data/train/train_cutted.csv'#分词结果
TEST_CUTTED='data/test/test_cutted.csv'#分词结果

TRAIN_SEG='data/train/train_seq.csv'#编码
TEST_SEG='data/test/test_seq.csv'#编码

TRAIN_LABLE='data/train/train_label.csv'#标签编码
TEST_LABLE='data/test/test_label.csv'

TEST_VALUE='data/test/test_value.csv'

train_data_value = pd.read_csv(TRAIN_CUTTED)
test_data_value = pd.read_csv(TEST_CUTTED)

#one-hot编码
#评论内容
train_cutted_seg=train_data_value.seg_cut
test_cutted_seg=test_data_value.seg_cut

#标签
train_cutted_label=train_data_value.label
test_cutted_label=test_data_value.label

label_encoder=LabelEncoder()
train_data_label=label_encoder.fit_transform(train_cutted_label).reshape(-1,1)
test_data=test_data_label=label_encoder.fit_transform(test_cutted_label).reshape(-1,1)

#one-hot
one_hot=OneHotEncoder()
train_data_label=one_hot.fit_transform(train_data_label).toarray()
test_data_label=one_hot.fit_transform(test_data_label).toarray()


#分词结果进行编码
token=Tokenizer(num_words=300)#只标记出现次数最多的
#训练集语料库
train_value=[]
#测试集语料
test_value=[]
#填充语料库
for value in train_cutted_seg.tolist():
    train_value.append(str(value))
for value in test_cutted_seg.tolist():
    test_value.append(str(value))
#统计训练集的语料库
token.fit_on_texts(train_value)

#保存
with open('data/coded/token.pickle', 'wb') as tok:
    pickle.dump(token,tok,protocol=pickle.HIGHEST_PROTOCOL)
with open('data/coded/token.pickle', 'rb') as tok:
    token=pickle.load(tok)

#调整序列长度\转换序列
train_sequences=token.texts_to_sequences(train_value)
test_sequences=token.texts_to_sequences(test_value)
train_seq=sequence.pad_sequences(train_sequences,maxlen=500)
test_seq=sequence.pad_sequences(test_sequences,maxlen=500)

a=[value for value in range(500)]
b=[value for value in range(2)]
with open(TRAIN_SEG, 'w') as f:
    write=csv.writer(f)
    write.writerow(a)
    write.writerows(train_seq)
with open(TRAIN_LABLE, 'w') as f:
    write=csv.writer(f)
    write.writerow(b)
    write.writerows(train_data_label)
with open(TEST_SEG,'w') as f:
    write=csv.writer(f)
    write.writerow(a)
    write.writerows(test_seq)
with open(TEST_LABLE,'w') as f:
    write=csv.writer(f)
    write.writerow(b)
    write.writerows(test_data_label)

with open(TEST_VALUE,'w') as f:
    write=csv.writer(f)
    write.writerow(['Label'])
    write.writerows(test_data)