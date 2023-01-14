import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model,load_model
from keras.layers import Embedding, Conv1D, MaxPool1D, concatenate, Flatten, Dropout, Dense

## 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  #指定默认字体 SimHei黑体
plt.rcParams['axes.unicode_minus'] = False   #解决保存图像是负号'

#建模
from sklearn import metrics

#训练数据
TRAIN_SEG='data/train/train_seq.csv'#编码
TEST_SEG='data/test/test_seq.csv'#编码
TRAIN_LABLE='data/train/train_label.csv'#标签编码
TEST_LABLE='data/test/test_label.csv'
TEST_VALUE='data/test/test_value.csv'

train_seq=pd.read_csv(TRAIN_SEG)
train_data_label=pd.read_csv(TRAIN_LABLE)
test_seq=pd.read_csv(TEST_SEG)
test_data_label=pd.read_csv(TEST_LABLE)
test_value=pd.read_csv(TEST_VALUE)

# print(len(test_seq))
# print(len(test_label))
# print(len(test_data_label))

max_words=5000
max_token=500
num_labels=2#标签种类
print("Model Training……")
data_input=Input(shape=(max_token,))
embed=Embedding(max_words+1,128,input_length=max_token,trainable=False)(data_input)
kernel_sizes=[3,4,5]
output=[]
for kernel_size in kernel_sizes:
    text_cnn=Conv1D(128,kernel_size,strides=1,padding='same',activation='relu')(embed)
    mp=MaxPool1D(pool_size=2)(text_cnn)
    output.append(mp)
output=concatenate([mp for mp in output],axis=-1)
flatten=Flatten()(output)
dropout=Dropout(0.2)(flatten)
data_out=Dense(2,activation='softmax')(dropout)
model=Model(inputs=data_input,outputs=data_out)
model.summary()
# model.compile( optimizer='adam',loss="categorical_crossentropy",metrics=["acc"])
model.compile( optimizer='adam',loss="binary_crossentropy",metrics=["binary_accuracy"])

# model.fit(train_seq,train_data_label,batch_size=256,epochs=200)
# model.save('text_cnn_train.h5')
# del model

#测试
model=load_model('text_cnn_train.h5')
label_p=model.predict(test_seq)

label_max=np.argmax(label_p,axis=1)
label_hat=list(map(str,label_max))
label_hat=[int(value) for value in label_hat]

print("ACC:",metrics.accuracy_score(test_value,label_hat))
print("F1:",metrics.f1_score(test_value,label_hat))
