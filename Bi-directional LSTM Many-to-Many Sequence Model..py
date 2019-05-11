
# coding: utf-8

# In[1]:


import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir(r'H:\document_classification\Dataset\training-data')


# In[4]:


os.getcwd()


# In[5]:


import glob


# In[6]:


filenames=glob.glob('*.txt')


# In[9]:


#Creating the training dataframe
path="H:\\document_classification\\Dataset\\training-data"
training=[]
for i in range(len(filenames)):
    new_path=path+'\\'+filenames[i]
    with open(new_path,'r') as infile:
        data=infile.read()
        data=data.split('\n')
        if(filenames[i][0]=='a'):
            train=[words.split("\t")[1] for words in data if (words!='')&(words!='-----')]
        else:
            train=[words for words in data if (words!='')&(words!='-----')]
        training.append(train)


# In[10]:


#Creating the training output 
import numpy as np
import re
y=[]
for i in range(0,len(training)):
    temp=[]
    for j in range(len(training[i])):
        training[i][j]=re.sub('[^A-Za-z0-9]',' ',training[i][j])
        s=[j+1]*len(training[i][j].split())
        temp.extend(s)
        k=np.array(temp)
        k=k.reshape(len(k),1)
        from sklearn.preprocessing import OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse=False)
        k = onehot_encoder.fit_transform(k)
    #k=pad_sequences()
    y.append(k)
print(len(y))


# In[11]:


#(each record belongs to one of the 10 segments)
temp2=[]
for i in range(0,len(y)):
    temp=[]
    for j in range(len(y[i])):
        temp1=list(y[i][j])
        temp1.extend([0]*(10-len(temp1)))
        temp.append(temp1)
    temp2.append(temp)


# In[12]:


y=temp2


# In[13]:


#Loading the word embedding model
file = r"H:\\glove.6B.100d.txt"
import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
model_dict= loadGloveModel(file) 


# In[ ]:


#Converting the word into vector
train=[]
for i in range(0,len(training)):
    temp1=[]
    for j in range(len(training[i])):
        training[i][j]=training[i][j].lower()
        #training[i][j]=re.sub('[^A-Za-z0-9]',' ',training[i][j])
        sent=training[i][j].split()
        for words in sent:
            if words in model_dict:
                temp1.append(list(model_dict[words]))
            else:
                temp1.append(list(np.zeros(100)))
    train.append(temp1)


# In[13]:


#Padding for making the tensor
for i in range(len(train)):
    train[i].extend([[0]*100]*(2500-len(train[i])))


# In[14]:


padded1=np.array(train)


# In[15]:


padded1.shape


# In[16]:


for i in range(len(y)):
    y[i].extend([[0]*10]*(2500-len(y[i])))


# In[17]:


len(y[0][0])


# In[18]:


X_train=padded1


# In[19]:


X_train.shape


# In[20]:


y_train=np.array(y)


# In[21]:


y_train.shape


# In[22]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Masking
from keras.layers import Bidirectional
 


# In[23]:


##########################Adding Validation Dataset####################################
os.chdir(r'H:\document_classification\Dataset\val-data')
filenames_val=glob.glob('*.txt')


# In[24]:


path='H:\\document_classification\\Dataset\\val-data'
validation=[]
for i in range(len(filenames_val)):
    new_path=path+'\\'+filenames_val[i]
    with open(new_path,'r') as infile:
        data=infile.read()
        data=data.split('\n')
        train_val=[words for words in data if (words!='')&(words!='-----')]
        validation.append(train_val)


# In[25]:


import numpy as np
import re
y_val=[]
for i in range(0,len(validation)):
    temp=[]
    for j in range(len(validation[i])):
        validation[i][j]=re.sub('[^A-Za-z0-9]',' ',validation[i][j])
        s=[j+1]*len(validation[i][j].split())
        temp.extend(s)
        k=np.array(temp)
        k=k.reshape(len(k),1)
        from sklearn.preprocessing import OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse=False)
        k = onehot_encoder.fit_transform(k)
    #k=pad_sequences()
    y_val.append(k)
print(len(y_val))


# In[26]:


temp2=[]
for i in range(0,len(y_val)):
    temp=[]
    for j in range(len(y_val[i])):
        temp1=list(y_val[i][j])
        temp1.extend([0]*(10-len(temp1)))
        temp.append(temp1)
    temp2.append(temp)


# In[27]:


y_val=temp2


# In[28]:


val=[]
for i in range(0,len(validation)):
    temp1=[]
    for j in range(len(validation[i])):
        validation[i][j]=validation[i][j].lower()
        #training[i][j]=re.sub('[^A-Za-z0-9]',' ',training[i][j])
        sent=validation[i][j].split()
        for words in sent:
            if words in model_dict:
                temp1.append(list(model_dict[words]))
            else:
                temp1.append(list(np.zeros(100)))
    val.append(temp1)


# In[29]:


for i in range(len(val)):
    val[i].extend([[0]*100]*(2500-len(val[i])))


# In[31]:


val_train=np.array(val)


# In[32]:


for i in range(len(y_val)):
    y_val[i].extend([[0]*10]*(2500-len(y_val[i])))


# In[33]:


val_test=np.array(y_val)


# In[34]:


val_train.shape


# In[35]:


val_test.shape


# In[36]:


#Buiding the Bidirectional model for classification
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(2500,100)))
model.add(Bidirectional(LSTM(100, input_shape=(2500,100), return_sequences=True)))
#model.add(Dense(50))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[37]:


X_train.shape


# In[38]:


y_train.shape


# In[39]:


val_train.shape


# In[40]:


val_test.shape


# In[41]:


model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1,validation_data=(val_train,val_test))


# In[63]:


classifier= model 


# In[66]:


os.chdir("H:\\document_classification\\Dataset\\test-data")
path="H:\\document_classification\\Dataset\\test-data"
for dirpath,dirnames,filenames in os.walk(path):
    filname=filenames 


# In[67]:


path="H:\\document_classification\\Dataset\\test-data"
test=[]
for i in range(len(filenames)):
    new_path=path+'\\'+filenames[i]
    with open(new_path,'r') as infile:
        data=infile.read()
        test.append(data)


# In[68]:


file = r"H:\\glove.6B.100d.txt"
import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
model_dict=loadGloveModel(file)


# In[70]:


original_test = []
new_test=[]
for i in range(0,len(test)):
    temp1=[]
    test[i]=test[i].lower()
    l=re.sub('[^A-Za-z0-9]',' ',test[i])
    temp2=l.split()
    original_test.append(temp2)
    for words in temp2:
        if words in model_dict:
            temp1.append(list(model_dict[words]))
        else:
            temp1.append(list(np.zeros(100)))
    new_test.append(temp1)


# In[142]:


from keras.preprocessing.sequence import pad_sequences
padded_test=pad_sequences(new_test,padding='post',maxlen=2500)


# In[71]:


for i in range(len(new_test)):
    new_test[i].extend([[0]*100]*(2500-len(new_test[i])))


# In[72]:


X_test=np.array(new_test)


# In[73]:


X_test.shape


# In[74]:


k=model.predict(X_test)


# In[76]:


y_pred=[]
for j in range(len(k)):
    pred=[]
    for item in k[j]:
        temp=[]
        item=list(item)
        for items in item:
            if(items==max(item)):
                temp.append(1)
            else:
                temp.append(0)
        pred.append(temp)
    y_pred.append(pred)


# In[77]:


corr_pred = []
for i in range(0,len(test)):
    corr_pred.append(y_pred[i][0:len(original_test[i])])


# In[78]:


fin_pred=[]
for i in range(0,len(k)):
    output=list(map(sum,zip(*corr_pred[i])))
    l=0
    new_pred=[]
    for items in output:
        if(l<len(original_test[i])):
            l=l+items
            new_pred.append(l)
    fin_pred.append(new_pred[:-1])


# In[79]:


dict_test=[]
for i in range(0,len(test)):
    test[i]=test[i].lower()
    l=re.sub('[^A-Za-z0-9]',' ',test[i])
    temp2=l.split()
    dict_test.append(temp2)


# In[167]:


temp=[1]
for i in range(len(fin_pred[0])):
    r=dict_test[0][fin_pred[0][i]+1]
    s=dict_test[0][fin_pred[0][i]+2]
    t=dict_test[0][fin_pred[0][i]+3]
    u=dict_test[0][fin_pred[0][i]+4]
    pattern=r+r"[^A-Za-z0-9]*"+s+r"[^A-Za-z0-9]*"+t+r"[^A-Za-z0-9]*"+u
    #print(r,s,t)
    p=re.compile(pattern)
    for m in p.finditer(test[0]):
        k=m.start()
    temp.append(k-1)
#temp.extend(len(test[0])-1)
temp.append(len(test[0]))
temp


# In[215]:


fin_pred[106]


# In[80]:


y_final_result=[]
for j in range(250):
    #ank=ank+1
    #print(ank)
    temp=[1]
    for i in range(len(fin_pred[j])):
        #print(j)
        r=dict_test[j][fin_pred[j][i]+1]
        s=dict_test[j][fin_pred[j][i]+2]
        t=dict_test[j][fin_pred[j][i]+3]
        u=dict_test[j][fin_pred[j][i]+4]
        pattern=r+r"[^A-Za-z0-9]*"+s+r"[^A-Za-z0-9]*"+t+r"[^A-Za-z0-9]*"+u
        #print(r,s,t)
        p=re.compile(pattern)
        for m in p.finditer(test[j]):
            k=m.start()
        temp.append(k-1)
#temp.extend(len(test[0])-1)
    temp.append(len(test[j]))
    y_final_result.append(temp)


# In[154]:


# In[590]:
y_final_result


# In[160]:


filenames


# In[81]:


segments=[]
for i in range(len(y_final_result)):
    temp=[str(i) for i in y_final_result[i]]
    segments.append('-'.join(temp))

segments


# In[162]:


len(segments)


# In[82]:


import pandas as pd


# In[83]:


ankit_result= pd.DataFrame(
    {'file_name':filenames,
     'segments':segments
    })
ankit_result.to_csv('test_result_ankit_lenovo.csv')

