from __future__ import print_function
import keras.callbacks as cb
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Flatten, Dense, Input, GlobalMaxPooling1D,Activation,Dropout, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.initializers import Constant
from matplotlib import pyplot as plt
import pickle
from keras import backend as K
K.clear_session()


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def plot_losses(lossesList, lables):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for losses, label in zip(lossesList, labels):
        ax.plot(losses, label=label)
    ax.set_title('Loss per batch')
    fig.show()

root_path = "./" #MyDrive/Aijava/
#The texts need preprocessing by adding spaces between keyword
#texts=['class KKBox  { int x = 3 ; }', 'class Exam  { Exam() { int x = 3 ; } }','class abc  { double x = 6.0 ; }']
errordict={}
texts=[]
errorcodes=[]

with open(root_path+'labels.txt', encoding="utf8") as f:
    readdata = f.readlines() 

for r in readdata:
    errordict.setdefault(r[0:r.index('\t')],r[r.index('\t')+1:]) # key - labels

    
x_train=[]
y_train=[]
x_test=[]
y_test=[]

with open(root_path+'73train.txt', encoding="utf8") as f:
    readdata = f.readlines()

for r in readdata:
    try:
        key=r[0:r.index('\t')]
        e=errordict[key]
        e=e.strip("\n")
        ee=e.split("\t") #error labels

        y_train.append(ee)
        x_train.append(r[r.index('\t')+1:])

    except KeyError:
        continue

with open(root_path+'73test.txt', encoding="utf8") as f:
    readdata = f.readlines()

for r in readdata:
    try:
        key=r[0:r.index('\t')]
        e=errordict[key]
        e=e.strip("\n")
        ee=e.split("\t") #error labels

        y_test.append(ee)
        x_test.append(r[r.index('\t')+1:])

    except KeyError:
        continue


labels_size=np.array(y_train).shape[1]

#
MAX_NUM_WORDS=4000
MAX_SEQUENCE_LENGTH=1000
EMBEDDING_DIM = 50
'''
print('Indexing word vectors.')
GLOVE_DIR=root_path
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
'''

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters=' ', lower=False, oov_token=('{', '}', '=', ';'))
#web
texts = x_train + x_test
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(x_train)
x_train_sequences = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)


test_sequences = tokenizer.texts_to_sequences(x_test)
x_test_sequences = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

y_test = np.array(y_test).astype(np.float)
y_train = np.array(y_train).astype(np.float)
r=np.sum(y_test,axis=0)
print([(i, p) for i, p in list(enumerate(r))])


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
'''
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
'''

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
#                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')
#outputs=49
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

#L1, L2 norm
#x = Dropout(0.2)(embedded_sequences)
x = Conv1D(16, 3)(embedded_sequences)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.5)(x)
x = MaxPooling1D(5)(x)

x = Conv1D(32, 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.5)(x)
x = MaxPooling1D(5)(x)

x = Conv1D(64, 3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.5)(x)
#x = GlobalMaxPooling1D()(x)
x = Flatten()(x)

x = Dense(20*labels_size)(x) #MLP
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.5)(x)
x = Dense(labels_size)(x)#softmax
x = BatchNormalization()(x)
x = Activation('sigmoid')(x)
model = Model(sequence_input, x)#mean_squared_error

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
#binary_crossentropy acc
'''
s=np.concatenate((data, errorcodes), axis=1)
np.random.shuffle(s)
data=s[::,0:MAX_SEQUENCE_LENGTH]
errorcodes=s[::,MAX_SEQUENCE_LENGTH:]
errorcodes=errorcodes.astype(np.float)##change float type
size=int (data.shape[0]/10)

x_train=data[0:size*8]#10
y_train=errorcodes[0:size*8]#10



x_val=data[size*8:size*9]#4
y_val=errorcodes[size*8:size*9]#4

x_test=data[size*9:]#5
y_test=errorcodes[size*9:]#5
'''

history = LossHistory()

print ('Training model...')

model.fit(x_train_sequences, y_train,
          batch_size=256,
          epochs=50, callbacks=[history],
          validation_split=0.2, verbose=2)
hloss1=history.losses
model.evaluate(x=x_test_sequences, y=y_test, batch_size=1024, verbose=1)

labels = [ 'RMSprop']
plot_losses([hloss1],labels)	
###save and load
##from keras.models import load_model
##model.save('MyDrive/AIJAVA/java_mode_mse_r6978_p9964.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
##del model  # deletes the existing model
##model2 = load_model('java_model579.h5')
#save model
model_to_save = {
	'classifier': model,
	'tokenizer': tokenizer,
	}  
with open(root_path+'model_CNN_unweight73_50', 'wb') as f:
    pickle.dump(model_to_save, f)



def totalSta(y) :
    tcnt=0 #total counts of correct 1 guess
    tcc=0 #total counts of real 1
    tpc=0 # total counts of 1-class guess 
    rcRate=0 #total counts of recall rate for 1
    pcRate = 0# total counts of precision rate for 1
    rcRate_0=0 #total counts of recall rate for 0
    pcRate_0 = 0 # total counts of precision rate for 0
    totalPre=0 #total preecision rate 
    for i in range(y_test.shape[0]): #for each test sample i
        cnt=0 #counts of correct 1 guess
        cnt_0=0 #counts of correct 0 guess
        cc=0 #counts of real 1
        pc=0 # counts of 1-class guess 
        for j in range (y_test[i].shape[0]): #for each label j
            if float(y_test[i][j])>0.5 : # real 1-class, (real 0-class: len(y[i]) - cc )
                cc=cc+1
            if (float(y_test[i][j])>0.5 and y[i][j]>0.5) : # correct 1 guess
                cnt=cnt+1
            if (float(y_test[i][j])<=0.5 and y[i][j]<=0.5) : # correct 0 guess
                cnt_0=cnt_0+1
            if y[i][j]>0.5 : # 1-class guess (0-class guess: len(y[i]) - pc)
                pc += 1

        tcnt +=cnt
        tcc  +=cc
        tpc += pc

        totalPre +=(cnt+cnt_0)/len(y[i]) #preecision rate of sample [i]
        
        if cc!= 0 :
            rcRate +=  cnt/cc
        if (len(y[i]) - cc) != 0:
            rcRate_0 +=  cnt_0/(len(y[i]) - cc)
            
    #        print ("recall rate:",cnt/cc)
    #    else:
    #    print (y_test[i])

        if pc!= 0 :
            pcRate +=  cnt/pc
        if (len(y[i]) - pc) != 0:
            pcRate_0 +=  cnt_0/(len(y[i]) - pc)
    #        print ("precision rate:",cnt/pc)
    #    else:
    #    print (y[i])
           
    #    print ("******")
    print ("1-class recall rate:",rcRate/y_test.shape[0]) # tcnt/tcc)
    print ("1-class precision rate:",pcRate/y_test.shape[0])  #tcnt/tpc)
    print ("0-class recall rate:",rcRate_0/y_test.shape[0]) # tcnt/tcc)
    print ("0-class precision rate:",pcRate_0/y_test.shape[0])  #tcnt/tpc)

    print ("total precision rate:",totalPre/y_test.shape[0])
    
 
def totalLabelSta(y_test, y,lab) :
    tt=0 #real true guess true
    tf=0
    ft=0
    ff=0
    for i in range(y_test.shape[0]): #for each test sample i
        j=lab
        if (float(y_test[i][j])>0.5 and y[i][j]>0.5): # real 1-class, (real 0-class: len(y[i]) - cc )
            tt+=1
        elif (float(y_test[i][j])>0.5 and y[i][j]<=0.5) : # correct 1 guess
            tf+=1
        elif (float(y_test[i][j])<=0.5 and y[i][j]>0.5) : # correct 0 guess
            ft+=1
        else :  # 1-class guess (0-class guess: len(y[i]) - pc)
            ff += 1

    if (tt+tf)==0:
        re=0
    else :
        re=tt/(tt+tf)
        
    if (tt+ft)>0:
        pre=tt/(tt+ft)
    else :
        pre=0
    
    if (re+pre)==0:
        F=0
    else :
        F=2*pre*re/(re+pre)

    print ("1-class recall rate for label :", lab, re) # tcnt/tcc)
    print ("1-class precision rate for label :",lab,  pre)  #tcnt/tpc)

    print ("F rate for label :", lab,F)
    return (re, pre, F)

tre=0
tpre=0
tF=0

target = y_train
codes = x_train_sequences

emptyLabel=[]
r=np.sum(target,axis=0)

for i, p in list(enumerate(r)):
  if p==0:  emptyLabel.append(i)   

y=model.predict(codes, batch_size=1024, verbose=1)

for j in range (target[i].shape[0]): #for each label j
  if not j in emptyLabel:
        (re, pre, F)=totalLabelSta(target, y,j)
        tre+=re
        tpre+=pre
        tF+=F
  else: print(j, " is absent")       
N=target[i].shape[0]-len(emptyLabel)     
print ("total 1-class recall rate:",tre/N) # tcnt/tcc)
print ("total 1-class precision rate:",tpre/N)  #tcnt/tpc)

print ("total F rate:",tF/N)



