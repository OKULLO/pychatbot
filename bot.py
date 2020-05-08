import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#import tensorflow as tf
#import tflearn
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np
import json
import pickle

with open('intents.json') as f:
    intents = json.load(f)
    print(intents['intents'])
    
    #group intents
    
    words =[]
    labels =[]
    docs_x =[]
    docs_y =[]
    
    for intent in intents["intents"]:
        for pattern in intents["patterns"]:
            #tokenize words
            wds = nltk.word_tokenize(pattern)
            words.extend(wds)
            docs_x.append(wds)
            docs_y.append(intent['tag'])
            if intent['tag'] not in labels:
                labels.append(intent['tag'])
                
words = [stemmer.stem(x.lower()) for x in words]
words = sorted(list(set(words)))
labels = sorted(labels)

#create bag of words for training

training = []
output =[]

out_empty = [0 for _  in range(len(labels))]

for x,docs in enumerate(docs_x):
    bag =[]
    
wds = [stemmer.stem(w) for w in docs]

for w in words:
    if w in wds:
        bag.append(1)
    else:
        bag.append(0)
out_row = out_empty[:]
out_row[(labels.index(docs_y[x]))]=1

training.append(bag)
output.append(out_row)

training = np.array(training)
output = np.array(output)


with open("data.pickle","wb") as f:
     pickle.dump((labels,words,training,output),f)
     
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        

    
#tf.reset_default_graph()
try:
  model.load_weights('data.hd5')

except:
  #fitting and saving the model 
    hist = model.fit(training,output, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h4', hist)   
   

   


#net = tflearn.input_data(shape =[None,len(training[0])])
#net = tflearn.fully_connected(net,0)
#net = tflearn.fully_connected(net,8)
#net = tflearn.fully_connected(net,len(output[0]), activation= 'sofmax')

#net = tflearn.regression(net)
#model = tflearn.DNN(net)
#model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
#model.save("model.tflearn")

                
                
                
