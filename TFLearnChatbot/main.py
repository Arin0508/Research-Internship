import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import pickle
import json

with open('intents.json') as file:
    data = json.load(file)

#We don't want our model to train again and again for each prediction it makes ...
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except :
    words = []
    labels = []
    docs_x = [] #List of all the different Patterns
    docs_y = [] #Tag for those patterns in doc_x (What intent it conveys)

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern) 	#Tokenizing the sentence
            words.extend(wrds) 					#Putting all the words into the array
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    #Stemming the Words
    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #Removing ending question marks
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #Creating Training and Test Output
    #Neural Networking works only on numbers, so we need to change the words
    #into numbers (Using one hot encoding)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = [] 	#Making a bag of words

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    #Changing lists into Array
    training = numpy.array(training)
    output = numpy.array(output)

    #Writing the required data to a file
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

#Input Layer
net = tflearn.input_data(shape=[None, len(training[0])]) #Size of the input we are expecting
net = tflearn.fully_connected(net, 10) #Two hidden layers of 8 Neurons
net = tflearn.fully_connected(net, 10)

#Output Layer 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #Softmax will give a probability to each of the output neuron
net = tflearn.regression(net)

#For Training the Model.. DNN is a type of a neural network
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=5000, batch_size=8, show_metric=True) #n_epoch is basically the number of times the model will see the same data
model.save("model.tflearn")                                               #Rest arg are basically for showcasing  

#We don't want our model to train again and again for each prediction it makes ...
# try:
#     model.load("model.tflearn")
# except:
            
#Taking the input and converting into the bag of words
def bag_of_words(s, words): 
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    context = ""
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0] #This returns probabiltiy for each output neuron
        results_index = numpy.argmax(results)   #Returns the index of the greatest probablitiy
        tag = labels[results_index]

        #Checking for the probabiltiy 
        flag = 0;
        if results[results_index] > 0.7 :
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    if 'context_set' in tg:
                        context = tg['context_set']
                    if not 'context_filter' in tg:
                        responses = tg['responses']
                    if('context_filter' in tg and tg['context_filter'] == context):
                        responses = tg['responses']
                        break 
                    elif('context_filter' in tg and context == "") :
                        flag = 1
                    # responses = tg['responses']
            if flag==0 :   
                print( "Chip : " + random.choice(responses))
            else :   
                print( "Chip : What course you want to know about ?") 
        else:
            print("Chip : I didn't get that. Try another question !")

chat()