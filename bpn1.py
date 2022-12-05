import json
import datetime
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

stemmer = LancasterStemmer()
training_data = []
training_data.append({"class":"anger", "sentence":"i feel agitated and annoyed"})
training_data.append({"class":"anger", "sentence":"I hate to live in this country"})
training_data.append({"class":"anger", "sentence":"I am extremely angry on you"})
training_data.append({"class":"anger", "sentence":"What a bad day!"})
training_data.append({"class":"anger", "sentence":"He is furious already"})
training_data.append({"class":"anger", "sentence":"Can you please shut up !!"})
training_data.append({"class":"anger", "sentence":"What a mad person"})
training_data.append({"class":"anger", "sentence":"I'm mad at you"})
training_data.append({"class":"anger", "sentence":"you are extremely annoying"})
training_data.append({"class":"anger", "sentence":"why are you soo annoying?"})
training_data.append({"class":"anger", "sentence":"I am extremely furious today"})


training_data.append({"class":"neutral", "sentence":"have a nice day"})
training_data.append({"class":"neutral", "sentence":"see you later"})
training_data.append({"class":"neutral", "sentence":"have a nice day"})
training_data.append({"class":"neutral", "sentence":"I like potato fries"})
training_data.append({"class":"neutral", "sentence":"how are you?"})
training_data.append({"class":"neutral", "sentence":"i am fine"})
training_data.append({"class":"neutral", "sentence":"I like be with you"})
training_data.append({"class":"neutral", "sentence":"talk to you soon"})
training_data.append({"class":"neutral", "sentence":"is everything going well ?"})
training_data.append({"class":"neutral", "sentence":"I'm reading a book"})
training_data.append({"class":"neutral", "sentence":"I like to study newspaper daily"})

training_data.append({"class":"sadness", "sentence":"i feel so jaded and bored"})
training_data.append({"class":"sadness", "sentence":"I am feeling soo sad today"})
training_data.append({"class":"sadness", "sentence":"i dont like do anything now"})
training_data.append({"class":"sadness", "sentence":"he seems depressed"})
training_data.append({"class":"sadness", "sentence":"I am not satisfied with the result "})
training_data.append({"class":"sadness", "sentence":"this is an not happy moment"})
training_data.append({"class":"sadness", "sentence":"this is soo depressing"})
training_data.append({"class":"sadness", "sentence":"i feel bored"})
training_data.append({"class":"sadness", "sentence":"this news is a sad one"})
training_data.append({"class":"sadness", "sentence":"You look sad"})
training_data.append({"class":"sadness", "sentence":"he is looking sad"})
training_data.append({"class":"sadness", "sentence":"this is not a happy news"})




training_data.append({"class":"happiness", "sentence":"I'm feeling pretty good right now"})
training_data.append({"class":"happiness", "sentence":"I'm in a very good mood"})
training_data.append({"class":"happiness", "sentence":"I feel great!"})
training_data.append({"class":"happiness", "sentence":"This is so awesome. I can't believe this happened."})
training_data.append({"class":"happiness", "sentence":"I got everything I ever wanted. I feel so blessed."})
training_data.append({"class":"happiness", "sentence":"I'm on cloud nine right now."})
training_data.append({"class":"happiness", "sentence":"I feel like I'm in paradise"})
training_data.append({"class":"happiness", "sentence":"I feel like a king."})
training_data.append({"class":"happiness", "sentence":"My brother was jumping around everywhere when he heard he got into Stanford"})
training_data.append({"class":"happiness", "sentence":"Today was the best day in my life"})
training_data.append({"class":"happiness", "sentence":"This place is nice. I love this place. "})


training_data.append({"class":"surprise", "sentence":"I was amazed really"})
training_data.append({"class":"surprise", "sentence":"Wow ! I never expected that"})
training_data.append({"class":"surprise", "sentence":"Ahh!!!"})
training_data.append({"class":"surprise", "sentence":"I won the match!!I cant believe that"})
training_data.append({"class":"surprise", "sentence":"You got a present for me !!"})
training_data.append({"class":"surprise", "sentence":"This is completely an unexpected thing"})
training_data.append({"class":"surprise", "sentence":"I was startled on seeing him"})
training_data.append({"class":"surprise", "sentence":"she was shocked to see her school friend next to her seat"})
training_data.append({"class":"surprise", "sentence":"They were astonished "})
training_data.append({"class":"surprise", "sentence":"I am really surprised"})
training_data.append({"class":"surprise", "sentence":"This is not something I expected"})
training_data.append({"class":"surprise", "sentence":"What a surprise !!"})




print ("%s sentences in training data" % len(training_data))

words = []
classes = []
documents = []
ignore_words = ['?','.']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

#print(documents);

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

training = []
output = []
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)
############################################333insert here



def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


def train(X, y, hidden_neurons=10, alpha=0.1, epochs=5000, dropout=False, dropout_percent=0.5):
    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
    hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1
    bias_hidden= [0]*hidden_neurons
    bias_out=[0]*5
    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
    plt.xlim(0, 10000)
    plt.ylim(0, 1)
    plt.xlabel('epochs')
    plt.ylabel('error')
    for j in iter(range(epochs + 1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0)+bias_hidden)

        if (dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                        1.0 / (1 - dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1)+bias_out)

        # how much did we miss the target value?
        layer_2_error = y - layer_2


        if (j % 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                break
        x_coord = j
        y_coord = np.mean(np.abs(layer_2_error))
        plt.plot(x_coord, y_coord,'o','black')



            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        bias_out1= layer_2_delta * alpha
        bias_hidden1 = layer_1_delta * alpha
        bias_out+=bias_out1
        bias_hidden+=bias_hidden1
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))#delta * zin
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))#delta * xin

        if (j > 0):
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 +=( alpha * synapse_1_weight_update)
        synapse_0 += ( alpha * synapse_0_weight_update)



        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(),
               'synapse1': synapse_1.tolist(),
               'bias_out':bias_out.tolist(),
               'bias_hidden':bias_hidden.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=6,sort_keys=True)
    print("saved synapses to:", synapse_file)

####################main function
X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=70, alpha=0.1, epochs=1000, dropout=False,dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")

# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])
    bias_out  = np.asarray(synapse['bias_out'])
    bias_hidden =   np.asarray(synapse['bias_hidden'])

#####for testing
def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    plt.show()
    return return_results

classify("I feel great",True)
