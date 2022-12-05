# Emotion-Prediction-From-Text

<b>Results analysis</b>

● There are 57 sentences in the training data. The output has 5 classes representing one of the five emotions
namely surprise, anger, neutral, sadness and happiness.<br>
● The 140 unique stemmed words are the result of tokenization and stemming. Tokenization is the process
by which a big quantity of text is divided into smaller parts called tokens. We use the method
word_tokenize() to split a sentence into words.<br>
● The training is done with 70 neurons in the hidden layer and learning rate as 0.1. The training of the
network takes 3.62 seconds for 1000 epochs. The weights and bias for hidden layer and output layer are
updated and stored in synapses.json.<br>
● While testing the network with the sentence “Is everything going well” the prediction made is neutral.
