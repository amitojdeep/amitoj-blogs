---
layout: default
title: Semantic Similarity
---
# [](#header-2)Siamese LSTM for Semantic Similarity Analysis

This project was done as a part of a larger project where my team designed a Predicitive Typing System using statistical techniques and it was compared with predicted words generated using Semantic Similarity. I have implemented Semantic Similarity analyzer using `Keras` on [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset. It is application of [*Learning Sentence Similarity with Siamese Recurrent Architectures*](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023) paper by J Mueller.

The following flowchart describes the system that I have built. I will describe it stepwise with code snippets in this post.

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/sem-sys.png">

## Dataset 

Quora Question Pairs Dataset which is publically available on Kaggle has been used to train the Siamese LSTM Model. It has 400,000 samples of potential question duplicate pairs. Each sample has two questions along with ground truth about their similarity(0 - dissimilar, 1- similar). These are split into test and training dataset. Further 40,000 pairs have been separated from training dataset for validation.

## Preprocessing

Raw text needs to be converted to list of word indices. A helper function takes a string as input and outputs a list of words from it after some preprocessing like accommodating specific signs as described in this piece of code.

{% highlight python%}
def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text
{% endhighlight%}

## Embeddings

Google’s word2vec has been used to turn words to embeddings. It is a Vector Space Model (VSM) that maps similar words to nearby points. So, similar words are represented by numbers that are closer to each other and this provides the representation a semantic value to it. This is more powerful than representing words as unique ids with no relation between words with closer ids as helps in training models that leverage the semantics of words to create context. Vocabulary dictionary stores word to index mapping and Inverse Vocabulary dictionary stores index to word mapping.

The embedding generation can be understood with the example below:

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/sem-emb.png" width="400">

The embeddings of each word in a given sentence is the input that goes to the model. Embedding list is created for each query pair and it is padded with zeroes to accommodate the sentence with longest representation and make length of all embedding lists equal. I have pickled the embeddings separately and loaded them for final deployment directly as geneerating embeddings each time is very slow.

##Siamese LSTM - A variant of Manhattan LSTM

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/sem-diag.png" width="400">

Manhattan LSTM models has two networks LSTMleft and LSTMright which process one of the sentences in a given pair independently. Siamese LSTM, a version of Manhattan LSTM where both  LSTMleft and LSTMright  have same tied weights such that  LSTMleft = LSTMright. Such a model is useful for tasks like duplicate query detection and query ranking. Here, duplicate detection task is performed to find if two queries and duplicates or not. Similar model can be trained for query ranking using hit data for a given query and it’s matching results as a proxy for similarity.

The model uses an LSTM which reads word-vector representations of two queries and represents it in final hidden state. Similarity between the is used generated from the equation below:

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/sem-eq.png" width="300">

This equation can be written in the form of a python function which I will be using as a lambda in `Keras` layer directly. 

{% highlight python %}
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
{% endhighlight %}

The model is coded using `Functional API` of Keras which allows for interesting non-linear designs.
{% highlight python %}
# Model variables
    n_hidden = 50
    gradient_clipping_norm = 1.25
    batch_size = 128
    n_epoch = 10
    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length,
                                trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                            output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    # Pack it all up into a model
    malstm = Model([left_input, right_input], [malstm_distance])
{% endhighlight %}

The model has two visible input layers, one for each side of the MaLSTM. These are followed by two embedding layers on each size and LSTM model of Keras Functional API. Training is run for 10 epochs due because the gain in accuracy per epoch begins to fall thereafter. Also training this model very resource intensive and it took over 11hrs on a machine with a 960m GPU. For deployment purposes the system can be trained for more epochs until the validation accuracy begin to stabilize (to avoid overfitting). The model is trained and the weights are stored for prediction tasks.

The figure below shows Model Loss (MSE) and Accuracy on y-axis as a function of number of epochs on the x-axis. 

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/sem-his.png" width="300">

## Results

### Mean Squared Error (MSE)

The mean squared error is the same as loss measure of the model. Here, the lower two plots are for MSE. The best value of training MSE obtained is **0.1395** and it is found to be decreasing at a decreasing rate throughout the training process of the model. Whereas the validation MSE (which is equivalent to test MSE) is found to be **0.1426** at the end of ten epochs and follows same behavior as discussed above.

It is worth noting that  MSE for training set as initially worse off than that for validation set as the model is under fitted then. It approaches Validation MSE and then crosses it slightly indicating that the model is performing equally well on training and validation. Our model seems to be well-fitted as these values are very close.

### Accuracy 

Validation accuracy which indicates the performance of model on unlabelled data is found to be **80.35%** at the end of 10 epochs. It increases at a decreasing rate as the model learns weights better after each epochs. Training accuracy reaches **80.88%** and follows a similar trend. There is not much disparity between these two indicating a well fitted model.

Training the model for a few more epochs will probably give some improvements in accuracy as the accuracy is still rising after 10 epochs instead of levelling off.

###Confusion Matrix

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/Figure_2.png" width="300">

Confusion matrix plotted above shows the assignment of classes to sample pairs of questions by the model. Most of the predictions are concentrated in the diagonal classes indicating a good classification. Non duplicate questions is clearly the dominant class in the model.

## Conclusion and Future Work

Semantic Similarity classifier based on Siamese LSTM model has given sufficiently good results on the Quora Question Pairs Dataset giving an accuracy of 80.35% indicating its suitability for the task. This model can be trained on task specific datasets for application in various domains as a part of future research. Also hyperparameter optimization & changing optimizer can be done. Upcoming techniques like Reinforcement Learning, Transfer Learning,  Augmentation using synonyms can be accommodated in the model for studying their impact on the model.









