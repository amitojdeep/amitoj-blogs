---
layout: default
title: Deep Stock Predictions
---
# [](#header-2) Multimodal & Multitask Deep Learning- Predicting Stock Prices

Stock price prediction has always been a lucrative area in Finance because it deals with making actual money! Investment firms commonly use statistical and AI based models to predict prices of various assets and keep them as trade secrets. [Qplum](https://www.qplum.co/) is a startup that claims to be relying solely on AI and Data Science for automated investments. There are many research papers in this area as well, but no one reveals their *secret sauce*, but I would like to shed some light with my model.

I tried my approach inspired by [Alex Honchar's](https://medium.com/@alexrachnog?source=post_header_lockup) expository work, during a project type course last semester. I used DJIA stock price data and RedditFinance(https://www.reddit.com/r/finance/) news for last 8 years. I built the model using the trusty `Keras` in a `Jupyter` notebook and trained on my GTX 960m laptop. You can jump straightaway to the [notebook](https://github.com/amitojdeep/deep-stock-preds/blob/master/Stock%20Prediction.ipynb) and start experimenting or read here for an overview. 

## Preprocessing

I have downloaded the price data from Yahoo Finance as a CSV file and it is ready for use. The scraped news headings are converted to embeddings using [Google's word2vec](https://www.tensorflow.org/tutorials/word2vec) model. It is a Vector Space Model
(VSM) that maps similar words to nearby points. So, similar words are represented by numbers that are closer to each other and this provides a semantic value to the representation.The process of embedding generation can be diagramatically understood as:

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/emb.JPG">

Here I have used word2vec model with 100 dimensions to keep the things faster and lighter. One may use a  bigger model with 300 dimensions, hoping to capture context even better.


## Implementation

First we will load the data to variables. Each of them contains the following data;

{% highlight python %}
X_train # Time series data
X_train_text # word2vec decoded text data
Y_train # Labels for voaltility
Y_train2 # Labels for classification (movement direction)
{% endhighlight}

### Model Design
Next, the model is defined using `Functional API` of Keras.
The model is made by concatenating two LSTMs, one (lstm1) takes OHLCV tuple as input and the other (lstm2) takes vector representation of news data. The two outputs of the model are volatility (standard deviation - x1) and price movement direction (up or down - x2)

{% highlight python %}
main_input = Input(shape=(30, 5), name='ts_input')
text_input = Input(shape=(30, 100), name='text_input')
lstm1 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(main_input)
lstm1 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(lstm1)
lstm1 = Flatten()(lstm1)
lstm2 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(text_input)
lstm2 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(lstm2)
lstm2 = Flatten()(lstm2)


lstms = concatenate([lstm1, lstm2])


x1 = Dense(64)(lstms)
x1 = LeakyReLU()(x1)
x1 = Dense(1, activation = 'linear', name='regression')(x1)

x2 = Dense(64)(lstms)
x2 = LeakyReLU()(x2)
x2 = Dropout(0.9)(x2)
x2 = Dense(1, activation = 'sigmoid', name = 'class')(x2)

final_model = Model(inputs=[main_input, text_input], 
              outputs=[x1, x2])
{% endhighlight}

## Visualisation

I have made this simple diagram using [draw.io](https://www.draw.io/) to make the model's design clearer.
 
<img src="https://github.com/amitojdeep/deep-stock-preds/raw/master/Multimodal.jpg">


