---
layout: default
title: Deep Stock Predictions
---
# [](#header-2) Multimodal & Multitask Deep Learning- Predicting Stock Prices

Stock price prediction has always been a lucrative area in Finance because it deals with making actual money! Investment firms commonly use statistical and AI based models to predict prices of various assets and keep them as trade secrets. [Qplum](https://www.qplum.co/) is a startup that claims to be relying solely on AI and Data Science for automated investments. There are many research papers in this area as well, but no one reveals their *secret sauce*, but I would like to shed some light with my model.

I tried my approach inspired by [Alex Honchar's](https://medium.com/@alexrachnog?source=post_header_lockup) expository work, during a project type course last semester. I used DJIA stock price data and [RedditFinance](https://www.reddit.com/r/finance/) news for last 8 years. I built the model using the trusty `Keras` in a `Jupyter` notebook and trained on my GTX 960m laptop. You can jump straightaway to the [notebook](https://github.com/amitojdeep/deep-stock-preds/blob/master/Stock%20Prediction.ipynb) and start experimenting or read here for an overview. 

## Preprocessing

I have downloaded the price data from Yahoo Finance as a CSV file and it is ready for use. The scraped news headings are converted to embeddings using [Google's word2vec](https://www.tensorflow.org/tutorials/word2vec) model. It is a Vector Space Model
(VSM) that maps similar words to nearby points. So, similar words are represented by numbers that are closer to each other and this provides a semantic value to the representation.The process of embedding generation can be diagramatically understood as:

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/emb.JPG" width="400" align ="middle">

Here I have used word2vec model with 100 dimensions to keep the things faster and lighter. One may use a  bigger model with 300 dimensions, hoping to capture context even better.


## Implementation

First we will load the data to variables. Each of them contains the following data;

{% highlight python %}
X_train # Time series data
X_train_text # word2vec decoded text data
Y_train # Labels for voaltility
Y_train2 # Labels for classification (movement direction)
{% endhighlight %}

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
{% endhighlight %}

## Visualisation

I have made this simple diagram using [draw.io](https://www.draw.io/) to make the model's design clearer.
 
<img src="https://github.com/amitojdeep/deep-stock-preds/raw/master/Multimodal.jpg" width="400" align="middle">

Inputs from stock price data and news vectors are fed into two separate LSTMs. These are flattened to same number of dimensions and are concatenated to form a single layer of 600 dimensions. This layer is further used as an input to two dense layers, one which predicts volatility and the other price movement. MSE is used to measure volatility loss because it is a regression problem whereas price movement loss is measured using binary cross entropy. Also the dense layers use leaky relu activation function and a dropout of 0.5 to prevent overfitting.

Another neat and easy way to visualize any keras model is to use `pyplot` and `plot_model` utility of Keras as described in *Visualize.ipynb*.

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/vis.JPG" width="400">

ts_input is the time series of stock market price and text_input is the news headline data of a
given date. The stock price input goes to two LSTM units sequentially before getting flattened
for concatenation. The news headlines are also passed through two sequential LSTM block and
then the output is flattened. Concatenation unit joins together the activations from price and
news LSTM units.

The concatenated layer goes as input to one Dense layer followed by a Leaky ReLu activation layer and a dropout layer. This gives the predicted direction of movement of stock price movement. Here 0 indicates a prediction of downward price movement and 1 indicates a
predicted upwards price movement.
This concatenated layer is also fed to another dense layer with a Leaky ReLu activation which predicts volatility of the price. It must be noted that this volatility is a normalized continuous value where 0 means no volatility and 1 denotes the volatility of the date in the dataset with most volatility. For other days the volatility predicted is the fraction of maximum volatility.

## Training

The model is compiled using Mean Square Error loss for volatility and binary cross entropy loss for the predicted price movement direction. A snapshot of the training process is as shown below. The model was trained for 100 epochs and the weights with the least total loss were used for making predictions.

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/stock-train.JPG">

Model loss measures the difference between true and predicted values of the price and volatility. The price movement is a 0-1 classification problem and hence binary cross entropy is used a measure of loss. Whereas volatility is a continuous measure and
thus mean squared error is used as loss function. Initially both test and training loss are quite high as the model has random
weights and they decrease rapidly as epochs go on. After around 20 epochs the test and training loss converge and settle indicating a well-fitted model.
The results of minimizing combined loss are as shown below.

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/train-his.JPG" width="400">

## Results

Model achieves an accuracy of **74.93%** on this test set which is better than any technique that is based solely on price data. It is in line with sophisticated models that use only news data despite its simplistic design and treatment of stock market as a whole rather than taking individual stocks.

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/stock-conf.JPG" width="400">

The confusion matrix above shows the relationship between true and predicted price movement for each of the 347 trading days of the test set. The model is slightly better at predicting positive price movements as compared to negative one’s but has a very balanced confusion matrix overall. This indicates the robustness and predictive power of the model.

<img src="https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/pred-vol.JPG" width="400">

Above is a plot comparing predicted and actual volatility for the test set. It is clear that the model is able to capture most of the dependencies and big jumps quite well giving a good estimate of market risk on any given day.

## Conclusion and thoughts

Multimodal and Multitask Deep Learning holds great potential for stock price predictions and has been shown to achieve better results than techniques relying solely on price data by supplementing it with news data.

It is worth noting that the model uses news information for stock market as a whole instead of targeted news to make trading decisions for individual stocks which restricts its performance and at the same time makes the implementation simplistic. Also the high frequency price data and live news feed can be used to make a real time High Frequency Trading (HFT) system using the architecture of proposed model.

To sum it up, this work provides an exposure into the application of Multimodal and Multitask learning to financial time series data. The positive results obtained validate the credibility of model for more complex tasks. In future work trading strategies using a target set of companies can be applied to observe the model’s performance for a real portfolio.



