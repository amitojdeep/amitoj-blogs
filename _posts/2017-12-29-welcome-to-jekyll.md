---
layout: default
title: Traffic Sign Recognition
---
# [](#header-2)Traffic Sign Recognition- using ensembling and CNNs on GTSRB

This project has my first few baby steps in the field of Deep Learning. It will definitely be a good read for someone very new to this domain as I will demonstrate how by creating a very simple deep learning model I managed to achieve a very high accuracy on GTSRB.When I was almost done with the awesome [Practical Deep Learning For Coders](fast.ai) tutorials, I was looking for a basic but cool deep learning project. Then I came across [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) which presents a rather old but interesting task. 

Our aim is to design a *Traffic Sign Recognition System* using a dataset of 40000+ labelled images across 42 categories. This model will then be used to make predictions on around 12000 unlabelled test images and the results will be compared on GTSRB benchmark. [`Keras`](https://keras.io/) with [`Theano`](http://deeplearning.net/software/theano/) backend will be our tool of trade. I highly recommend `Keras` to beginners as well as those looking for quick implementation as it provides decent results without all the programming efforts involved in `TensorFlow`. I will be writing the code in `Jupyter` notebooks and recommend that you have [Anaconda](https://anaconda.org/) installed.
 
Without any further delays, let's jump straight into the work. You can refer to the [project repo](https://github.com/amitojdeep/traffic-sign-reco) for all the code as well as a trained model for quick implementations.
First of all we will separate the validation set from training set and convert the entire dataset to png images. You can write your own implementation or use the `road_sign_sep_val.ipynb` for separation and `road_sign_png_gen.ipynb` for conversion.


{% highlight python %}
path = 'data/train'
path_val = 'data/valid'
# 
train_datagen =  image.ImageDataGenerator(rotation_range=5, width_shift_range=0.08, shear_range=0.1,
                               height_shift_range=0.1, zoom_range=0.1)
test_datagen = image.ImageDataGenerator()

{% endhighlight %}


