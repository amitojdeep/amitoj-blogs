---
layout: default
title: Traffic Sign Reco
---
# [](#header-2)Traffic Sign Recognition- using ensembling and CNNs on GTSRB

This project has my first few baby steps in the field of Deep Learning. It will definitely be a good read for someone very new to this domain as I will demonstrate how by creating a very simple deep learning model I managed to achieve a very high accuracy on GTSRB.When I was almost done with the awesome [Practical Deep Learning For Coders](fast.ai) tutorials, I was looking for a basic but cool deep learning project. Then I came across [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) which presents a rather old but interesting task. 

Our aim is to design a *Traffic Sign Recognition System* using a dataset of 40000+ labelled images across 42 categories. This model will then be used to make predictions on around 12000 unlabelled test images and the results will be compared on GTSRB benchmark. [`Keras`] (https://keras.io/) with [`Theano`] backend will be our tool of trade. I highly recommend `Keras` to beginners as well as those looking for quick implementation as it provides decent results without all the efforts involved in writing in say `TensorFlow`.



Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}


