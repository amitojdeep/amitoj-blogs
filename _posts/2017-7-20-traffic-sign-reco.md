---
layout: default
title: Traffic Sign Recognition
---
# [](#header-2)Traffic Sign Recognition- using ensembling and CNNs on GTSRB

This project has my first few baby steps in the field of Deep Learning. It will definitely be a good read for someone very new to this domain as I will demonstrate how by creating a very simple deep learning model I managed to achieve a very high accuracy on GTSRB.When I was almost done with the awesome [Practical Deep Learning For Coders](fast.ai) tutorials, I was looking for a basic but cool deep learning project. Then I came across [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) which presents a rather old but interesting task. 

Our aim is to design a *Traffic Sign Recognition System* using a dataset of 40000+ labelled images across 42 categories. This model will then be used to make predictions on around 12000 unlabelled test images and the results will be compared on GTSRB benchmark. [`Keras`](https://keras.io/) with [`Theano`](http://deeplearning.net/software/theano/) backend will be our tool of trade. I highly recommend `Keras` to beginners as well as those looking for quick implementation as it provides decent results without all the programming efforts involved in `TensorFlow`. I will be writing the code in `Jupyter` notebooks and recommend that you have [Anaconda](https://anaconda.org/) installed.
 
Without any further delays, let's jump straight into the work. You can refer to the [project repo](https://github.com/amitojdeep/traffic-sign-reco) for all the code as well as a trained model for quick implementations.
First of all we will separate the validation set from training set and convert the entire dataset to png images. You can write your own implementation or use the `road_sign_sep_val.ipynb` for separation and `road_sign_png_gen.ipynb` for conversion. I will retain the directory structures as in GTSRB dataset because we need it for image generators of `Keras`.

Setting up the data generators,

{% highlight python %}
path = 'data/train'
path_val = 'data/valid'

train_datagen =  image.ImageDataGenerator(rotation_range=5, width_shift_range=0.08, shear_range=0.1,
                               height_shift_range=0.1, zoom_range=0.1)
test_datagen = image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(50, 50),
        batch_size=48,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        path_val,
        target_size=(50, 50),
        batch_size=48,
        class_mode='categorical')

{% endhighlight %}

Note that I am using the image size `50*50` as just it seems to be sufficiently appropriate for complexity of traffic sign without making convolutional layers too heavy. Later I will demonstrate how using `80*80` image size increased training times tremendously without any significant gains in accuracy. I have used 48 batch size which was a decent number considering I had a GTX 960m with 4GB memory at my disposal. Any smaller and training will be too slow, any larger and you risk running out of memory. So decide according to your GPU, or let it be 48 only.

Now let's jump to the model that we will design using Sequential API of `Keras`.

{% highlight python %}
def get_model_bn():
    model = Sequential([
        Lambda(get_x,input_shape=(3,50,50)),
        Convolution2D(32,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,3,3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        MaxPooling2D(),
        Convolution2D(128,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(128,3,3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        #Dense(4096, activation='relu'),
        #Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(43, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
{% endhighlight %}

The beauty of Keras is that you can understand the model architecture by just looking at a small piece of code. This model is a miniature of renowned `vgg16` model with additional [batch normalization](https://keras.io/layers/normalization/) layers to improve accuracy and reduce training time. Applying the actual `vgg` model will be an overkill considering it's larger image size and more depth. I designed another model with same number of convolutional and dense layers as `vgg16` and there was negligible gain in accuracy. Adam optimizer is used to control the learning rate.

Now, I will be training the model for 25 epochs. I will store the model weights after every epoch as a safegaurd against any system failure. You can also use *checkpoints* of Keras for the same.

{% highlight python %}
modelA = get_model_bn()
for i in range(1,25):
    modelA.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.n,
        nb_epoch=1,
        validation_data=validation_generator,
        nb_val_samples=validation_generator.n)
    modelA.save_weights("data/tr50A" + str(i) + ".h5")
{% endhighlight %}

Let's see how the model trains,
<img src="https://github.com/amitojdeep/amitoj-blogs/blob/master/assets/training-progress.png?raw=true" width="554" height="574" >







