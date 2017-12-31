---
layout: default
title: Ensemble of CNNs for Traffic Sign Recognition
---
# [](#header-2)Traffic Sign Recognition- Ensemble of CNNs

This project was my first few baby steps in the field of Deep Learning. It will definitely be a good read for someone very new to this domain as I will demonstrate how by creating a very simple deep learning model I managed to achieve a very high accuracy on GTSRB.When I was almost done with the awesome [Practical Deep Learning For Coders](fast.ai) tutorials, I was looking for a basic but cool deep learning project. Then I came across [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) which presents a rather old but interesting task. 

Our aim is to design a *Traffic Sign Recognition System* using a dataset of 40000+ labelled images across 42 categories. This model will then be used to make predictions on around 12000 unlabelled test images and the results will be compared on GTSRB benchmark. [`Keras`](https://keras.io/) with [`Theano`](http://deeplearning.net/software/theano/) backend will be our tool of trade. I highly recommend `Keras` to beginners as well as those looking for quick implementation as it provides decent results without all the programming efforts involved in `TensorFlow`. I will be writing the code in `Jupyter` notebooks and recommend that you have [Anaconda](https://anaconda.org/) installed.

## Implementation
Without any further delays, let's jump straight into the work. You can refer to the [project repo](https://github.com/amitojdeep/traffic-sign-reco) for all the code as well as a trained model for quick implementations.
First of all we will separate the validation set from training set and convert the entire dataset to png images. You can write your own implementation or use the `road_sign_sep_val.ipynb` for separation and `road_sign_png_gen.ipynb` for conversion. I will retain the directory structures as in GTSRB dataset because we need it for image generators of `Keras`.
I have used *data augmentation* to introduce random transformations to the training images.
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

## Training

Now, I will be training the model for 25 epochs. I will store the model weights after every epoch as a safeguard against any system failure. You can also use *checkpoints* of Keras for the same.

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
<img src="https://github.com/amitojdeep/amitoj-blogs/blob/master/assets/training-progress.png?raw=true" width="692" height="717" >

The training goes on pretty much ideally and it stabilizes to accuracy of **99.5%** on training set and about **99.9%** on test set. It took me around 3 hours to train one instance of the model.

Now, I will generate predictions on the test set and upload them to GTSRB.

{% highlight python %}
model.load_weights('data/tr50A25.h5')

path_test = 'data/test_full/'
test_datagen = image.ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        path_test,
        target_size=(50, 50),
        batch_size=16,
        class_mode=None,
        shuffle = False)
pred = model.predict_generator(test_generator,test_generator.n)
{% endhighlight %}

This prediction will come out as confidence levels between 0 to 1 assigned to each class, where the class with highest number is the predicted class.
Let's take an example,

`[[  4.6133e-07   3.2123e-07   1.0266e-06   8.9983e-07   8.5720e-06   5.1291e-06   9.1533e-07
    2.2246e-06   1.1797e-07   1.4529e-06   9.9455e-01   5.8414e-04   7.2590e-05   1.7934e-04
    3.8776e-07   2.7640e-07   7.7135e-07   2.0421e-06   1.7570e-04   1.2043e-07   1.8051e-04
    9.3152e-06   3.9429e-07   2.2013e-05   5.8755e-08   4.0380e-04   9.3221e-08   2.0156e-07
    4.6435e-07   6.1648e-08   5.1891e-07   4.1398e-05   1.0640e-06   1.6840e-03   2.1620e-06
    3.9760e-07   7.7743e-06   4.5777e-07   1.2642e-03   1.2395e-05   1.5695e-05   1.1570e-06
    7.6219e-04]]`

In this example *class 10* is predicted for the following, rather blurry image.

<img src = "https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/00101.png">

Here is an actual image from *class 10*,

<img src = "https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/00013_00025.png">


## Results

So, you can see that the model is working reasonably well even for a blurry, low resolution image of a bit complexed traffic sign.
You can see the results of my approach as the bottom most entry of the team *CSIO_Trainees* carrying the label *CNN with data augmentation*. **98.60%**, the number is rather impressive but quite below the state of the art that's 99.99%. 

<img src = "https://github.com/amitojdeep/amitoj-blogs/raw/master/assets/res.png">

## Improvements - Ensembling

Now, I trained 5 models with the same architecture that we have discussed above. Voila! The accuracy improve to **99.26%** percent, bringing it to top 10. But why did it go up? Each of the models is trained separately, with different data points occuring at different points of time during training and with separate augmentations applied by the generator. As a result, their errors are not expected to overlap, resulting in a better accuracy when we take the *most commonly predicted class*

I tried a few more things and achieved an accuracy of **99.38%** using an ensemble of 8 models, 3 of them were trained on `80*80` images and 5 were the one's I used priorly. Outputs for a few more approaches used by me are also in the table. Ensembling is helpful, but not beyond 7-8 models.

## Further tracks of work

A few more approaches that can be tried to improve accuracy,

* Pseudo Labelling: Labels can be generated for test images using initial predictions of the model. These can be used to further finetune the model weights. It is a form of *Semi Supervised Learning* and has given promising results in many fields.

* Image Improvement: Images can be passed through filters that detect shapes and traffic signs can be separated by removing all the noise in surroundings. This must be done for both test and training images.

* Transfer Learning: Weights of an already trained model like `vgg16` can be by finetuning the last few layer's for our specific task. This will save the training time and provide the benefits of extensively trained model. 

Hope you liked my first blog post! Do share it among your peers, especially those who are on the fence of taking a dive into the beautiful world of deep learning.










