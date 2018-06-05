# **Behavioral Cloning** 



[//]: # (Image References)

[image0]: ./examples/cover.png
[image1]: ./examples/cnn-architecture-624x890.png


This is a brief writeup report of Self-Driving Car Engineer P3.


<img src="./examples/cover.png" width="50%" height="50%" />

---

**Steps Of This Project**


* Use the [Simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model autonomously drives around track one without leaving the road




## Files and Code


#### 1. The project includes the following files:


* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 


#### 2. Functional code

Using the **simulator** and **drive.py** file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Model code

The **model.py** file contains the code for training and saving the convolution neural network. Also, the file shows the pipeline, which is used for training and validating the model.


## Data Collection

Before training, the human behavior data on **Track 1** needs to be collected. 

Here are data collection strategies:

> Counter-Clockwise ( the default direction of Track 1 )

* **Center lane driving** × 2 laps
* **Recovery driving from the sides** × 1 lap
* **Focusing on driving smoothly around curves** × 1 lap

> Clockwise

* **Center lane driving smoothly** × 2 laps

After that, the collected data includes a 'driving_log.csv' file and a bunch of camara images in shape of (160,320,3). 

All those are approach 600MB.

## Model Architecture

The model of project was built by **Keras** and referenced some kind of common architectures.

> First Try ( flatten only )

At first, for testing the pipeline, the model consisted of a flatten layer only, which was, not surprisingly, too simple to have a good result. 「model.py lines 71-74」

> Try Again ( LeNet-5 )

The Old friend LeNet is a more powerful network architecture. It helped the simulator vehicle pass through the straight lane smoothly, but failed at curves. 「model.py lines 77-87」

*'Hey, look, there's a car in the river, what happened?'*

> Final Model ( NVIDIA's CNN )

The architecture published by the autonomous vehicle team at NVIDIA is a even more powerful network. It has 5 convolution layers (with 3x3 or 5x5 filter sizes and depths between 24 and 64) and 3 full-connected layers.

<img src="./examples/cnn-architecture-624x890.png" width="50%" height="50%" />

Some tricks were used on the basis of the NVIDIA's architecture:

* Normalization ( nomalize and 0 mean )

`model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))`

* Cropping ( crop useless pixels of image, the sky, trees, hood, etc)

`model.add(Cropping2D(cropping = ((70, 20), (0, 0))))`

* Dropout layer ( reduce overfitting )

`model.add(Dropout(0.2))`

The final model makes the simulator vehicle be able to drive autonomously around the Track 1 without leaving the road. 「model.py lines 90-103」

## Training Strategy


#### 1. Training data process

The vehicle has three camaras ( center, left and right ), and the side camara images carries two benifits rather than the center camara images only:

* more training data ( 3 times as much )
* help teach the network how to steer back to the center when drifting off

By taking the actual steering measurement and adding a small correction `0.2` factor to it, those side camara images can be appended to the training data set appropriately. 「model.py lines 28-40」

Simultaneously, data augmentation by flipping the images and steering measurements is also a common trick to expand training data set and help the model generalize better. In the project, the way of augmentation is flipping images by `cv2.flip()` function and taking the opposite sign of the corresponding steering measurements. 「model.py lines 46-51」

#### 2. Training data process


#### 3. Training data process
 2r2rqwer2


The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

## Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

```
Train on 51657 samples, validate on 12915 samples
physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
Epoch 1/7
51657/51657 [==============================] - 83s - loss: 0.0200 - val_loss: 0.0240
Epoch 2/7
51657/51657 [==============================] - 72s - loss: 0.0112 - val_loss: 0.0128
Epoch 3/7
51657/51657 [==============================] - 72s - loss: 0.0061 - val_loss: 0.0096
Epoch 4/7
51657/51657 [==============================] - 72s - loss: 0.0054 - val_loss: 0.0095
Epoch 5/7
51657/51657 [==============================] - 72s - loss: 0.0051 - val_loss: 0.0096
Epoch 6/7
51657/51657 [==============================] - 72s - loss: 0.0048 - val_loss: 0.0096
Epoch 7/7
51657/51657 [==============================] - 72s - loss: 0.0046 - val_loss: 0.0094
```

```
python drive.py model.h5 run1
python video.py run1
100%|████████████████| 15860/15860 [01:20<00:00, 198.19it/s]
```