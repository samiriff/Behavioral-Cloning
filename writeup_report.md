
# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* data_cleaner.py - for cleaning up the data recorded by the Simulator, and presents the data as a zip file that can be downloaded to the Udacity workspace from my laptop while create a model
* data_extractor.py - used by the Udacity workspace to extract the contents of the zipped file provided by the DataCleaner
* model.py - containing the script to generate batches of data from the file extracted by the DataExtractor, to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidia Convolutional neural network, as described in this [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The model definition can be found in the `build()` method of the `SelfDrivingModel` class in `model.py`, and is as shown in the table below:
| Layer | Description | Number of Trainable Parameters |
|--|--|--|
| Input | `160x320x3` RGB Image | 0 |
| Lambda | Normalizes the input image | 0 |
| Cropping2D | Crops away 50 and 20 pixels from the top and bottom of the image respectively | 0 |
| Lambda | Resizes the image to match the input size of `66x200`specified in the Nvidia model architecture diagram | 0 |
| Convolution `5x5`| `2x2` stride, same padding, outputs `33x100x24`, bias `24x1` | 1824 |
| RELU | | 0 |
| Dropout | Probability of dropping = 0.1 for training set | 0 |
| Convolution `5x5` | `2x2` stride, valid padding, outputs `15x48x36`, bias `36x1` | 21636 |
| RELU | | 0 |
| Dropout | Probability of dropping = 0.1 for training set | 0 |
| Convolution `5x5` | `1x1` stride, valid padding, outputs `11x44x48`, bias `48x1`| 43248 |
| Max pooling | `2x2` stride, valid padding, outputs `5x22x48` | 0 |
| RELU | | 0 |
| Dropout | Probability of dropping = 0.1 for training set | 0 |
| Convolution `3x3`| `1x1` stride, valid padding, outputs `3x20x64`, bias `64x1`| 27712 |
| RELU | | 0 |
| Dropout | Probability of dropping = 0.1 for training set | 0 |
| Convolution `3x3`| `1x1` stride, valid padding, outputs `1x18x64`, bias `64x1`| 36928 |
| RELU | | 0 |
| Dropout | Probability of dropping = 0.1 for training set | 0 |
| Flatten | outputs `1152` | 0 |
| Fully connected | outputs `100`, bias `100x1` | 115300 |
| Dropout | Probability of dropping = 0.25 for training set | 0 |
| Fully connected | outputs `50`, bias `50x1` | 5050 |
| Dropout | Probability of dropping = 0.25 for training set | 0 |
| Fully connected | outputs `10`, bias `10x1` | 510 |
| Fully connected | outputs `1`, bias `1x1` | 11 |


My model consists of 252,219 trainable parameters,

To introduce nonlinearity, the model includes RELU layers after every Convolutional Layer.

The following layers were added to the beginning of the model to pre-process data within the model itself :

 1. Lambda layer to normalize the data
 2. Cropping2D layer to crop the data
 3. Lambda layer to resize the data

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after every Convolutional and Fully connected layer (except for the last 2) in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

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
