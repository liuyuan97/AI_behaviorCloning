# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
*

[//]: # (Image References)

[image1]: ./model.png "CNN model"
[image2]: ./md_pics/flipped/center_2017_12_09_22_40_42_257.jpg "center Image"

[image3]: ./md_pics/center_2017_12_09_22_40_32_336.jpg "Right Recovery Image 1"
[image4]: ./md_pics/center_2017_12_09_22_40_33_532.jpg "Right Recovery Image 2"
[image5]: ./md_pics/center_2017_12_09_22_40_35_913.jpg "Right Recovery Image 3"
[image6]: ./md_pics/center_2017_12_09_22_40_38_874.jpg "Right Recovery Image 4"

[image7]: ./md_pics/center_2017_12_09_22_44_38_970.jpg "Left Recovery Image 1"
[image8]: ./md_pics/center_2017_12_09_22_44_48_167.jpg "Left Recovery Image 2"
[image9]: ./md_pics/center_2017_12_09_22_44_49_721.jpg "Left Recovery Image 3"
[image10]: ./md_pics/center_2017_12_09_22_44_50_475.jpg "Left Recovery Image 4"

[image11]: .//md_pics/flipped/left_2017_12_09_22_40_42_257.jpg "left Image"
[image12]: .//md_pics/flipped/center_2017_12_09_22_40_42_257.jpg "center Image"
[image13]: .//md_pics/flipped/right_2017_12_09_22_40_42_257.jpg "right Image"

[image14]: .//md_pics/flipped/left_2017_12_09_22_40_42_257_fliped.jpg "left Image"
[image15]: .//md_pics/flipped/center_2017_12_09_22_40_42_257_fliped.jpg "center Image"
[image16]: .//md_pics/flipped/right_2017_12_09_22_40_42_257_fliped.jpg "right Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* behavioral_cloning.md summarizing the results
* run1.mp4 containing recored video
*

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 91 (clone.py lines 83-88) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 84). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clone.py lines 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and recovering from the left and right sides of the road. I also augmented data by flipping the data, and reversing the steering.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a neural network which could drive the car autonomously using the camera pictures by cloning the human drive behavior.  The overall architecture is based on the Convolutional Neural Network (CNN)

My first step was to use a convolution neural network model similar to the LeNet, which is used to classify the traffic signs classification project. I used this model as the first step since I might start to understand this problem, and get some preliminary results.  I updated the classification CNN to be the regression model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.  To combat the overfitting, I modified the model to add the dropout layer.

Then I augmented the image data by flipping images and taking the opposite sign of the steering measurement, and used them as same as the original data.  Also, I used the images captured by the left and right camera, and update the steering measurement accordingly.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, and the vehicle cannot recover this. The reason is that the training data did not include that case. To improve the driving behavior in these cases, I re-run the simulator, captured these case, and used them in the training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network, which mimic the network architecture proposed by the Nvidia team.  The big change is to increase the depth of the last convolutional layer, and add the dropout layer to avoid overfit.  However, the original Nvidia network is well designed.  Thus, the dropout rate is chosen low as 0.1.  The overall network architecture is shown as following:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover these cases.  These following images show what a recovery from right side of the road as:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

These following images show what a recovery from left side of the road as:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I used all images from left, center, and right cameras.  I then flipped images and angles.  For example, here are an images from left, center, and right cameras, and their flipped ones:  
** Left Camera Image **     
![alt text][image11]
![alt text][image14]   
** Center Camera Image **  
![alt text][image12]
![alt text][image15]   
** Right Camera Image **  
![alt text][image13]
![alt text][image16]

I then preprocessed the data by making them zero mean, and in the range of -0.5 and 0.5.  Also, I cropped the image to keep only the road pixels, which resulted in the smaller image size, and less computation costs..

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by the following figure as .


#### 4. Continuous Processing and Generator

Storing all simulator images would take over a lot of memory, and huge amount time.  Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator means that only process the pieces of data on the fly only when needing them, which is much more memory-efficient.  Also, it saves the preprocess time.  That is another technique used in this project