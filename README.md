# TrafficSignClassifier
Classifies German traffic sign images using Deep Learning 
#**Traffic Sign Recognition** 
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/image1.png 
[image2]: ./images/image2.png 
[image3]: ./images/image3.png 
[image4]: ./images/image4.jpg 
[image5]: ./images/image5.jpg 
[image6]: ./images/image6.jpg 
[image7]: ./images/image7.jpg 


### Writeup / README

[project code](https://github.com/bhavanabhasker/TrafficSignClassifier)

### 1. Data Set Summary & Exploration

To better understanding the training and test data, I used numpy library in python. 

The summary statistics are: 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2.Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

Bar chart showing the distribution of training images per class

![alt text][image1]

Bar chart showing the distribution of testing images per class 

![alt text][image2]

The maximum (label, count) from the training sample set (2, 2010)

The minimum (label, count) from the training sample set (0, 180)

### Design and Test a Model Architecture

The following preprocessing steps were performed on the images: 
1.Conversion of images to grayscale. This enables faster processing as we have only one channel 

Here is an example of a traffic sign image after grayscaling.

![alt text][image3]

2. Normalization of Images. 

#### 2.Model Architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution      	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Normalization					|												|
| Convolution 	    |  1x1 stride, Valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Normalization					|												|
| Fully connected		|        									|
| RELU					|												|
| Fully connected		|        									|
| RELU					|												|
| Softmax				|       									|

 
#### 3.Training the model 

The parameters used for training the model are: 
1. Learning rate with a value of 0.005 
2. Batch size of 128 
3. Number of epochs used were: 15 

#### 4. Approach to find the optimal solution 

Initially, LeNet architecture was used which achieved a higest accuracy of 88%. An iterative approach consisting of convolution layers along with some additional normalization layer was finally used. This achieves the accuracy of 93.3% on testing dataset. 
The network architecture is described in the above table 

My final model results were:
* validation set accuracy of 95.3%
* test set accuracy of 93.3%

-> LeNet Architecture was initially chosen as it is easy to implement
-> The architecture could achieve the maximum accuracy of 88% 
-> To improve the model accuracy, severl modifications were performed, 
1. dropout layer with the keep prob of 0.5 and 0.7 was added. This reduced the accuracy from 88 to 85 
2. Removal of third fully connected layer from the architecture 
3. Finally normalization layer was introduced after layer 1 and 2 which could reach an accuracy of 93.3% on testing data 

The epochs and learning rate were adjusted with each modification to network architecture. The epoch and learning rate for the final model are 15 and 0.05 respectively. 
Initial architecture had three fully connected layers. The final layer was removed to improve the performance of the model. 

###Test a Model on New Images

For the testing the performance of the model on the unseen images, images were downloaded from German website. 
Here are German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7]


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|Go straight or left    		|Go straight or left									| 
|Priority road     			| Priority road								|
| Yield					| Yield											|
| Speed limit (20km/h)      		| Speed limit (30km/h)	|

The model was able to correctly guess 3 of the 4 traffic signs, which gives an accuracy of 75%. 

For the first image, the model is relatively sure that this is a "Go straight or left" (probability of 0.99), and the image does contain a "Go straight or left" image. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| Go straight or left 									| 
| .999     				| Priority road 									|
| 1					| Yield											|
| .54      			| Speed limit (20km/h) 				 				|


The last prediction is incorrectly classified as 30 km/h 

The third image was fed to the outputFeatureMap function to dermine the features.It was observed that model was able to classify the image as yield sign in FeatureMap14. 
Each of 16 featuremaps provides good insights about how the model used it's weights to learn the image characteristics. 



