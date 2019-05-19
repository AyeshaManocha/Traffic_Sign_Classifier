# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually. (Refer cell 2)

I used the numpy library and python function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Please refer to Include an exploratory visualization of the dataset in html output or refer to cell 3

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.) (Refer cell 4)

I decided to shuffle the images so that everytime the training that happens is not biased in nature for preprocessing the image data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| size 2x2, 2x2 stride,  outputs 14x16x6 		|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| size 2x2, 2x2 stride,  outputs 5x5x16 		|
| Flatten	        	| input 5x5x16,  output 400              		|
| Fully connected		| input 400, output 120 						|
| RELU  				|           									|
| Fully connected		| input 120, output 84   						|
| RELU  				|           									|
| Fully connected		| input 84, output 43   						|
|						|												|
|						|												|
 
The output of the above layers was the logits.

To compute the loss function supplied to the optimizer, I took the cross entropy of softmax(logits) with the one-hot-encoded labels of the ground truth data. The loss was defined to be the average of the cross entropy across the batch.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate. (Refer cell 5 and 8)

To train the model, I used an Adams optimizer with the following hyperparameters: 
EPOCHS = 50 
BATCH_SIZE = 128
rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.930 
* test set accuracy of 0.918

I started with the LeNet architecture from the lab. In our lab, LeNet proved very effective at classifying black-and-white handwritten digits, so it was plausible to expect that it could identify street sign images, which are of similar overall complexity.

I first modified LeNet to accept an input of color depth 3 and yield an output of 43 classes instead of 10. As a "zero order optimization" I put the number of training epochs to 50 to see where the validation accuracy would plateau. I observed validation set accuracy to be maximum 93%, although the accuracy on the test set climbed to 99.9%. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Five German traffic signs that I found on the web are uploaded in '../test_images/'. The size of the images were different so they were resized when loaded.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)	| Road work   									| 
| Turn right ahead    	| Turn right ahead 								|
| Speed limit (60km/h)	| Speed limit (70km/h)							|
| Speed limit (50km/h)	| End of speed limit (80km/h)	 				|
| Stop      			| Stop              							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

