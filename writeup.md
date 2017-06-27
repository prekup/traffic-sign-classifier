#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Acknowledgements
Udacity course material, Ian Goodfellow "Deep Learning" book, Sermane & LeCun 2011 traffic sign paper, Stanford CS231n lecture materials, [blog post](https://navoshta.com/traffic-signs-classification/) by Alex Staravoitau (especially idea on symmetries between traffic signs) were very helpful in completing this project.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how may samples there are per class per data set (training, validation, testing).

![Data set summary][./raw_dataset_summary.png]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I decided to convert the images to grayscale because [as described in Sermanet & LeCun 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper, colour information was not very useful for the kind of neural net architecture that is going to be used in this homework.

Here is an example of a traffic sign image before and after grayscaling.
![alt text][./google/color_sign.png]
![alt text][./google/gray_sign.png]

Afterwards contrast-limited adaptive histogram equalization was applied, to bring more detail out of the images and help in the training. It also adjusted the general lightness of the signs as well, if the sign was in a shadow, for example.
![alt text][./google/clahe_sign.png]

As a last step, I normalized the image data to interval [-1,1] because this helps to improve gradient descent performance.

I decided to generate additional data because, some traffic sign classes were underrepresented. For example while '50 km/h speed limit' had 2010 training samples, 'dangerous curve to the left', had only 180 samples.

To add more data to the the data set, I looked at the symmetries between the traffic sign classes, and either flipped or rotated them. Also some traffic signs classes could be converted to the other traffic sign classes.

LeCun and Sermanet also create additional samples by applying various randomized transformations, like perpective transformations, but as the network trained well enough, did not pursue that direction. For over 99% traffic sign recognition accuracy, that would be needed.

Here is an example of an original image and an augmented image:
![left-most is the original image, others augmented versions][image3]

The difference between the original data set and the augmented data set is the following ... 
Originally, there were 34799 test samples, after augmentation, 55859. Validation sample set was not augmented in this way, but it would be desirable in the future, as it would give a more exact information on the training process.

Breakdown of the augmented data set by class and set:
![Data set summary][./augmented_dataset_summary.png]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 
My final model was a LeNet model, augmented with multi-scale pooling described in Sermanet2011 paper. It consists of three convolution layers and two fully connected layers. The outputs of all convolution layers were inputted to the first fully connected layers (additional maxpooling with stride 4x4 was applied to 1st conv layer output; for 2nd conv layer output the additional maxpooling stride was 2x2; no additional maxpooling for 3rd convolutional layer). After the first fully connected layer, an additional fully connected layer was added.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image         					|
| Preprocessing         | 32x32x1 grayscale image   					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x108 					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x64 		|
| RELU					|												|
| Max pooling 3rd conv	| 2x2 stride,  outputs 4x4x64 					|
| Max pooling 1st layer | 4x4 stride,  outputs 4x4x128 					|
| Max pooling 2nd layer | 2x2 stride,  outputs 4x4x32 					|
| Fully connected		| (64+128+32)*4*4, multi-scale, outputs 1024	|
|						| concatenation of last 3 max pools	, 			|
| Dropout				| Dropout regularization						|
| Fully connected		| Outputs 400									|
| Dropout				| Dropout regularization						|
| Softmax				| Outputs 43									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

AdamOptimizer of Tensorflow was used for optimization, with batch size of 256 images, for upto 300 epochs.
Learning rate was set at 0.0002, L1 regularizer penalty at 0.00008, L2 regularizer penalty at 0.0001, dropout probability of the first fully connected layer was set at 0.7 and of the second at 0.5.

For regularization, following methods were used:
* Early stopping regularization, where the training is rolled back to the epoch where the validation data set cross entropy loss is the smallest.
* Dropout regularization, which improves the generalization of the model and helps the optimizer explore more local minima. 
* Elastic net regularization, which is basically a linear sum of L1 loss and L2 loss. L2 regularization helps to better converge the training process, while L1 regularization helps in increasing the sparsity of the weights and thus increase the importance of the trained weights, while at the same time reducing complexity of the trained model (as more weights will be at or near zero). With elastic net regularization, after 20 epochs, training accuracy was 0.999, and validation accuracy = 0.957. Thanks to dropout regularization, overfitting did not occur, as other local minima were also tried, while elastic net regularization ensured a robust loss value to guide the optimization process.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.982
* test set accuracy of 0.972

Training data set was augmented using symmetries between the traffic signs. Validation and test data sets were not modified 
ˇˇˇ
X_augmented_train, y_augmented_train = generate_data(X_train, y_train)
X_processed_train, y_processed_train = preprocess(X_augmented_train), y_augmented_train
X_processed_test = preprocess(X_test)
X_processed_valid = preprocess(X_valid)    
ˇˇˇ

Accuracy of the data sets was calculated by setting all the dropout rates to 1.
ˇˇˇ
def evaluate(X_data, y_data, find_top_k = False, k = 3):
    num_examples = len(X_data)
    total_accuracy, total_loss = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation, loss_operation], 
                                  feed_dict={x: batch_x, 
                                             y: batch_y, 
                                             keep_prob_1: 1.0,
                                             keep_prob_2: 1.0})
        total_accuracy += accuracy * len(batch_x)
        total_loss += loss * len(batch_x)
    if find_top_k:
        top_k = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k), 
                         feed_dict={ x: X_data, 
                                     y: y_data, 
                                     keep_prob_1: 1.0,
                                     keep_prob_2: 1.0})
    else:
        top_k = None
    return total_loss / num_examples, total_accuracy / num_examples, top_k
ˇˇˇ

An iterative approach was chosen.
* Current architecture is a slightly modified version of the multi-scale feature LeNet architecture from Sermanet&LeCun 2011 paper. It achieved in the paper 99.17% accuracy.
* Problem with the architecture was that it suffered from overfitting. 
Thus regularization had to be added. Dropout reduced issues with overfitting and elastic net regularization (L1 + L2 regularization) improved the convergence of the training. Early stopping based on the cross entropy loss of the validation data set classification enabled to choose the version of the fitted network that should fit best against the test data set.
Also an additional fully connected layer was added, as it helped to improve accuracy of test data classification by around 1%.
* Parameters that were tuned:
    * Learning rate was adjusted from initial 0.001 to 0.0002, because it was too fast and lead to underfitting.
    * Dropout rate of the first fully connected layer was set at 0.7. This did not effect the training a lot. Gave around 0.3-0.4% accuracy.
    * Dropout rate of the second fully connected layer was set at 0.5. This hyperparameter turned out to be important, as it gave around 1% improvement in accuracy. It helped to prevent overfitting.
    * Batch size affected the performance of the training. 256 seems to be optimal for my hardware, a Nvidia P5000 graphics card.
    * L2 penalty was set at 0.0001 and L1 penalty at 0.00008

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

From the examples one can deduce, that signs where small features make all the difference, like 'Roundabout mandatory' sign vs 'Keep right', are more difficult to successfully classify.

Here are five German traffic signs that I found on the web:

![alt text][./google/sign_1.jpg] ![alt text][./google/sign_3.jpg] ![alt text][./google/sign_4.jpg] 
![alt text][./google/sign_5.jpg] ![alt text][./google/sign_6.jpg]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        	|     Prediction						| 
|:-------------------------:|:-------------------------------------:| 
| Yield      				| Yield   								| 
| Turn righ-ahead     		| Turn righ-ahead 						|
| Vehicles over 3.5 		| Vehicles over 3.5						|
| metric tons prohibited	| metric tons prohibited				|
| Keep right	      		| Keep right			 				|
| Road narrows on the right	| Road narrows on the right				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.2%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.927), and the image does contain a yield sign. In case of the second image, model was more unsure about 'Turn right-ahead', with probability 0.803.

The top five soft max probabilities were

| Probability  	|     Prediction	        					| 
|:-------------:|:---------------------------------------------:| 
| 0.927      	| Yield   										| 
| 0.803     	| Turn righ-ahead 								|
| 0.999 		| Vehicles over 3.5								|
| 				| metric tons prohibited						|
| 1.000	      	| Keep right					 				|
| 0.999			| Road narrows on the right	      				|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

