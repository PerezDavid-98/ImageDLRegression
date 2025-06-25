# Deep Learning Image Regression

This project is for one of my Master's Courses.

The task for this project was to use a dataset containing a total of 1027 images of trays containing different foods, with a spatial dimension of 3264 x 2448 pixels in RGB. For each image, the percentage of pixels containing food was calculated.

Based on these (image, percentage) pairs, models based on Convolutional Neural Networks (CNNs) were defined and trained to provide the percentage of food pixels in a given image.

The quality of the models was estimated using the `Root Mean Square Error (RMSE)` error metric.

Specifically, three approximations were implemented and compared with a baseline with an RMSE of 4.8.

The models had to be implemented using the `keras` module form the library `TensorFlow`.

## Overview
Three models were trained for this task. The first two models were created from 0 and for the third, transfer learning was used to adapt an existing model to the task at hand. 

All three models were trained for 10 epochs each and with a batch size of 16. All of the training was done using [Google Colab](https://colab.google/). 

The learning rate was initialized at a value of `0.0001` for all the models trained.  

### First model

The first was a completely new and simple model created by me. It followed a very simple structure with 5 blocks, each containing:
* 2 `Conv2D` layers with `ReLu` activation.
* 1 `MaxPooling2D` layer.

The filters were multiplied for each block:
* 8 and 16 for the first block (8 the first `Conv2D` layer and 16 the second one).
* 32 for the second block
* 64 for the third block.
* 128 for the fourth block.
* 256 for the fifth block.

Default values were used for the `stride` and `padding` parameters.

Lastly, a `Flatten` layer was used to connect the `Conv2D` block with two `Dense` layers at the end of the model to compute the data from the convolution layers and to get an output.
The first with 1024 Neurons and the last one with only one Neuron to get the Regression output. 

After training and validation, the model ended up with an RMSE of 4.86 during tests, slightly above the provided baseline of 4.8.
Although the RMSE after evaluation falls short compared with the baseline by a little bit, the purpose for this model was to serve as a basic design to be improved in model 2.

### Second Model
The second model is an evolution of the first, where an attempt is made to improve the model based on the results obtained. It is expected to improve the model's results by using techniques such as weight initialization, dropout, early stopping, etc.

The changes made to the model are:

* One of the `Conv2d` and `MaxPooling2D` blocks has been removed; the reason is that simplifying the model is expected to prevent overfitting.
* The `Flatten` layer has been replaced with a `GlobalAveragePooling2D` layer.
* The number of neurons in the fully connected layer has been reduced.
* A second fully connected layer has been added before the model output.
* The filters for the `Conv2D` layers also change:
    * The fourth block is eliminated (block with 128 filters)
    * The fifth block now has one `Conv2D` with 128 filters and another one with 256.

Part of the changes, compared to the previous model, occurred during training.

In the first model, since it was a baseline for improvement, improvements were not taken into account during model training to improve results and, above all, to avoid overfitting.

In the second model various measures have been implemented to improve the model:

* Learning Rate Decay (`ReduceLROnPlateau`) has been implemented.
* `Early Stopping` has been implemented to stop training if it is detected that the model is beginning to show signs of overfitting.

After training and validation, the model ended up with an RMSE of 3.40 during tests. This time the RMSe obtained beats the baseline proposed at the beginning of the project. 
Although the methods applied to the model during training (`ReduceLROnPlateau` and `Early Stopping`) did not trigger at any point. It is possible to change the hyperparameters to make these methods more strict,
but by only training for 10 Epochs, there is barely any time for them to activate for this model arquitecture. 


### Third model
For the third model transfer learning is used, the model used is the `DenseNet` model pre-trained with the ImageNet dataset.

Since `Keras` has three `DenseNet` implementations:
* DenseNet121
* DenseNet169
* DenseNet201

Where the main difference is the model depth, it was decided to train all three and choose the best one, taking into account the results, training time and model complexity.

The same training improvements seen in model 2 are applied here, that is, `ReduceLROnPlateau` and `Early Stopping`.

Additionally, two layers were added at the end of every model to adapt it to the task at hand, same as with the two first models, two `Dense` layers were added at the end of the model to compute the data from the convolution layers and to get an output.
The first with 1024 Neurons and the last one with only one Neuron to get the Regression output.


Lastly, the pre-trained weights were fixed to not be overwritten during training. This way, only the new layers would learn based on the training. This was done to make the training process faster.
A future work on this project could be to retrain the base models using a very small Learning rate to fine-tune the model to the needs of the project and reduce RMSE even further. 

The results for the models are as follows:

| Model          |   RMSE   | Training Time (seconds)| Avg Time per Epoch (seconds) | 
| :-------------:| :------: | :--------------------: | :--------------------------: |
| DenseNet121    |   2.38   |          1627          |               163            |
| DenseNet169    |   2.17   |          1829          |               183            |
| DenseNet201    |   2.22   |          2410          |               241            |

As seen in the table, the sweet spot for results, training time and model complexity is the `DenseNet169` model. Incidentally, both `ReduceLROnPlateau` and `Early Stopping` were applied for this method during training, in Epoch 9 and 10 respectively. 


## Final testing
For the final test, a sample of `n` random images was selected from the test split of the dataset, and a prediction was generated for each of the 3 models developed.
In my case, I selected `n = 5` images. 

The predictions were then compared against the real values. 

## Implementation
All the code can be found in the `ImageDLRegression.ipynb` Jupyter Notebook. Every step is explained (In Spanish) and reasoning for every decision is provided.

> [!WARNING]
> At the time of writing this README, there is a bug on GitHub where the Jupyter Notebooks do not render in the preview page. More info on this issue can be found here: [Notebook no longer rendering on GitHub: "state" key missing from metadata.widgets #155944](https://github.com/orgs/community/discussions/155944). If a solution is found to this issue I will update this README to remove this warning.

Additionally, a full report for this task can be read (in Spanish) in the `Deep Learning Image Regression.pdf` file in this repository.