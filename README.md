# Material_Recognition_CNN
This project aims to perform material recognition task on [Flickr Material Database](https://people.csail.mit.edu/lavanya/fmd.html) and [Materials in Context Database](http://opensurfaces.cs.cornell.edu/publications/minc/).
The pre-trained deep convolutional neural network model [VGG19](https://keras.io/api/applications/vgg/) is refined to fit the dataset. 

## Data split
The dataset downloaded from the website needs to be split into train data, test data (and validation data).  
### FMD
Using the labels in folder ["DS_FMD/splits"](https://github.com/StephanieMussi/Material_Recognition_CNN/tree/main/DS_FMD/splits), the dataset is split into train set and test set with a ratio of __80:20__.  
The codes for splitting the dataset are in ["FMD_split.ipynb"](https://github.com/StephanieMussi/Material_Recognition_CNN/blob/main/FMD_split.ipynb), and the directories need to be modified based on the locations of labels and images.  
```python
train_split_dir = "./DS_{}/splits/train_split_{}.txt".format(dataset, split)
test_split_dir = "./DS_{}/splits/test_split_{}.txt".format(dataset, split)

train_folder = './DS_{}/data/split_{}/train'.format(dataset, split) 
test_folder = './DS_{}/data/split_{}/test'.format(dataset, split) 
```
Using the codes as shown above, the directories of splitted data are:  
* DS_FMD
  * test
    * fabric
    * foliage
    * ...
    * wood
  * train
    * fabric
    * foliage
    * ...
    * wood


### MINC-2500
The labels come with the images when downloaded from the website. Using the labels, the data is split into train set, validation set and test set.  

In ["MINC_split.ipynb"](https://github.com/StephanieMussi/Material_Recognition_CNN/blob/main/MINC_split.ipynb), the directories are specified in the following way:  
```python
train_split_dir = "minc-2500/labels/train1.txt"
validate_split_dir = "minc-2500/labels/validate1.txt"
test_split_dir = "minc-2500/labels/test1.txt"

train_folder = 'minc-2500/train'
validate_folder = 'minc-2500/validate'
test_folder = 'minc-2500/test'
```

The splitted data has the hierarchy of:  
* minc-2500
  * test
    * brick
    * carpet
    * ...
    * wood
  * train
    * brick
    * carpet
    * ...
    * wood
  * validate
    * brick
    * carpet
    * ...
    * wood


## VGG19 model
The structure of VGG19 model is shown as below:  

<img src="https://github.com/StephanieMussi/Material_Recognition_CNN/blob/main/Figures/VGG19.png" width="500" height="200">  

As it can be seen, it contains 5 pairs of convolutional layer followed by maxpooling layer of decreasing sizes. After that, the data is flattened and fed into 2 fully connected layer. Lastly there is a Softmax output layer of 1000 nodes.  


### On FMD
The default VGG19 model is modified to fit the output classes of FMD dataset. The output layer of 1000 classes is removed, and a Softmax output layer of 10 nodes is added.  
```python
name_last_layer = str(vgg19_model.layers[-1])
for layer in vgg19_model.layers:
    if str(layer) != name_last_layer:
        model.add(layer)
```
```python
n_classes = 10
model.add(keras.layers.Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
          bias_regularizer=regularizers.l2(0.01)))
```

The model is trained for 30 epochs, and the accuracies of the model are:  
Train Accuracy | Test Accuracy
------------ | -------------
99.87% | 78.50%

The graphs of accuracies and losses are plotted:  
<img src="https://github.com/StephanieMussi/Material_Recognition_CNN/blob/main/Figures/FMDAccuracy.png" width="300" height="200">
<img src="https://github.com/StephanieMussi/Material_Recognition_CNN/blob/main/Figures/FMDLoss.png" width="300" height="200">   


## On MINC-2500
Similar to the model on FMD, The default VGG19 model is modified to fit the output classes of MINC dataset. The output layer of 1000 classes is removed, and a Softmax output layer of 10 nodes is added.  
```python
name_last_layer = str(vgg19_model.layers[-1])
for layer in vgg19_model.layers:
    if str(layer) != name_last_layer:
        model.add(layer)
```
```python
n_classes = 23
model.add(keras.layers.Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
          bias_regularizer=regularizers.l2(0.01)))
```

In order to make a better comparison, this model is also trained for 30 epochs. The accuracies of the model are:  
Train Accuracy | Test Accuracy
------------ | -------------
67.00% | 65.24%

The graphs of accuracies and losses are plotted:  
<img src="https://github.com/StephanieMussi/Material_Recognition_CNN/blob/main/Figures/MINCAccuracy.png" width="300" height="200">
<img src="https://github.com/StephanieMussi/Material_Recognition_CNN/blob/main/Figures/MINCLoss.png" width="300" height="200"> 

As it is observed, there is a slight degration in performance when training on a large dataset like MINC-2500 as compared to on a small dataset like FMD. However, there is no significant gap between train accuracy and test accuracy is the former case, which indicates that the performance may further improve if the number of epochs increases.
