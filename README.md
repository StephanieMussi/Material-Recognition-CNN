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


## VGG19 on FMD

## VGG19 on MINC-2500
