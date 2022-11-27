# Image Retrieval System

## Dataset

Please first download the dataset [static.zip](https://www.dropbox.com/sh/qnjoc15yjfs2epo/AADpE6CSB3ZOUjW6FeJSzRvca?dl=0).

Then, unzip it and move it to your project path.


## Requirements

The python version is Python3.7. Here are the required packages. You can run the program on you PC with CPUs or GPUs.

```
Flask==1.1.1
opencv-python==4.0.0
torch=1.13.0
torchvision==0.14.0
timm==0.6.11
numpy==1.15.4
Pillow==5.3.0
```

## Usage

First download the image repository through the link. Put the downloaded images to the './static/image_database'. And then run the following command to start the server:

```
$ python image_retrieval.py arg1 arg2
```

arg1 is the model you want to use and you can choose from {'alexnet', 'resnet', 'vit'}. arg2 is the similarity metric and you can choose from {'e', 'cos', 'kl'}. After running the command, you can enter the system through http://127.0.0.1:9008/ in the browser. To do a retrieval, you should first upload an image query and then click the run button to retrieve the results.
# image_retrieval
