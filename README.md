# Scean_Recognition for Place365

### Installation
if you want to use pretrained models, then all you need to do is:
```sh
git clone https://github.com/djang000/Scene_Recognition.git
```

if you also want to train new modes, you will need natural images for training and MobileNet wegihts by running.

you can download MobileNet_V2 weight from below site
```sh
https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
```
To prepare the Place365 dataset for use with train_mobilenet_V2.py, you can download Place365 dataset
I used place365-standard dataset for training (105GB with 256 x 256 images).

if you download Place365 dataset, put train image and meta directory to data folder.
```sh
http://places2.csail.mit.edu/download.html
```

### Usage

Following are examples of how the scripts in this repo can be used. 

- Scene_eval.ipynb

	you can show the evaluation result using trained model.

- train.py

	you should run below command for training.

	```sh
	python train_mobilenet_V2.py
	```

