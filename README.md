# Attention Over Attention Mechanism on Reviews for Rating Prediction

Our project is to improve a well known paper in the field of Recommender Systems.

We chose the paper "Neural Attentional Regression model with Review-level Explanations" (NARRE):

*Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. [Neural Attentional Rating Regression with Review-level Explanations.](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf) 
In WWW'18.*

Our main task was predication of rating by utilizing the users & items reviews.
We proposed two extensions to the original NARRE model:

1) NARRE_DNN: In the original NARRE model, the prediction layer was consists of a simple linear multiplication for the final prediction value. We created fully connected layers with non-linear activations, to find new relations and to improve the prediction accuracy.

2) NARRE_Attention: In the original NARRE model, the final features representation were built using simple concatenation between all features, we proposed a smarter way to combine the features using the feature-attention mechanism.


**All models can be found in '/model' folder, and can be imported directly.**


## Environments

- python 2.7
- Tensorflow (version: 1.14)
- numpy
- pandas
- Tensorboard


## Dataset

In our experiments, we use the datasets from Amazon 5-core(https://nijianmo.github.io/amazon/index.html).
Since GitHub has a limitation space, you must download the data and perform the preprocess stage before running the model.

We have also used Word2Vec embedding of "GoogleNews-vectors-negative300" (https://code.google.com/archive/p/word2vec)

## Visualization

For visualization and graphs we used tensorboard.
Our code support logging of all model outputs into the "./log" and "./output" folders


## Example to run the codes

Data preprocessing:
```
python loaddata.py
```

Train and evaluate the model:
```
python train.py
```

models:
```
./model/NARRE.py
./model/NARRE_DNN.py
./model/NARRE_Attention.py
```

**An Example for running the code can be found in main.ipynb**



Last Update Date: July 31, 2020
