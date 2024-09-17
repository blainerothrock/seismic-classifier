# Clustering Mars Seismic Data with a Convolutional Auto Encoders
__In Progress__

## Overview
The goal of this experiment is to use a convolutional auto encoders to cluster embeddings of sesimic samples from the Mars Insight Lander.

Auto Encoders use a encoder and decoder to model to learn to reproduce data. This provides a testing ground unsupervised learning in deep networks. The Encoder extracts features from the data into a higher dimensional embeddings space. Decoders upscale data from the hidden state produced 

**Research Inspiration**:
* [Deep Clustering with Convolutional
Autoencoders](https://www.researchgate.net/profile/Xifeng_Guo/publication/320658590_Deep_Clustering_with_Convolutional_Autoencoders/links/5a2ba172aca2728e05dea395/Deep-Clustering-with-Convolutional-Autoencoders.pdf)
* [Variational Recurrent Auto Encoder](https://pythonawesome.com/variational-recurrent-autoencoder-for-timeseries-clustering-in-pytorch/)

## Run
Update `gin.config` with desired parameters.
```shell script
python train.py
```
View results with Tensorboard
```shell script
tensorboard --logdir runs
```
Models will be stored in `./models` (ignored from source control)

## Results
The current model configuration in running, but there are some bugs to workout before reporting results.

## Discussion & Next Steps
It is worth exploring a few variations of Auto Encoders as this can be used for other seismic experiments. Implementing Fully Connected and Recurrent autoencoders will be useful to the framework. 