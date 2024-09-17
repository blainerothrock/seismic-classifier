# SeisML
![Test](https://github.com/blainerothrock/seisml/workflows/Test/badge.svg)
[![codecov](https://codecov.io/gh/blainerothrock/seisml/branch/master/graph/badge.svg)](https://codecov.io/gh/blainerothrock/seisml)

a deep learning experimentation framework for seismic data.

Created to fullful of research practicum for Masters in Artificial Intelligence @ 
Northwestern University. Advised by [Suzan van der Lee](https://www.earth.northwestern.edu/our-people/faculty/van-der-lee-suzan.html).

The main goal of this project is to explore a project structure that allowed for rapid experimentation, with hyperparameter reporting for use by seismology researchers.

The codebase includes a reproduction and generalization of the paper [Automating the Detection of Dynamically Triggered Earthquakes via a Deep Metric Learning Algorithm](https://pubs.geoscienceworld.org/ssa/srl/article-abstract/91/2A/901/579921/Automating-the-Detection-of-Dynamically-Triggered?redirectedFrom=fulltext). The goal was to extend the research the research to dynamically triggered tremors. In addition, explored new experiments using the dataset introduced [Generalized Seismic Phase Detection with Deep Learning](https://pubs.geoscienceworld.org/ssa/bssa/article/108/5A/2894/546740/Generalized-Seismic-Phase-Detection-with-Deep?casa_token=XTBzGohrUBoAAAAA:Y2OHyF4u4VknJXTjPxs6lPjA5rW-2yIkjEVSSvJDSP4GoSzxZu7-GeN9F7nE0pI663-EqpU) and unsupervised deep clustering with data from SEIS, a seismic instrument deployed as part of the InSight Mars lander, in an attempt to automate Martian seismic event classification.

Developing a baseline phase detection experiment using work and data from the paper Generalized Seismic Phase Detection with Deep Learning. • Experimented with unsupervised deep clustering with data from SEIS, a seismic instrument deployed as part of the InSight Mars lander, in an attempt to automate Martian seismic event classification.

## Active Experiments
* [Triggered Earthquake Detection](experiments/triggered_earthquake/README.md)
* [Triggered Tremor Detection](experiments/triggered_tremor/README.md)
* [Mars Insight Seismic Data Unsupervised Clustering](experiments/mars_insight_clustering/README.md)

## Supported Models
* [Deep Convolutional Network](seisml/networks/convnet.py)
* [Dialated Convolutional Network for Deep Clustering](seisml/networks/dilated_convolutional.py)
* [Convolutional Auto Encoder](seisml/networks/convolutional_autoencoder.py)

## Datasets
* [Triggered Earthquake](experiments/triggered_earthquake)
* [Triggered Tremor](experiments/triggered_tremor)
* [Mars Insight](experiments/mars_insight_clustering)

## Repo Structure
* `experiments/`
    - the directory that contains code specific to a experiment utilizing seisml components.
    - This includes, model training code, inference and hyperparameter configuration 
* `playground/`
    - A place for experiments in progress, example code, data exploration, etc.
* `seisml`
    - The root directory for the framework (python package)
    * `core`
        - contains universal components
        - `transforms`
            - model after transforms in `torchvision`, used for preprocessing steps before feeding data into a model
    * `datasets`
        - build in Pytorch datasets for use in experiments
    * `metrics`
        - calculations modeling
    * `networks`
        - custom Pytorch models
    * `utility`
        - universal helper methods
* `tests`
    - Pytest unit tests for seiml code. These test run in continuous integration and utilize limited resources
* `tests-non-ci`
    - Pytest test that are more specific to experiement debugging. Not intended for continuous integration because of the amount of resources used.
* `environment.yml`
    - conda environment file for CI and use
    
## Installation
* clone the repository and `cd` to root
* create a new Anaconda environment
```shell script
conda create env -f environment.yml
``` 
* run a experiments following the `README.md` found in the specific experiment directory.


## Inspiration 
The inspiration and starting codebase for this model is from the Seismological Research Letters paper 
*[Automating the Detection of Dynamically Triggered Earthquakes via a Deep Metric Learning Algorithm](https://pubs.geoscienceworld.org/ssa/srl/article-abstract/91/2A/901/579921/Automating-the-Detection-of-Dynamically-Triggered)*
([original codebase](https://github.com/interactiveaudiolab/earthquakes)). 
* Steps to reproducing this paper can be found [here](experiments/triggered_earthquake/README.md).
