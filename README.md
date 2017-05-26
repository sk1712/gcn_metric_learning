# Metric Learning using Graph Convolutional Neural Networks (GCNs)

The code in this repository implements a metric learning approach for irregular
graphs. The method has been applied on brain connectivity networks and is
presented in our paper:

Sofia Ira Ktena, Sarah Parisot, Enzo Ferrante, Martin Rajchl, Matthew Lee,
Ben Glocker, Daniel Rueckert, [Distance Metric Learning using Graph Convolutional
Networks: Application to Functional Brain Networks][arXiv], Medical Image Computing
and Computer-Assisted Interventions (MICCAI), 2017.

The code is released under the terms of the [MIT license](LICENSE.txt). Please
cite the above paper if you use it.

There is also implementations of the filters and graph coarsening used in:
* Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural
  Networks on Graphs with Fast Localized Spectral Filtering][arXiv], Neural
  Information Processing Systems (NIPS), 2016.

The global loss function is based on 
* Vijay Kuma, Gustavo Carneiro, Ian Reid [Learning Local Image Descriptors with Deep
Siamese and Triplet Convolutional Networks by Minimizing Global Loss Functions],


[arXiv]:  https://arxiv.org/abs/1606.09375

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/sk1712/gcn_metric_learning
   cd gcn_metric_learning
   ```

2. Install the dependencies. Please edit `requirements.txt` to choose the
   TensorFlow version (CPU / GPU, Linux / Mac) you want to install, or install
   it beforehand. The code was developed with TF 0.8 but people have used it
   with newer versions.
   ```sh
   pip install -r requirements.txt  # or make install
   ```

## Using the model

To use our siamese graph ConvNet on your data, you need:

1. pairs of graphs as matrices where each row is a node and each column is a node feature,
2. a class label for each graph,
3. an adjacency matrix which provides the structure as a graph; the same structure 
   will be used for all samples.

Please get in touch if you are unsure about applying the model to a different
setting.
