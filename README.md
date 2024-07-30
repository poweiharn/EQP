# Evolutionary Quadtree Pooling for Convolutional Neural Networks
#Overview
Despite the success of Convolutional Neural Networks (CNNs) in computer vision, it can be beneficial to reduce parameters, increase computational
efficiency, and regulate overfitting. One such reduction technique is the use of so-called pooling, which gradually reduces the spatial dimensions of the data throughout the network.
Recently, Quadtree-based Genetic Programming has achieved state-of-the-art results for optimizing spatial areas on customized
requirements in different grid structures. Motivated by its success, we
propose to extend this approach to pooling layers of CNNs.
In this direction, this paper introduces a new way to look at each pooling layer.
Specifically, we propose an Evolutionary Quadtree Pooling (EQP) method that can identify the best pooling scheme. By embedding multiple
quadtrees set as a pooling scheme in the pooling layers of a CNN, we are able to operate crossover and mutation on the feature maps.
The evolutionary process of EQP guides the
search to provide more reliable evaluations, where each individual
can be seen as a CNN with a new type of pooling scheme.
Our experimental results show that the best candidate network of EQP outperforms state-of-the-art max, average, stochastic, median, soft, and mixed pooling in accuracy and overfitting reduction while maintaining low computational costs.


## Minimum Requirements
* python3.6
* pytorch1.6.0 + cuda10.1

 
## Usage
* Run main.py using configs/config.txt is for EQP
* Run main.py using configs/config_ea.txt is for GA
* Run main_random.py using configs/config_random.txt is for Random Search
* Run train.py is for pooling baselines

The results are saved in the /log folder.


## Datasets
MNIST, CIFAR10, CIFAR100, and SVHN

## Pooling  args
```
'M': Max Pooling
'A': Average Pooling
'S': Stochastic Pooling
'C': Median Pooling
'X': Soft Pooling
'T': Mixed Pooling
```

## Dataset args
```
'mnist', 'cifar10', 'cifar100', 'svhn'
```

## Model args
```
'vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn'
```

## Fitness Metrics args
```
'best_acc': Search for the best validation accuracy
'gen_gap': Search for the minimum generalization gap 
'best_op': Search for the least number of pooling operations 
'ea_op': Search for the least number of pooling operations for GA
'best_acc&op': Search for the best validation accuracy and the least number of pooling operations 
'best_acc&op&gen' Search for the best validation accuracy, he minimum generalization gap, and the least number of pooling operations.
```

