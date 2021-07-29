# ConstrainedNeuralNetworks
Training Neural Networks to Behave Using Interpretability Methods

Artificial neural networks are excellent machine learning models but are often referred to as “black boxes”, meaning that the reasoning behind their decisions is obscured. The field of neural network interpretability attempts to explain why these models make the decisions they do.

In my research I combine methods for interpreting neural network decisions with the neural network training process to develop networks that learn to solve problems in a specified way. 

Rather than training neural networks only to maximise prediction accuracy, I train the networks while enforcing a constraint that the network’s behaviour interpretation matches our human expectations, with the goal of improving our ability to understand and trust neural networks. 

Finally, I explore an alternative training objective that seeks to replicate the effects of this guided training method but without the need for a predefined set of human expectations.

Files

Constrained_Training.ipynb contains mask loss training and SLP interpretability code

nnet_experiments.ipynb contains code from denoising and double MNIST experiments

dotloss.ipynb contains code focused on Dotloss constrained training

double_mnist_dotloss.ipynb contains the code from the double MNIST experiment run with dotloss

iris_cancer_constrained.ipynb contains code from the experiments on iris-cancer constrained training + dotloss
