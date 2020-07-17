# EMA-mess-up
Introducing linear behaviour in-between training samples to reduce generalization error was the main objective of <a href = 'https://arxiv.org/pdf/1710.09412.pdf'>Mixup <a/> augmentation/regularization. Mixup inturn reduces "undesirable oscillations when predicting outside the training examples". The coefficient of the linear combination is sampled from beta distribution(Convex combination to be precise). 

In this repository inspired by mixup, I introduce exponential moving average while sampling training samples, in other words samples are mixed and are exponentially weighed down as training progresses.I call this method <b>messup</b>.

Concretely the algorithm is as follows,

![alt-text-1](images/MessupAlgorithm.PNG "i1")

![alt-text-2](images/RemaAlgorithm.png "i2")

![alt-text-3](images/CemaAlgorithm.png "i3")

To explain the algorithm, EMA in general weighs down all the training points in any problem exponentially as time progresses. In other words past examples are weighed down exponentially and the most recent training example is assigned with a higher weight. This weight is the smoothing constant.

Like EMA, Messup has a smoothing constant alpha and an extra hyperparameter called <b>Reset Cycle (C)</b>. All <b> C </b> does is reset the weights or reset exponential dependency between training samples at the value time step <b> C </b>. To understand better Messup like an EMA when the value <b>C</b> in the algorithm is equal to int(Number of training samples/Batch size) in a particular epoch. 
