# EMA-mess-up
Introducing linear behaviour in-between training samples to reduce generalization error was the main objective of <a href = 'https://arxiv.org/pdf/1710.09412.pdf'>Mixup <a/> augmentation/regularization. Mixup inturn reduces "undesirable oscillations when predicting outside the training examples". The coefficient of the linear combination is sampled from beta distribution(Convex combination to be precise). 

In this repository inspired by mixup, I introduce exponential moving average while sampling training samples, in other words samples are mixed and are exponentially weighed down as training progresses.I call this method <b>messup</b>.

Concretely the algorithm is as follows,

![alt-text-1](images/MessupAlgorithm.PNG "i1")

![alt-text-2](images/RemaAlgorithm.png "i2")

![alt-text-3](images/CemaAlgorithm.png "i3")

