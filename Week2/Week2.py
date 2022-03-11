#Implement perceptron
import numpy as np

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    d, n = data.shape
    theta_0 = np.zeros((1, 1))
    theta = np.zeros((d, 1))

    for t in range(T):
        for i in range(n):
            y = labels[0, i]
            x = data[:, i]
            m = np.dot(x, theta) + theta_0
            if np.sign(m * y) <= 0:
                theta[:, 0] = theta[:, 0] + y * x
                theta_0 = theta_0 + y
    return (theta, theta_0)

# Implement averaged perceptron
#procedure averaged_perceptron({(x_(i), y_(i)), i=1,...n}, T)
   # th = 0 (d by 1); th0 = 0 (1 by 1)
    #ths = 0 (d by 1); th0s = 0 (1 by 1)
    #for t = 1,...,T do:
     #   for i = 1,...,n do:
	  #  if y_(i)(th . x_(i) + th0) <= 0 then
	   #     th = th + y_(i)x_(i)
		#th0 = th0 + y_(i)
	    #ths = ths + th
	    #th0s = th0s + th0
    #return ths/(nT), th0s/(nT)
    import numpy as np

    def averaged_perceptron(data, labels, params={}, hook=None):
        # if T not in params, default to 100
        T = params.get('T', 100)

        # Your implementation here
        d, n = data.shape
        th = np.zeros((d, 1))
        th0 = np.zeros(1)
        ths = np.zeros((d, 1))
        th0s = np.zeros(1)
        for t in range(T):
            for i in range(n):
                y = labels[0, i]
                x = data[:, i]
                k = np.dot(x, th) + th0
                if np.sign(y * k) <= 0:
                    th[:, 0] = th[:, 0] + y * x
                    th0 = th0 + y
                ths = ths + th
                th0s = th0s + th0
        return (ths / (n * T), th0s / (n * T))

#Evaluating a classifier - onstruct a testing procedure that uses a training data set, calls a learning algorithm to get a linear separator (a tuple of \theta, \theta_0θ,θ
#0 ​ ), and then reports the percentage correct on a new testing set as a float between 0. and 1..
import numpy as np

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0=learner(data_train,labels_train)
    m = data_test.T@th + th0
    res = np.sign(m.T)
    p=(labels_test==res)
    print(res)
    return np.mean(p)
    pass

#Construct a testing procedure that uses a training data set, calls a learning algorithm to get a linear separator (a tuple of \theta, \theta_0θ,θ
#0 ​), and then reports the percentage correct on a new testing set as a float between 0. and 1..
import numpy as np

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0=learner(data_train,labels_train)
    m = data_test.T@th + th0
    res = np.sign(m.T)
    p=(labels_test==res)
    print(res)
    return np.mean(p)
    pass

#Construct a testing procedure that takes a learning algorithm and a data source as input and runs the learning algorithm multiple times, each time evaluating the resulting classifier as above. It should report the overall average classification accuracy.
def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    s=0
    for i in range(it):
        data_train, labels_train = data_gen(n_train)
        print(data_train)
        data_test, labels_test = data_gen(n_test)
        s+=eval_classifier(learner, data_train, labels_train, data_test, labels_test)
    return s/it
    pass

#Evaluating a learning algorithm with a fixed dataset- cross-validation
import numpy as np
def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k