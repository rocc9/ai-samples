import numpy as np

trainingFile = 'data/mnist_train_0_1.csv'
testingFile = 'data/mnist_test_0_1.csv'
alpha = 0.5
epochs = 10

# import the contents of fileName into an ndarray
def importData(fileName):
    try: 
        csvData = open(fileName)
    except OSError:
        print("Datset file not found")
        raise SystemExit(0)
    resultList = np.genfromtxt(csvData, delimiter=",")
    csvData.close()
    return resultList

# squeeze all feature values between 0 and 1
def squeezeData(data):
    return data/255

# generate an ndarray of random values between -1 and 1
def randomMatrix(rows, cols):
    rng = np.random.default_rng()
    return 2 * rng.random((rows, cols)) - 1

# activation function; applies sigmoid to an entire ndarray
def sigmoid(x):
    return (np.exp(-x) + 1)**(-1)

# derivative of sigmoid
def sigmoidPrime(x):
    return np.multiply(sigmoid(x), 1-sigmoid(x))

# load training data
trainingData = importData(trainingFile)
classLabels = trainingData[:, 0]
examples = trainingData[:, 1:]
examples = squeezeData(examples)

# create weight matrices with random values and bias matrices 
weights_h1 = randomMatrix(784,100)
weights_h2 = randomMatrix(100,50)
weights_o = randomMatrix(50,1)

# start with biases all = 1 at every layer 
bias_h1 = np.ones(100)
bias_h2 = np.ones(50)
bias_o = np.ones(1)
 
print("beginning training")
# training loop
for n in range(epochs):
    for i in range(len(examples)):
        trainPoint = examples[i]
        # compute first hidden layer 
        in_h1 = trainPoint.dot(weights_h1) + bias_h1
        out_h1 = sigmoid(in_h1)
        # compute second hidden layer 
        in_h2 = out_h1.dot(weights_h2) + bias_h2
        out_h2 = sigmoid(in_h2)
        # compute output layer 
        in_o = out_h2.dot(weights_o) + bias_o
        out_o = sigmoid(in_o)
        # discretize output
        if out_o > 0.5:
            prediction = 1
        else:
            prediction = 0
        # compute delta at output and propogate it backwards, first to h2 then h1
        delta_o = (classLabels[i] - prediction) * sigmoidPrime(in_o)
        delta_h2 = np.multiply(weights_o.dot(delta_o), sigmoidPrime(in_h2))
        delta_h1 = np.multiply(weights_h2.dot(delta_h2), sigmoidPrime(in_h1))
        # update weights and biases
        weights_o = weights_o + alpha * np.outer(out_h2, delta_o)
        bias_o = bias_o + alpha * delta_o
        weights_h2 = weights_h2 + alpha * np.outer(out_h1, delta_h2)
        bias_h2 = bias_h2 + alpha * delta_h2
        weights_h1 = weights_h1 + alpha * np.outer(trainPoint, delta_h1)
        bias_h1 = bias_h1 + alpha * delta_h1 
print("completed training")

# load testing data
testingData = importData(testingFile)
classLabels = testingData[:, 0]
dataPoints = testingData[:, 1:]
dataPoints = squeezeData(dataPoints)

print("now testing model")
score = 0
for i in range(len(dataPoints)):
    testPoint = dataPoints[i]
    # compute network output
    in_h1 = testPoint.dot(weights_h1) + bias_h1 
    out_h1 = sigmoid(in_h1)
    in_h2 = out_h1.dot(weights_h2) + bias_h2 
    out_h2 = sigmoid(in_h2)
    in_o = out_h2.dot(weights_o) + bias_o 
    out_o = sigmoid(in_o)
    # discretize output
    if out_o > 0.5:
        prediction = 1
    else: 
        prediction = 0 
    # compare output to the known class label
    if prediction == classLabels[i]:
        score = score + 1
score = 100 * score / len(dataPoints)
print("Accuracy: " + str(score) + "%")