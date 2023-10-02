import numpy as np
import matplotlib.pyplot as plt

from TreeMaker import TreeMaker

# imports file named fileName and returns a numpy array. Prints warning and ends execution if the file is not found
def importData(fileName):
    try: 
        csvData = open(fileName)
    except OSError:
        print("Datset file not found")
        raise SystemExit(0)
    resultList = np.genfromtxt(csvData, delimiter=",")
    csvData.close()
    return resultList

# same as importData, but skips the first line. Used for pokemonStats.csv
def importData2(fileName):
    try: 
        csvData = open(fileName)
    except OSError:
        print("Datset file not found")
        raise SystemExit(0)
    resultList = np.genfromtxt(csvData, delimiter=",", skip_header=1)
    csvData.close()
    return resultList

# same as importData2, but reads the data as strings and converts them to int. Used for pokemonLegendary.csv
def importData3 (fileName):
    try: 
        csvData = open(fileName)
    except OSError:
        print("Datset file not found")
        raise SystemExit(0)
    tempList = np.genfromtxt(csvData, delimiter=",", skip_header=1, dtype="str")
    resultList = []
    for i in tempList:
        if(i == 'True'):
            resultList.append([1])
        else: 
            resultList.append([0])
    csvData.close()
    return resultList

# displays the current plot 
def displayPlot():
    plt.suptitle("Decision surfaces of 4 decision trees trained on synthetic datasets")
    plt.legend()
    plt.show()

# plots decision tree boundary for tree trained on syntheticData 
# comes from the provided blog post on scikit-learn.org
def plotDecisionSurface(tree, syntheticData, syntheticFeatures, syntheticTarget, i):
    n_classes = 2
    plot_colors = "br"
    plot_step = 0.02

    X = syntheticData[:, syntheticFeatures]
    y = syntheticData[:, syntheticTarget]

    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    plt.subplot(2, 2, i + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = []
    for gridX in range(len(xx)):
        for gridY in range(len(xx[gridX])):
            pt = [xx[gridX, gridY], yy[gridX, gridY]]
            Z.append(tree.predict(pt))
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel("Attribute 1 for Synthetic Dataset #" + str(i+1))
    plt.ylabel("Attribute 2")
    plt.axis("tight")

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=syntheticFeatures[i],
                    cmap=plt.cm.Paired, edgecolor="black")

    plt.axis("tight")

syntheticDataFiles = ['data/synthetic-1.csv', 'data/synthetic-2.csv', 'data/synthetic-3.csv', 'data/synthetic-4.csv']
pokeDataFiles = ['data/pokemonStats.csv', 'data/pokemonLegendary.csv']
# these variables represent indices in a data vector
syntheticFeatures = range(2)
syntheticTarget = 2
pokeFeatures = range(44)
pokeTarget = 44

depthLimit = 3
treeMaker = TreeMaker(depthLimit)

print("Accuracy rates for synthetic datasets:")
# for each of the synthetic datasets
for i in range(len(syntheticDataFiles)):
    syntheticData = importData(syntheticDataFiles[i])
    # create the tree 
    tree = treeMaker.ID3(syntheticData, syntheticTarget, syntheticFeatures, 0)
    # plot the decision surface
    plotDecisionSurface(tree, syntheticData, syntheticFeatures, syntheticTarget, i)
    # check accuracy
    score = 0
    for point in syntheticData:
        prediction = tree.predict(point)
        if prediction == point[syntheticTarget]:
            score = score + 1
    print(100*score/len(syntheticData))

pokeData = importData2(pokeDataFiles[0])
pokeDataTemp = importData3(pokeDataFiles[1])
pokeData = np.column_stack((pokeData, pokeDataTemp))

# create the tree
tree = treeMaker.ID3(pokeData, pokeTarget, pokeFeatures, 0)

# check accuracy
print("Accuracy rate for Pokemon dataset:")
score = 0
for point in pokeData:
    prediction = tree.predict(point)
    if prediction == point[pokeTarget]:
        score = score + 1
print(100*score/len(pokeData))

displayPlot()