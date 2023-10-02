class TreeNode(object):
    # when a TreeNode is first created, everything is empty
    def __init__(self):
        self.feature = -1
        self.prediction = -1
        self.children = dict() 
        self.depth = -1

    def __str__(self):
        result = "Feature: " + str(self.feature) + "\nDepth: " + str(self.depth) + "\nPrediction: " + str(self.prediction) + "\nChildren :\n"
        for i in self.children:
            result = result + "val: " + str(i) + "\n"
            print(self.children[i])
        return result

    # classify a given datapoint 
    def predict(self, dataPoint):
        # if we are at a leaf, return prediction
        if self.feature == -1:
            return self.prediction 
        else:
            minDistance = 10000
            closestChild = -1
            # find the closest child to dataPoint 
            for i in self.children:
                distance = abs(dataPoint[self.feature] - i)
                if distance < minDistance:
                    minDistance = distance
                    closestChild = i 
            # recursively classify the data point using the nearest child node
            return self.children[closestChild].predict(dataPoint)
