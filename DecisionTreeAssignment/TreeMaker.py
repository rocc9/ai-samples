import numpy as np
from TreeNode import TreeNode

class TreeMaker(object):

    def __init__(self, d):
        self.maxDepth = d

    # returns false if there are any negative examples in e, true if there aren't
    def allPositive(self, e, t):
        for i in e:
            if i[t] == 0:
                return False 
        return True

    # returns false if there are any positive examples in e, true if there aren't
    def allNegative(self, e, t):
        for i in e:
            if i[t] == 1:
                return False 
        return True

    # returns most common class label in e. If the data is split 50-50, or e is empty, returns negative.
    def mostCommonLabel(self, e, t):
        pos = 0
        neg = 0
        for i in e:
            if i[t] == 1:
                pos = pos + 1
            elif i[t] == 0:
                neg = neg + 1
        if pos > neg:
            return 1 
        else:
            return 0

    # calculate entropy of set e with target attribute t
    def entropy(self, e, t):
        pos = 0
        neg = 0 
        total = 0
        # determine number of positive or negative examples
        for i in e:
            if i[t] == 1:
                pos = pos + 1
            elif i[t] == 0:
                neg = neg + 1
            total = total + 1
        # if all the entries are the same, then entropy is 0
        if pos == 0 or neg == 0:
            return 0
        # if there is uncertainty, use the formula to calculate entropy
        else:
            return -1*(pos/total)*np.log2(pos/total)-(neg/total)*np.log2(neg/total)

    # returns true if a given attribute in e is discrete, false otherwise 
    # all discrete data in the pokemon set is binary (0 or 1)
    def isDiscrete(self, e, a):
        observed = set()
        for i in e:
            if not i[a] in observed:
                observed.add(i[a])
        return len(observed) < 2

    # finds max and min value of the attribute a in e
    def findMaxMin(self, e, a):
        max = -10000
        min = 10000
        for ex in e:
            if ex[a] > max:
                max = ex[a]
            elif ex[a] < min:
                min = ex[a]
        return [min, max]

    # splits continuous data into n children based on a
    def split(self, e, a, n):
        result = []
        bounds = self.findMaxMin(e, a) 
        # data split into equidistant bins
        interval = (bounds[1]-bounds[0])/n 
        for i in range(n):
            currList = []
            for ex in e:
                if ex[a] >= bounds[0]+i*interval and ex[a] < bounds[0]+(i+1)*interval:
                    currList.append(ex)
                if i == n-1 and ex[a] == bounds[1]: 
                    currList.append(ex)
            result.append(currList)
        return result

    # find the best split and then return the feature and # of children 
    def findBestSplit(self, e, t, a):
        resultFeature = -1
        resultNum = -1
        bestInfoGain = -1
        startE = self.entropy(e, t)
        for att in a:
            # if the data is discrete there are only 2 possible values: 0 and 1
            if self.isDiscrete(e, att):
                splitData = self.split(e, att, 2)
                splitE = 0
                for i in splitData:
                    currE = self.entropy(i, t)
                    splitE = splitE + (len(i)/len(e))*currE
                infoGain = startE - splitE 
                if(infoGain > bestInfoGain):
                        bestInfoGain = infoGain
                        resultFeature = att
                        resultNum = 2
            # if the data is continuous, test different numbers of "buckets"
            else:
                for i in range(2, 6):
                    splitE = 0
                    splitData = self.split(e, att, i)
                    for j in splitData:
                        currE = self.entropy(j, t)
                        splitE = splitE + (len(j)/len(e))*currE
                    infoGain = startE - splitE 
                    if(infoGain > bestInfoGain):
                        bestInfoGain = infoGain
                        resultFeature = att
                        resultNum = i
        return [resultFeature, resultNum]

    # creates tree of maximum depth 3
    def ID3(self, examples, targetAttribute, attributes, depth):
        result = TreeNode()
        result.depth = depth
        # check if the exampls all have the same classification
        if self.allPositive(examples, targetAttribute):
            result.prediction = 1
            return result
        elif self.allNegative(examples, targetAttribute):
            result.prediction = 0
            return result
        # check if there are attributes remaining to split on, or if depth limit exceeded
        elif len(attributes) == 0 or depth == self.maxDepth:
            result.prediction = self.mostCommonLabel(examples, targetAttribute)
            return result;
        # if a tree has not been returned yet, determine best split
        else:
            splitResult = self.findBestSplit(examples, targetAttribute, attributes)
            result.feature = splitResult[0]
            numSplit = splitResult[1]
            # split the data 
            splitData = self.split(examples, result.feature, numSplit)
            bounds = self.findMaxMin(examples, result.feature) 
            interval = (bounds[1]-bounds[0])/numSplit
            # attributes.remove(result.feature)
            # add branches 
            for i in range(numSplit):
                val = bounds[0]+interval*(2*i+1)/2
                # if there are no examples in the i-th split, add a leaf
                if len(splitData[i]) == 0:
                    newNode = TreeNode()
                    newNode.prediction = self.mostCommonLabel(examples, targetAttribute)
                # if there are more examples, add a node 
                else:
                    result.children[val] = self.ID3(splitData[i], targetAttribute, attributes, depth + 1)
        return result