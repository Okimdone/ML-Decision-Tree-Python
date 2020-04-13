import pandas as pd
import math
import numpy as np

class Node:
    def __init__(self, trainingData, maxDepth):
        self.label = None # The attribute's name decided upon
        self.outputColumn = trainingData.columns[-1]
        self.trainingData = trainingData
        self.maxDepth = maxDepth 
        self.children = {}

    '''
    S : is a Series
    '''
    @staticmethod
    def selfEntropy(S): # S is a pandas::Series 
        value_probabilies = S.value_counts() / len(S)
        return - ( value_probabilies * value_probabilies.apply(math.log2) ).sum()

    ''' 
    X_Y : a Dataframe of lenght of 2 
    it MUST be of the form ('X', 'Y') 
    -> returns the conditional entropy H(Y | X)  using : 
        sum(p(x and y)* log(p(x) / p(x and y)) )
    '''
    @staticmethod
    def conditionalEntropy(X_Y:list): 
        # Series of the form (index = (X,Y), value = P(x and y) )
        probability_XandY = X_Y.groupby(list(X_Y.columns)).size() / len(X_Y) 
        probability_X = probability_XandY.groupby(level=0).sum()
        return sum( (probability_XandY[(x, y)] * math.log2( probability_X[x] / probability_XandY[(x, y)]) ) 
                    for x, y in probability_XandY.index )

    def trainModel(self):
        selfEntropy_Y = self.selfEntropy(self.trainingData[ self.outputColumn ])
        if selfEntropy_Y == 0:
            self.label = self.trainingData[ self.outputColumn ][0]
            return 
        elif len(self.trainingData.columns) == 1 or self.maxDepth <= 0:
            self.label = self.trainingData[ self.outputColumn ].value_counts().index[0]
            return 

        colums = self.trainingData.columns[:-1]
        temp_dict = dict()
        for featureName in colums:
            conditionalEntropy_Y_X = self.conditionalEntropy( self.trainingData[[featureName, self.outputColumn]] )
            temp_dict[conditionalEntropy_Y_X] = featureName
        else :
            self.label = temp_dict[ min(temp_dict) ]

        newSubTrainingSet = self.trainingData.set_index( self.label )
        for attribute in set(newSubTrainingSet.index):
            self.children[ attribute ] = Node( newSubTrainingSet.loc[[attribute]].reset_index(drop=True), self.maxDepth - 1)
            self.children[ attribute ].trainModel()

    def runModel(self, testData):
        l = []
        heightOfFrame = testData.shape[0]
        for i in range(heightOfFrame):
            l.append( self.testOnLinedFrame( testData.iloc[i] ) )
        r = pd.DataFrame(l, index=testData.index, columns=['Survived'])
        #r.index = testData.index
        return r 
            
    def testOnLinedFrame(self, dataFrameLine):
        if len(self.children) == 0 :
            return self.label
        try :
            return self.children[ dataFrameLine[self.label] ].testOnLinedFrame(dataFrameLine)
        except KeyError:
            voteCounter = dict()
            for prediction in (node.testOnLinedFrame(dataFrameLine) for node in self.children.values()):
                if prediction not in voteCounter:
                    voteCounter[prediction] = 1
                else : 
                    voteCounter[prediction] += 1
            return  max(voteCounter, key=voteCounter.get)

    def printTree(self, level=0):
        if len(self.children) == 0:
            print(f'{"    "*(level)}return {repr(self.label)}')
        else :
            for i, child in enumerate(self.children) :
                print(f'{"|   "*level}{"if" if i==0 else "elif"} {self.label} == {repr(child)} :')
                self.children[child].printTree(level = level + 1)

            
class DecisionTree:
    def __init__(self, trainingData, maxDepth):
        self.trainingData = trainingData
        self.maxDepth = maxDepth
        self.root = Node(self.trainingData, maxDepth)

    def trainModel(self):
        self.root.trainModel()

    def runModel(self, testingData):
        return self.root.runModel(testingData)
    
    def printTree(self):
        self.root.printTree(level=0)


if __name__ == "__main__":
    import sys
        
    #data = pd.read_csv(sys.argv[1])
    #testData = pd.read_csv(sys.argv[2])
    #maxDepth = int(sys.argv[3])

    data = pd.read_csv("titanic_cleaned/cleaned_training_data.csv")
    data.set_index('PassengerId', inplace=True)
    maxDepth = 9

    tree = DecisionTree(data, maxDepth)
    tree.trainModel()

    testData = pd.read_csv("titanic_cleaned/cleaned_testing_data.csv" )#.set_index(['PassengerId'])
    testData.set_index('PassengerId', inplace=True)
    predicted = tree.runModel(testData)
    
    
    print(predicted)
    testing_labels = pd.read_csv('gender_submission.csv')#['Survived']
    testing_labels.set_index('PassengerId', inplace=True)
    #testData['Survived'] 

    predicted.to_csv('titanic_subm.csv')
    
    tree.printTree()
    print((testing_labels == predicted).sum()/ len(testing_labels))
