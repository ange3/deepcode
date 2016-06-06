import pickle
import csv
import random
import numpy

__author__ = 'Larry'

def openTrajectoriesCSV(hocNum):
    with open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/trajectories/Trajectory_ASTs_' + hocNum + '_new.csv', 'rb') as f:
        reader = csv.reader(f)
        trajectories = list(reader)
    return trajectories

def getCounts(hocNum):
    counts = {}
    with open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/trajectories/counts.txt', 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        for row in reader:
            counts[row[0]] = int(row[1])
    return counts

# Deprecated: k-fold validation code to generate train/test/val sets
#
# ##############################
# # Generates numFolds folds given a deck
# #
# #   numFolds - an integer
# #   deck - an foldedMatrix of the trajectory ids in sequence
# ##############################
# def generateFolds(deck, numFolds):
#     if numFolds <= 2:
#         print "Folds are too large, and too few folds. Use more folds"
#         return null
#
#     #Get sizes of each set
#     deckSize = len(deck)
#     foldSize = deckSize / float(numFolds)
#
#     folds = []
#     for foldNum in range(0,numFolds):
#         folds.append(generateFoldNum(deck, foldNum))
#     return
#
#
# ##############################
# # Generates foldNum-th fold in a deck
# #
# #   foldNum - fold number of the test set. Val set assumed to be the immediate prior set.
# #   foldSize - size of each fold
# #   deck - an foldedMatrix of the trajectory ids in sequence
# ##############################
# def generateFoldNum(deck, foldNum):
#     return


hocNums = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

def splitData(hocNum):
    trajectories = openTrajectoriesCSV(hocNum)
    counts = getCounts(hocNum)
    numTrajs = len(trajectories)

    deck = [] #Sequence of trajIDs for shuffling
    foldedMatrix = [] #TrajID, RowNum Train, Val, Train
    rowToTrajID = {} #Map to convert from row to trajID
    trajIDToRow = {} #Map to convert from trajID to row

    for row in range(0, numTrajs):
        currTraj = trajectories[row]
        currTrajID = currTraj[0]

        #How many times does this trajectory occur?
        currTrajFreq = counts[currTrajID]

        #Placeholder in foldedMatrix
        foldedMatrix.append([currTrajID,row,0,0,0])

        #Add this trajID to the deck the appropriate number of times
        deck.extend([currTrajID for i in range(currTrajFreq)])

        #Update conversion maps
        rowToTrajID[row] = currTrajID
        trajIDToRow[currTrajID] = row

    #Shuffle the deck
    random.shuffle(deck)

    deckSize = len(deck)
    numFolds = 8 # val and test sets are each 1/8th the total number of students
    foldSize = deckSize / float(numFolds)

    #Calculate starting indices for each set against the given deck size
    trainDeckEndIndex = int(foldSize * (numFolds-2))
    valDeckStartIndex = trainDeckEndIndex + 1
    valDeckEndIndex = int(valDeckStartIndex + foldSize)
    testDeckStartIndex = valDeckEndIndex + 1

    #Segment the deck into the subDecks
    trainDeck = deck[0:trainDeckEndIndex]
    valDeck = deck[trainDeckEndIndex:valDeckEndIndex]
    testDeck = deck[valDeckEndIndex:deckSize]

    #SubDeck sizes
    trainDeckSize = len(trainDeck)
    valDeckSize = len(valDeck)
    testDeckSize = len(testDeck)

    #Update the foldedMatrix (i.e. frequency counting)
    for i in range(0,trainDeckSize):
        foldedMatrix[trajIDToRow[trainDeck[i]]][2] += 1

    for i in range(0,valDeckSize):
        foldedMatrix[trajIDToRow[valDeck[i]]][3] += 1

    for i in range(0,testDeckSize):
        foldedMatrix[trajIDToRow[testDeck[i]]][4] += 1

    #Save to files
    foldedMatrixFilename = 'C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/trajectories/foldedMatrix_' + hocNum + '.npy'
    deckFilename = 'C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/trajectories/deckByTrajID_' + hocNum + '.npy'
    numpy.save(foldedMatrixFilename, foldedMatrix)
    numpy.save(deckFilename, deck)

    #Create deck by rowID
    deckByRowID = []
    for i in range(0,deckSize):
        deckByRowID.append(trajIDToRow[deck[i]])

    deckByRowIDFilename = 'C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/trajectories/deckByRowID_' + hocNum + '.npy'
    numpy.save(deckByRowIDFilename, deckByRowID)

for hocNum in hocNums:
    print "Processing hoc",hocNum
    splitData(hocNum)