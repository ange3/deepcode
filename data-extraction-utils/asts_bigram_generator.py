import numpy as np
import pickle
import csv
import operator

#mat = np.load('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/Processed_ASTs/Processed_ASTs/traj_matrix_1.npy')

#pck = pickle.load(open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/Processed_ASTs/Processed_ASTs/map_ast_row_1.pickle','rb'))

hocNum = "1"

def weighted(hocNum):
    trajectories = openTrajectoriesCSV(hocNum)
    counts = getCounts(hocNum)
    bigrams = generateBigramPerTrajectory(trajectories, counts)
    sorted_mB = coalesceBigrams(bigrams)
    guesses = createGuesses(sorted_mB)
    return

def unweighted(hocNum):
    trajectories = openTrajectoriesCSV(hocNum)
    counts = getCounts(hocNum)
    bigrams = generateUnweightedBigramPerTrajectory(trajectories)
    sorted_mB = coalesceBigrams(bigrams)
    guesses = createGuesses(sorted_mB)
    return

def openTrajectoriesCSV(hocNum):
    with open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/trajectories/Trajectory_ASTs_' + hocNum + '.csv', 'rb') as f:
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

def generateBigramPerTrajectory(trajectories, counts):
    bigrams = {}
    for row in trajectories:
        # print row
        trajID = row[0]
        coeff = counts[trajID]
        # print trajID, coeff
        currAST = row[1]
        bigram = {}
        for i in range(2, len(row)):
            nextAST = row[i]
            if currAST != nextAST:
                currBigram = (currAST, nextAST)
                #print currBigram
                if bigram.has_key(currBigram):
                    bigram[currBigram] += coeff
                else:
                    bigram[currBigram] = coeff
                currAST = nextAST
        bigrams[trajID] = bigram
    return bigrams

def generateUnweightedBigramPerTrajectory(trajectories):
    bigrams = {}
    for row in trajectories:
        # print row
        trajID = row[0]
        currAST = row[1]
        bigram = {}
        for i in range(2, len(row)):
            nextAST = row[i]
            if currAST != nextAST:
                currBigram = (currAST, nextAST)
                #print currBigram
                if bigram.has_key(currBigram):
                    bigram[currBigram] += 1
                else:
                    bigram[currBigram] = 1
                currAST = nextAST
        bigrams[trajID] = bigram
    return bigrams

def coalesceBigrams(bigrams):
    masterBigrams = {}
    for trajID in bigrams.keys():
        bigramSet = bigrams.get(trajID)
        for gram in bigramSet.keys():
            if masterBigrams.has_key(gram):
                masterBigrams[gram] += bigramSet.get(gram)
            else:
                masterBigrams[gram] = bigramSet.get(gram)
        # print
    sorted_mB = sorted(masterBigrams.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_mB
# print sorted_mB

def createGuesses(sorted_mB):
    guesses = {}
    for i in range(0,len(sorted_mB)):
        first = sorted_mB[i][0][0]
        if not guesses.has_key(first):
            guesses[first] = sorted_mB[i][0][1]
            #print first, sorted_mB[i][0][1], sorted_mB[i][1]
    print guesses
    return guesses

def writeToFile(hocNum, guesses):
    writer = csv.writer(open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/trajectories/AST_guesses_weighted_' + hocNum + '.csv', 'wb'))
    for key, value in guesses.items():
       writer.writerow([key, value])
    return

weighted(hocNum)
unweighted(hocNum)