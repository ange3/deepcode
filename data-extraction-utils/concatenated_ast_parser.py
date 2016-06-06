__author__ = 'Larry'

import json
import csv
from pprint import pprint

hocNum = "9"

def getCounts(hocNum):
    counts = {}
    with open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/asts/counts.txt', 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        for row in reader:
            counts[row[0]] = int(row[1])
    return counts

def getFilelist(hocNum):
    filelist = []
    with open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/asts/filelist.txt', 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        for row in reader:
            filelist.append(row[0].replace('.json',''))
    return filelist

with open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/asts/concatenated_ASTs.json') as data_file:
    data = json.load(data_file)

numASTs = len(data)
counts = getCounts(hocNum)
filelist = getFilelist(hocNum)
outputLines = []

#print filelist

def innerScope(parent):
    children = parent[u'children']
    for child in range(0, len(children)):
        return

def processChild(child):
    if u'children' not in child:
        return "," + str(child[u'type'])
    grandchildren = child[u'children']
    numGrandchildren = len(child[u'children'])
    type = str(child[u'type'])
    if type != "DO":
        line = "," + str(child[u'type'])
        for grandchildIndex in range(0, numGrandchildren):
            line += processChild(child[u'children'][grandchildIndex])
        line += ",end_loop"
    else:
        line = ""
        for grandchildIndex in range(0, numGrandchildren):
            line += processChild(child[u'children'][grandchildIndex])
    return line

def processChildWrapper(child):
    if u'children' not in child:
        return "," + str(child[u'type']) + ",end_program"
    grandchildren = child[u'children']
    numGrandchildren = len(child[u'children'])
    line = "," + str(child[u'type'])
    for grandchildIndex in range(0, numGrandchildren):
        line += processChild(child[u'children'][grandchildIndex])
    line += ",end_program"
    return line

# for index in range(0, len(data)):
#     line = ""
#     line += str(index) #Make first item the astID
#     line += "," + str(counts[index])
#     children = data[index][u'children']
#     for child in range(0, len(children)):
#         if u'children' in child:
#             line += innerscope(child)
#         else:
#             line += "," + str(children[child][u'type'])
#
#     print line

def writeToFile(hocNum, outputLines):
    file = open('C:/Users/Larry/Dropbox/Stanford University/2015-2016/Winter Quarter/CS191W/hoc1-9_new/hoc' + hocNum + '/asts/AST_to_blocks_' + hocNum + '.csv', 'wb')
    for line in outputLines:
        #print line
        file.write(line + '\n')
    return

for rowIndex in range(0, numASTs):
    line = ""
    astID = filelist[rowIndex]
    line += astID #Make first item the astID
    line += "," + str(counts[astID])
    #print astID
    # children = data[rowIndex][u'children'] #index into the rowIndex-th json object
    # numChildren = len(children)
    # for childIndex in range(0, numChildren):
    #     line += processChild(children[childIndex])
    line += processChildWrapper(data[rowIndex])

    #print line

    outputLines.append(line)

writeToFile(hocNum, outputLines)

#pprint(data)

