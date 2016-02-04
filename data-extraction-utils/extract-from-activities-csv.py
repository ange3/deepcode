#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: extract-from-activities.py
# @Author: Angela Sy
# @created: Feb 04 2016
#
#==============================================================================
# DESCRIPTION: Allows us to extract the student IDs of students who answered 
#   each problem correctly and students who answered incorrectly
#   Saves info for every problem in CSV files and saves overall stats in info.txt
# 
#==============================================================================
# CURRENT STATUS: Not modularized into separate functions yet.
# Note: Right now the uncommented code generates the info.txt file from 
#   already created csv files
#==============================================================================
# USAGE:
# run in Terminal:  python extract-from-activities-csv.py
#==============================================================================
#
###############################################################################

import numpy as np
import csv

DATA_FOLDER = '/Volumes/ANGELA SY/From Chris - senior project knowledge tracing/hoc1-9'
NUM_PROBLEMS = 9


if __name__ == '__main__':
  mapProblemToCorrectDict = {}
  # extract each problem's set of correct student IDs and save in the dict
  f = open('results/info.txt', "r+b")
  for problemNum in range(1,NUM_PROBLEMS+1):
    # correctSet = set()
    # attemptSet = set()
    # activities_filename = DATA_FOLDER + '/hoc' + str(problemNum) + '/dumps/activities.csv'
    # print 'Loading data from: {}'.format(activities_filename)
    # data_array = np.array(list(csv.reader(open(activities_filename,"rb"),delimiter=',')))
    # for line in data_array:
    #   user_id = line[1]
    #   test_score = line[9]
    #   if test_score == "100":
    #     correctSet.add(user_id)
    #   else:
    #     attemptSet.add(user_id)
    # mapProblemToCorrectDict[problemNum] = correctSet

    # Save each problem's correct set and attempt set in individual files
    output_folder = 'results/' + str(problemNum) + '_'
    output_correct_filename = output_folder + 'perfectSet.csv'
    output_attempt_filename = output_folder + 'attemptSet.csv'

    # with open(output_correct_filename, 'w') as fp:
    #   a = csv.writer(fp, delimiter=',')
    #   a.writerow(list(correctSet))

    # with open(output_attempt_filename, 'w') as fp:
    #   a = csv.writer(fp, delimiter=',')
    #   a.writerow(list(attemptSet))

    # print 'Saved file: {}'.format(output_correct_filename) 
    # print 'Saved file: {}'.format(output_attempt_filename) 
    # print 'Size of Attempt Set: {}, Correct Set: {}'.format(len(attemptSet), len(correctSet))

    # Checking if saved properly
    correct_array = np.array(list(csv.reader(open(output_correct_filename,"rb"),delimiter=',')))
    attempt_array = np.array(list(csv.reader(open(output_attempt_filename,"rb"),delimiter=',')))

    # print 'Checking saved CSV file...'
    # if (len(attemptSet) != attempt_array.shape[1]):
    #   print 'WATCH OUT: Did not save list of attempt student IDs properly'
    # if (len(correctSet) != correct_array.shape[1]):
    #   print 'WATCH OUT: Did not save list of correct student IDs properly'

    # Printing info to text file
    problemInfoStr = 'Problem ' + str(problemNum) + '\n'
    students_correct = correct_array.shape[1]
    students_attempt_wrong = attempt_array.shape[1]
    students_total = students_correct + students_attempt_wrong
    countStatsStr = 'Total Attempted: ' + str(students_total) + ', Attempted but Incorrect: ' + str(students_attempt_wrong) + ', Correct: ' + str(students_correct) + '\n'
    percentCorrStatsStr = 'Percentage of Correct/Total Attempted: ' + str(students_correct/float(students_total)) + '\n'
    percentAttemptStatsStr = 'Percentage of Attempted Incorrect/Total Attempted: ' + str(students_attempt_wrong/float(students_total)) + '\n\n'

    f.write(problemInfoStr)
    f.write(countStatsStr)
    f.write(percentCorrStatsStr)
    f.write(percentAttemptStatsStr)


  # ENCODE AS MATRIX
  # figure out size of entire student set
  # allStudentSet = set()
  # for problemNum in range(1,NUM_PROBLEMS+1):
  #   allStudentSet.update(mapProblemToCorrectDict[problemNum])
  # num_students = len(allStudentSet)
  # print 'Total number of students: ', num_students

  # # Encode all problems in N x P matrix
  # # 0 = wrong, 1 = correct
  # # Note: row_index is specified in userIdToRowMap and column_index is problemID - 1
  # encoding = np.zeros((num_students, NUM_PROBLEMS), dtype=np.int)
  # userIdToRowMap = {}
  # next_row_index = 0
  # for problemNum in range(1,NUM_PROBLEMS+1):
  #   correctSet = mapProblemToCorrectDict[problemNum]
  #   for userID in correctSet:
  #     if userID not in userIdToRowMap:
  #       userIdToRowMap[userID] = next_row_index
  #       next_row_index += 1
  #     row_index = userIdToRowMap[userID]
  #     column_index = problemNum-1
  #     encoding[row_index, column_index] = 1  # mark this problem for this student as correct

  # print encoding[:5,:]

  # # Output to CSV file
  # np.savetxt("hoc_1-9_binary_input.csv", encoding, fmt="%u", delimiter=',')


