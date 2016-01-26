# Contains feature extractor for each feature
# Usage: extract_features(feature options)

import numpy as np
import copy, random

# Test function
def extractFeature():
  print 'hello!'


################################
# Function: Generate Student IDs
# --------------
# Generates random unique numbers to add to an existing set of student IDs in order 
# to reach the specified number of elements in the set
# --------------
# Arguments:  Number of students, set of students, maxNumberStudents
# Return:     Set of student IDs
################################
def generateStudentIDs(numStudents, allStudentSet, attemptStudentSet, correctStudentSet, maxNumberStudents, verbose=False):
  print 'Generating additional students with no correct or attempted solution...'
  allStudentSet.update(correctStudentSet)
  allStudentSet.update(attemptStudentSet)
  numStudentsNeeded = numStudents - len(correctStudentSet) - len(attemptStudentSet)
  possibleIDs = set(range(1,maxNumberStudents))
  # Remove existing student IDs that are already correct or attempted from set of possible IDs
  possibleIDs -= correctStudentSet
  possibleIDs -= attemptStudentSet
  # Add generated IDs of no attempt students
  addIDs = random.sample(possibleIDs, numStudentsNeeded)
  allStudentSet.update(addIDs)
  if verbose:
    # print 'Number of Student IDs with correct solutions: {}'.format(len(correctStudentSet))
    # print 'Need {} students'.format(numStudentsNeeded)
    # print 'Number of Generated possible IDs: {}'.format(len(possibleIDs))
    print 'Number of student IDs added with no attempted solution: {}'.format(len(addIDs))
  print 'Total number of students: {}'.format(len(allStudentSet))

################################
# Function: Extract Correct Set
# --------------
# Extracts set of correct submission student IDs and set of attempted but incorrect submission
# student IDs from data files.
# --------------
# Arguments:  Name of data file
# Return:     Nothing (updates attemptStudentSet and correctStudentSet)
################################
def extractCorrectSet(correctDataFile, attemptDataFile, attemptStudentSet, correctStudentSet, verbose=False):
  print 'Extracting features from {}'.format(correctDataFile)
  correct_data_file = open(correctDataFile, 'r')
  for line in correct_data_file:
    id = int(line)
    correctStudentSet.add(id)
  print 'Extracting features from {}'.format(attemptDataFile)
  attempt_data_file = open(attemptDataFile, 'r')
  attemptsAndCorrectSet = set()
  for line in attempt_data_file:
    id = int(line)
    attemptsAndCorrectSet.add(id)
  attemptStudentSet = attemptsAndCorrectSet - correctStudentSet
  if verbose:
    print 'Num elements in correct student set: {}'.format(len(correctStudentSet))
    print 'Num elements in attempted but incorrect student set: {}'.format(len(attemptStudentSet))
  
  
  

################################
# Function: One Hot Encoding for Correct and Wrong Submissions
# --------------
# Create a matrix representing the one-hot encodings of correct and wrong submissions 
# of N students over M problems. We have 3M columns (problems) and N rows (students).
# In the first set of M columns in the matrix, a 1 is placed 
# in the ith row for a student with a CORRECT submission for the ith problem. 
# The same is done for the second set of M columns, but for WRONG and ATTEMPTED submissions.
# Finally, the last set of M columns is for UNATTEMPTED submissions.
#
# NOTE 1: Rows = Students and Columns = Problems
# Rows and columns are indexed at 0, row and column indices are adjusted so that the first row
# correspond to the first student and the first column corresponds to the first problem, and so on.
# NOTE 2: We use a map to keep track of column indices to student IDs since student IDs are not necessarily continuous.
# --------------
# Arguments:  number of problems, problem ID for student ID sets, set of student IDs with correct submissions, set of all student IDs, map {columnIndex: studentID}
# Return:     N x 3M matrix (numpy)
################################
def oneHotCorrectAttemptedNotAttempted(numProblems, problemID, allStudentSet, attemptStudentSet, correctStudentSet, mapColumnIndexToStudentID, verbose=False):
  print 'Extracting one-hot encoding: Correct / Attempted but Wrong / Not Attempted'
  numStudents = len(allStudentSet)
  encodings = np.zeros((numStudents, numProblems*3))
  row_index = 0
  for studentID in allStudentSet:
    mapColumnIndexToStudentID[row_index] = studentID
    # Determine in which set of problems to place this student
    if studentID in correctStudentSet:
      column_index = problemID-1
    elif studentID in attemptStudentSet:
      column_index = numProblems+problemID-1
    else:
      column_index = numProblems*2+problemID-1
    # Add 1 to appropriate place in encodings matrix
    encodings[row_index, column_index] = 1
    row_index += 1
  show_num_students = 10
  print 'Encodings Data Matrix: {}'.format(encodings.shape)
  if verbose:  
    print 'We have {} column to student id mappings'.format(len(mapColumnIndexToStudentID))
    print 'Printing a slice of the encodings of the first {} students'.format(show_num_students)
    slice = encodings[:show_num_students, :]
    print slice.shape
    print slice
    print 'Student IDs of first {} students are:'.format(show_num_students)
    for i in xrange(show_num_students):
      studentID = mapColumnIndexToStudentID[i]
      print 'Student ID {}: Correct? {} Attempted? {}'.format(studentID, studentID in correctStudentSet, studentID in attemptStudentSet)
  return encodings

################################
# Function: Main
# --------------
# Specifies feature extractors to be used on a given data set. 
# Creates a matrix of these features.
# --------------
# Return: Numpy matrix representing input vector
################################
if __name__ == '__main__':
  # TEST_DATA_SET = 'testPerfectSet.txt'
  # NUM_PROBLEMS = 7
  # TEST_CORRECT_STUDENT_SET = {3,5,7,10}
  # TEST_ALL_STUDENT_SET = set(range(1,NUM_STUDENTS+1))
  # PROBLEM_ID = 5
  # encoding = oneHotCorrectWrong(NUM_PROBLEMS, PROBLEM_ID, TEST_CORRECT_STUDENT_SET, TEST_ALL_STUDENT_SET)

  # ACTUAL DATA SET
  CORRECT_DATA_FILE = '../data-code-org/hoc4/nextProblem/perfectSet.txt'
  ATTEMPT_DATA_FILE = '../data-code-org/hoc4/nextProblem/attemptSet.txt'
  NUM_PROBLEMS = 20
  PROBLEM_ID = 5
  NUM_STUDENTS = 509405 # from code.org data's readme.txt
  MAX_NUMBER_STUDENTS = 999999
  VERBOSE_FLAG = True

  print 'INFO: Looking at {} students answering problem # {} out of {} problems.'.format(NUM_STUDENTS, PROBLEM_ID, NUM_PROBLEMS)

  # Variables
  mapColumnIndexToStudentID = {}
  allStudentSet = set()
  attemptStudentSet = set()
  correctStudentSet = set()

  extractCorrectSet(CORRECT_DATA_FILE, ATTEMPT_DATA_FILE, attemptStudentSet, correctStudentSet, VERBOSE_FLAG)
  generateStudentIDs(NUM_STUDENTS, allStudentSet, attemptStudentSet, correctStudentSet, MAX_NUMBER_STUDENTS, VERBOSE_FLAG)
  encoding = oneHotCorrectAttemptedNotAttempted(NUM_PROBLEMS, PROBLEM_ID, allStudentSet, attemptStudentSet, correctStudentSet, mapColumnIndexToStudentID, VERBOSE_FLAG)

