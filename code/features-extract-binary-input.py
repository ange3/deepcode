################################
# Function: Main
# --------------
# Specifies feature extractors to be used on a given data set. 
# Creates a matrix of these features.
# --------------
# Return: Numpy matrix representing input vector
################################

if __name__ == '__main__':
  NUM_PROBLEMS = 9

  for problemNum in range(1,NUM_PROBLEMS)
    CORRECT_DATA_FILE = '../data-code-org/hoc' + problemNum + '/nextProblem/perfectSet.txt'
    ATTEMPT_DATA_FILE = '../data-code-org/hoc' + problemNum '/nextProblem/attemptSet.txt'

    print 'Extracting binary student correct/wrong: Problem ' + problemNum

    # Variables
    mapColumnIndexToStudentID = {}
    allStudentSet = set()
    attemptStudentSet = set()
    correctStudentSet = set()

    extractCorrectSet(CORRECT_DATA_FILE, ATTEMPT_DATA_FILE, attemptStudentSet, correctStudentSet, VERBOSE_FLAG)
