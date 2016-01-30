import numpy as np
import csv

DATA_FOLDER = '/Volumes/ANGELA SY/From Chris - senior project knowledge tracing/hoc1-9'
NUM_PROBLEMS = 9

if __name__ == '__main__':
  mapProblemToCorrectDict = {}
  # extract each problem's set of correct student IDs and save in the dict
  for problemNum in range(1,NUM_PROBLEMS+1):
    correctSet = set()
    attemptSet = set()
    activities_filename = DATA_FOLDER + '/hoc' + str(problemNum) + '/dumps/activities.csv'
    print activities_filename
    data_array = np.array(list(csv.reader(open(activities_filename,"rb"),delimiter=',')))
    for line in data_array:
      user_id = line[1]
      test_score = line[9]
      if test_score == "100":
        correctSet.add(user_id)
      else:
        attemptSet.add(user_id)
    mapProblemToCorrectDict[problemNum] = correctSet

  # figure out size of entire student set
  allStudentSet = set()
  for problemNum in range(1,NUM_PROBLEMS+1):
    allStudentSet.update(mapProblemToCorrectDict[problemNum])
  num_students = len(allStudentSet)
  print 'Total number of students: ', num_students

  # Encode all problems in N x P matrix
  # 0 = wrong, 1 = correct
  # Note: row_index is specified in userIdToRowMap and column_index is problemID - 1
  encoding = np.zeros((num_students, NUM_PROBLEMS), dtype=np.int)
  userIdToRowMap = {}
  next_row_index = 0
  for problemNum in range(1,NUM_PROBLEMS+1):
    correctSet = mapProblemToCorrectDict[problemNum]
    for userID in correctSet:
      if userID not in userIdToRowMap:
        userIdToRowMap[userID] = next_row_index
        next_row_index += 1
      row_index = userIdToRowMap[userID]
      column_index = problemNum-1
      encoding[row_index, column_index] = 1  # mark this problem for this student as correct

  print encoding[:5,:]

  # Output to CSV file
  np.savetxt("hoc_1-9_binary_input.csv", encoding, fmt="%u", delimiter=',')

  # If we want to save each problem's correct set and attempt set in individual files
  # Within for loop for every problem as it's being read (first for loop)
    # output_folder = 'results/' + str(problemNum) + '/'
    # output_correct_filename = output_folder + 'perfectSet.csv'
    # output_attempt_filename = output_folder + 'attemptSet.csv'

    # with open(output_correct_filename, 'w') as fp:
    #   a = csv.writer(fp, delimiter=',')
    #   a.writerow(list(correctSet))

    # with open(output_attempt_filename, 'w') as fp:
    #   a = csv.writer(fp, delimiter=',')
    #   a.writerow(list(attemptSet))

    # print 'Attempt Set: ', len(attemptSet)
    # print 'Correct Set: ', len(correctSet)

