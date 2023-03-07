# trainingSet.py
# CE '23
# 01/31/2022
# create/add data to the NN training set
# USE THIS SCRIPT TO ADD DATA TO TRAINING SET

# needed for matrix stuff
import numpy as np

# needed for esp32
import serial

# needed for trainingSet functions
import trainingScript as ts

# define esp32
esp = serial.Serial(
        port = '/dev/ttyUSB0',
    baudrate = 115200,
    timeout = 5,
    writeTimeout = 2
)

# setup esp
ts.setup(esp)

# trials we want to add to the training set
m=10; 

# allocate newData array -> m col, 7 rows
newData=np.zeros( (m, 7) )

# fill newData vector with result from training script
i=0
while i<m:
    # fill data into newData matrix
    newData[i]=ts.trial(esp)
    #print(newData[i])
    # wait for new data before entering again
    var=input("press enter to continue to next trail, -1 to end \nif ending, need to append trainingSet.txt before continuing: ")
    if var=='-1':
        break
    else:
        i+=1

# read data from *.dat file
oldData = np.loadtxt('trainingSet.txt')

# add oldData and newData together to form a new matrix
print("Old:")
print(oldData)
print("New:")
print(newData)
dataOut = np.vstack((oldData,newData))

# write data to *.dat file
dataOut = np.matrix(dataOut)
with open('trainingSet.txt', 'wb') as f:
    for line in dataOut:
        np.savetxt(f, line, fmt='%.2f')
