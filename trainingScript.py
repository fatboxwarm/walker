# im just a function

# needed for matrix stuff
import numpy as np

def setup(dev):
    # establish connection to esp32
    try:
        dev.write(32)
        data = dev.readline()
        if data:
            print("connection to ESP32 successful")
            return 1
        else:
            print("failed setup")
    except Exception as e:
            print("connection failed.")
            dev.close()
            return 0

def trial(dev):
    i=0
    m=6
    
    # clear data
    data=np.zeros( (1,7) )

    # gather nn input data
    dev.write(1) # lets esp know we want to receive data
    while i<m:
        data[0,i]=dev.read_until(b'\n')
        print(data[0,i])
        i += 1

    # user input nn output data
    data[0,6] = input("User Fall: ")
    return np.matrix(data)



