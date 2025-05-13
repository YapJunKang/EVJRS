import numpy as np
import pandas as pd
import random
import os

timespan = 30 # minutes
nbEV = 25
n = 25
s_d_pair = [(19,25), (19,23), (3,25), (3,23)]
station_shift_1 = [6, 32, 104, 106, 107]
station_shift_2 = [101, 102, 103, 105]
shift_1 = list(np.arange(6/(timespan/60), 12/(timespan/60), 1)) # from 6am to 12pm
shift_2 = list(np.arange(14/(timespan/60), 20/(timespan/60), 1)) # from 2pm to 10pm

samples = [] # list of generated samples
for i in range(n):
    
    new_sample = np.ones((nbEV*4,3))

    for k in range(nbEV):
        s_d = random.choice(s_d_pair)
        time1 = random.choice(shift_1)
        time2 = random.choice(shift_2) 

        if s_d[0] == 19:
            station1 = random.choice(station_shift_1)
            station2 = random.choice(station_shift_2)
        elif s_d[0] == 3:
            station1 = random.choice(station_shift_2)
            station2 = random.choice(station_shift_1)

        new_sample[k*4:k*4+4, 0] = k

        new_sample[k*4, 1] = s_d[0]
        new_sample[k*4+1, 1] = station1
        new_sample[k*4+2, 1] = station2
        new_sample[k*4+3, 1] = s_d[1]

        new_sample[k*4, 2] = 0
        new_sample[k*4+1, 2] = time1
        new_sample[k*4+2, 2] = time2
        new_sample[k*4+3, 2] = 47
    
    new_sample = new_sample.astype("float16")

    if any(np.array_equal(new_sample, i) for i in samples):
        print("Duplicated!!!")
    else:
        
        samples.append(new_sample)

    # input("Press Enter to continue...")
samples = np.array(samples)

for id, i in enumerate(range(samples.shape[0])):
    schedule = samples[i,:,:]
    dataframe = pd.DataFrame(schedule)
    dataframe.to_csv(os.path.join(os.getcwd(), f"dataGeneration\EV_Schedule\EV_schedules{id}.csv"), index=False)