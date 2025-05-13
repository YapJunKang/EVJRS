import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(), 'systemData')
load_coefficient = pd.read_csv(os.path.join(data_dir, 'load_coefficient.csv')).to_numpy()

# generated between 0.9 to 1.1
mu, sigma, n = 1, 0.03, 200 # mu = mean, sigma = standard deviation, n = samples to be generated
x = mu + sigma*np.random.randn(n)
x = np.clip(x,0.9,1.1)

# n, bins, patches = plt.hist(x) # plot histogram of generated ratio

# generated samples
load = load_coefficient[:,1].reshape(1,-1) * x.reshape(n,-1)

# plot out generated samples
for i in range(n):
    plt.plot(load[i,:])
plt.show()

# save generated coefficients
dataframe = pd.DataFrame(load)
dataframe.to_csv(os.path.join(os.getcwd(), "dataGeneration/generatedLoad.csv"))
