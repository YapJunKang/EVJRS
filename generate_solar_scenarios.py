"""
This script is for generating solar scenarios based on each timestep. There is a total of 48 timesteps of 0.5 hour intervals for each solar scenario.

This script includes a scenario reduction step, where the number of scenarios is reduced from 100 to 10. The scenario reduction algorithm used is fast forward selection.

We generate a total of 1000 samples of scenarios and probabilities, and save them into a pickle file.

This script takes in a command line argument for the number of scenarios to be generated. If no argument is given, the default number of scenarios is 100.
"""


import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime
import pickle 

import scenred.scenario_reduction as scen_red

if __name__ == "__main__":
    filename = "systemData/Irradiance_39.xlsx"

    # takes in command line argument for nScenarios

    nScenarios = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_sc_red = 5
    # read data from excel file, and from columns A to E
    df = pd.read_excel(filename, index_col=0, usecols="A:E")

    # drop the "Ghi Prev Day" column
    df.drop("Ghi Prev Day", axis=1, inplace=True)

    # group the data by season and separate them into different dataframes
    df_1 = df[df["Season"] == 1]
    df_2 = df[df["Season"] == 2]
    df_3 = df[df["Season"] == 3]

    # drop the "Season" column
    df_1.drop("Season", axis=1, inplace=True)
    df_2.drop("Season", axis=1, inplace=True)
    df_3.drop("Season", axis=1, inplace=True)

    # normalize Ghi Curr Day to between 0 and 1, record the min and max values for scaling
    min_irradiance = df["Ghi Curr Day"].min()
    max_irradiance = df["Ghi Curr Day"].max()
    df["Ghi Curr Day"] = (df["Ghi Curr Day"] - df["Ghi Curr Day"].min()) / (df["Ghi Curr Day"].max() - df["Ghi Curr Day"].min())

    # %%
    #  for each group, fit a beta distribution to the data
    # for name, group in df.groupby("Time"):
    #     # clear a, b , loc, scale
    #     a, b, loc, scale = None, None, None, None
    #     max_x = max(group["Ghi Curr Day"])
    #     std_x = np.std(group["Ghi Curr Day"])
    #     x = np.linspace(0, max_x + 1*std_x, 2000)
    #     # if the group contains only zeros, skip it
    #     if group["Ghi Curr Day"].sum() > 0:
    #         # fit a beta distribution to the data

    #         a, b, loc, scale = scipy.stats.beta.fit(group["Ghi Curr Day"]) 


    # %%
    #  for each group, if a beta distribution is fitted, set fitted flag of dictionary to True
    fit_dict = {}
    for name, group in df.groupby("Time"):
        fit_dict[name] = {}
        # clear a, b , loc, scale
        a, b, loc, scale = None, None, None, None
        max_x = max(group["Ghi Curr Day"])
        min_x = min(group["Ghi Curr Day"])
        std_x = np.std(group["Ghi Curr Day"])
        x = np.linspace(0, max_x + 1*std_x, 2000)
        # normalize group["Ghi Curr Day"] which is the irradiance
        # normalized_irradiance = (group["Ghi Curr Day"] - min_x) / (max_x - min_x)

        # if the group contains only zeros, skip it
        if group["Ghi Curr Day"].sum() > 0:
            # fit a beta distribution to the data
            a, b, loc, scale = scipy.stats.beta.fit(group["Ghi Curr Day"], floc=0)
            # set fitted flag to True
            fit_dict[name]["fit"] = True
            # get parameters of beta distribution into the dictionary
            fit_dict[name]["params"] = [a, b, loc, scale]
        else:
            fit_dict[name]["fit"] = False
            fit_dict[name]["params"] = [None, None, None, None]


    T_irradiance_list = [
        group["Ghi Curr Day"].values for name, group in df.groupby("Time")
    ]
    # turn T_irradiance list into a 2d array 
    T_irradiance = np.array(T_irradiance_list)

    # check the maximum difference between irradiance of subsequent time intervals
    max_diff_dict = {}
    max_diff = 0
    for i in range(47):
        max_diff = max(T_irradiance[i+1] - T_irradiance[i])
        # print("max_diff: ", max_diff, "at i: ", i)
        max_diff_dict[i] = max_diff


    #%%
    scenarios_list = []
    probabilities_list = []

    # number of samples to be generated
    nSamples = 4

    # %%

    for i in range(nSamples):

        # for each time interval, use the corresponding distribution to sample 10 points, equivalent to 10 scenario
        # Add the samples into a numpy array
        scenarios = np.zeros((48, nScenarios)) # 48 time intervals, 100 scenarios
        
        for i, (name, group) in enumerate(df.groupby("Time")):
        
            # if the group contains only zeros, skip it
            if group["Ghi Curr Day"].sum() > 0:
                # get the parameters of the fitted distribution
                a, b, loc, scale = fit_dict[name]["params"]
                # sample 10 points from the distribution
                samples = scipy.stats.beta.rvs(a, b, loc, scale, size=nScenarios)

                # if the difference in irradiance between previous time step and current timestep is greater than 0.44
                # resample the respective points until the difference is less than 0.44
                while(np.max(samples - scenarios[i-1]) > max_diff_dict[name]):
                    # get the indices of the points that need to be resampled
                    idx = np.where(samples - scenarios[i-1] > max_diff_dict[name])[0]
                    # resample the points
                    samples[idx] = scipy.stats.beta.rvs(a, b, loc, scale, size=len(idx))
                    
            else:
                # sample zeros instead
                samples = np.zeros(nScenarios)
            # add the samples to the scenarios array
            scenarios[i] = samples
        # scale the scenarios to the original irradiance range, and convert to kW by multiplying 0.0005
        # for i in range(48):
        #     scenarios[i] = scenarios[i] * (max_irradiance - min_irradiance) + min_irradiance

        prob = np.ones((nScenarios,))*(1/nScenarios)


    #     S = scen_red.ScenarioReduction(scenarios, probabilities=None, cost_func='General', r = 2)
    #     S.fast_forward_sel(n_sc_red=n_sc_red, num_threads = 4)  # use fast forward selection algorithm to reduce to 5 scenarios with 4 threads 
    #     scenarios_reduced = S.scenarios_reduced  # get reduced scenarios
    #     probabilities_reduced = S.probabilities_reduced  # get reduced probabilities

        scenarios_list.append(np.transpose(scenarios))
        probabilities_list.append(prob)

    # save the scenarios and probabilities into a dictionary
    scenarios_dict = {
        "scenarios": scenarios_list,
        "probabilities": probabilities_list,
    }
    # save the dictionary into a pickle file
    with open(f"systemData/{nSamples}_samples_{n_sc_red}_scenario.pkl", "wb") as f:
        pickle.dump(scenarios_dict, f)
