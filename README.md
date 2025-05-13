# Electric Vehicle Joint Routing and Scheduling (EVJRS)
This repository contains code for reproducing the experiments in the paper titled "Joint Optimisation of Electric Vehicle Routing and Scheduling: A Deep Learning-Driven Approach for Dynamic Fleet Sizes" that has been accepted for publication in IJCNN 2025. The code is for training a CNN to assist Gurobi in solving the EVJRS problem.

# Gurobi License
Although any solver can be used for the framework discussed in the paper, this project has chosen a commercial solver, Gurobi. Therefore, a valid license will be required to execute most of the code. However, Gurobi does offer free academic licenses for acedemic personnels. More details regarding Gurobi license can be found [here](https://www.gurobi.com/features/academic-named-user-license/).

# Data Generation
This project involves training a CNN using supervised learning to predict the optimal binary solutions of the EVJRS problem. These prediction are fixed to reduce the solution search space, making it easier to solve by Gurobi. To generate the labelled training dataset, first generate sets of solar generation scenarios, load demand and EV job schedules using 'generate_solar_scenarios.py', 'generateLoad.py' and 'generate_schedule.py'. The generated inputs can then be used to generate different problem instnaces using 'EVScheduling_DistFlow_Optimised_train.py' and 'EVScheduling_DistFlow_Optimised_test.py'.

# Pre-processing
Run 'preprocess_train_padding.ipynb' and 'preprocess_test_padding.ipynb' to preprocess the generated problem instances for training and testing. The main difference between training and testing is that training uses deterministic version of EVJRS problem while testing uses stochastic verison of it (scenario-based_.

# Training
After pre-processing, run the script 'CNN_1D_training.ipynb' to train the CNN.

# Testing
After training, run th escript 'CNN_1D_testing.ipynb' to test the trained model.
