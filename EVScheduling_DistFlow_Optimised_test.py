import os
import sys
import math
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import gurobipy as gb
from gurobipy import GRB

# -----------------------------------------------------------------------------------------------------------------------------------------
# ========================================================== initial setup =========================================================
# -----------------------------------------------------------------------------------------------------------------------------------------

gurobi_env = gb.Env()
gurobi_env.setParam("OutputFlag", 0)

# read data from csv
data_dir = os.path.join(os.getcwd(), 'systemData')

# params of fixed gen, line and bus params
main_gen_params = pd.read_csv(os.path.join(data_dir, 'gen_params.csv')).to_numpy()
line_params = pd.read_csv(os.path.join(data_dir, 'line_params.csv')).to_numpy()[0:32]
bus_params = pd.read_csv(os.path.join(data_dir, 'bus_params.csv')).to_numpy()

# read in params decided by PSOGA
var_gen_params = pd.read_csv(os.path.join(data_dir, 'gen_params_variable.csv')).to_numpy()  # read in the bus location and DG size 
var_pv_params = pd.read_csv(os.path.join(data_dir, 'pv_params_variable.csv')).to_numpy()  # read in the bus location and PV size 
cs_param = pd.read_csv(os.path.join(data_dir, 'cs_params_variable.csv')).to_numpy()  # read in the charging station bus location 

# read in other EV params
EV_routes = pd.read_csv(os.path.join(data_dir, 'EV_routes.csv')).to_numpy()
EV_schedules_dir = os.listdir(os.path.join(os.getcwd(), "dataGeneration\EV_Schedule"))

# read in load and solar profile coefficient to build solar and load at each bus
load_coefficient_samples = pd.read_csv(os.path.join(data_dir, 'generatedLoad.csv'), index_col=[0]).to_numpy()

with open(os.path.join(data_dir, f"{200}_samples_{5}_scenario.pkl"), 'rb') as f:
    loaded_dict = pickle.load(f)
    pv_coefficient_sample = loaded_dict["scenarios"]
    prob_sample = loaded_dict["probabilities"]
    f.close()

pv_coefficient_sample = np.array(pv_coefficient_sample)
prob_sample = np.array(prob_sample)

# -----------------------------------------------------------------------------------------------------------------------------------------
# ========================================================== parameter setup =========================================================
# -----------------------------------------------------------------------------------------------------------------------------------------

# combine main and variable params (combine all element at same bus and sort them)
gen_params = (pd.DataFrame(var_gen_params)).groupby([0], sort=True).sum().reset_index().to_numpy()
pv_params = (pd.DataFrame(var_pv_params)).groupby([0], sort=True).sum().reset_index().to_numpy()

nbScen, nbTime, nbGen, nbPV, nbLine, nbBus, nbRoute = pv_coefficient_sample.shape[1], 48, (main_gen_params.shape[0]+np.array(var_gen_params).shape[0]) , pv_params.shape[0], line_params.shape[0], bus_params.shape[0], EV_routes.shape[0]

# traffic parameters
traffic = np.zeros(nbTime-1)
traffic[14:20] = 1      # from 7-10am 
traffic[32:40] = 1      # from 4-8pm

# set charging and non-charging stations (buses and transportation nodes)
charging_station = np.squeeze(cs_param)
non_charging_station = np.array([i for i in range(nbBus) if i not in charging_station])
nbCS = len(charging_station)

normal_nodes =  list(charging_station) + list(range(101,108))
virtual_nodes = list(range(201,205))
congest_nodes = list(range(301,324))

destination_nodes = normal_nodes + virtual_nodes + congest_nodes
nbDestination = len(destination_nodes)

# grid params
v_0 = 12.66  # slack bus base voltage (in kV)
loss = 0.1
line_limit_max = line_params[:,2]*1000
line_limit_min = line_limit_max*-1

# Generator params and combine params of DGs at each bus 
fuel_cost = 15   # per kW
gen_params = np.concatenate((main_gen_params, gen_params), axis=0)
gen_params[:,0] = np.round(gen_params[:,0])
gen_params[:,1] = gen_params[:,1] * 1000
gen_max = gen_params[:,1]

# get pv params
pv_params[:,0] = np.round(pv_params[:,0])
pv_params[:,1] = pv_params[:,1] * 1000

#EV params 
EV_Capacity, EV_MaxChargingRate, EV_MinChargingRate = 64, 19, -19     # (kWh, kW, kW)
EV_MinCapacity, EV_MaxCapacity = EV_Capacity*0.05, EV_Capacity*0.95     # 5%, 95% SOC
Initial_charge = EV_Capacity*0.05
Terminal_Charge = EV_Capacity*0.8    # 5%, 5% SOC

# Charging/Discharging Price
off_peak, mid_peak, peak = 0.281, 0.357, 0.584
EV_ChargeCost, EV_DischargeEarn = np.zeros(nbTime), np.zeros(nbTime)

EV_ChargeCost[0:16], EV_ChargeCost[40:48] = off_peak, off_peak
EV_ChargeCost[16:22], EV_ChargeCost[24:28], EV_ChargeCost[34:40] = mid_peak, mid_peak, mid_peak
EV_ChargeCost[22:24], EV_ChargeCost[28:34] = peak, peak

EV_DischargeEarn = EV_ChargeCost*0.9
EVTravelCost = 0 # set to 0 now, not sure if it was on purpose
EVTripConsumption = 9 # kW

# -----------------------------------------------------------------------------------------------------------------------------------------
# ========================================= setting up how the samples should be sampled =================================================
# -----------------------------------------------------------------------------------------------------------------------------------------
model_num = 63

n = 100
min_EV = 101
max_EV = 120
interval = 5
EV_num = list(range(min_EV, max_EV+1))
# EV_num = list(range(min_EV, max_EV+1, interval))

# if max_EV not in EV_num:
#     EV_num.append(max_EV)

# EV_num = list(set(range(min_EV, max_EV+1)) - set(EV_num)) # testing done for the rest of EV number that is not in train set
EV_num.sort()

EV_gap = len(EV_num)
print(EV_num, EV_gap)
# sys.exit()
# -----------------------------------------------------------------------------------------------------------------------------------------
# ========================================= use to generate nbEV x load_id x pv_id samples =================================================
# -----------------------------------------------------------------------------------------------------------------------------------------

while (model_num < n):
    model: gb.Model = gb.Model("EV_Routing", env=gurobi_env)
    model.setParam("TimeLimit", 120*60)

    # select which load or solar to use
    load_id = np.random.randint(0, load_coefficient_samples.shape[0])
    pv_id = np.random.randint(0, pv_coefficient_sample.shape[0])

    # load_id = 0
    # pv_id = 0

    # select the number of EVs
    option = int((np.floor(model_num / (n/EV_gap))))
    nbEV = EV_num[option]

    # select the EV schedule to use
    schedule_id = np.random.randint(0, len(EV_schedules_dir))
    # schedule_id = 0
    EV_schedules = pd.read_csv(os.path.join(os.getcwd(), f"dataGeneration\EV_Schedule\EV_schedules{schedule_id}.csv")).to_numpy()
    EV_schedules = EV_schedules[0:nbEV*4,:]

    # build the load and solar profile using the randomly generated coefficients
    load_coefficient = load_coefficient_samples[load_id,:]
    pv_coefficient = pv_coefficient_sample[pv_id,:,:]
    pv_coefficient = np.repeat(pv_coefficient[:, np.newaxis, :], nbPV, axis=1) # use to expand dimension for each panel (only works if all panel same size)

    prob = prob_sample[pv_id,:]

    bus_active_load = (bus_params[:,2].reshape(-1,1) * load_coefficient.reshape(1,-1)) * 1000
    bus_reactive_load = (bus_params[:,3].reshape(-1,1) * load_coefficient.reshape(1,-1)) *1000
    pv_power_max = (pv_params[:,1].reshape(1,-1,1) * pv_coefficient).squeeze()

    print("=============================================================================================================================================================================================")
    print(f"Building Model Number {model_num}")
    print(f"Load Sample: {load_id}, Solar Sample: {pv_id}, EV Schedule: {schedule_id}")
    print(f"Simulation for (Scenario number {nbScen}, EV number: {nbEV}), (Solar Samples: {pv_coefficient_sample.shape[0]}), (Load Samples: {load_coefficient_samples.shape[0]}), (Schedule Number: {len(EV_schedules_dir)})")
    print(f"Important Parameters (Scenario: {nbScen}), (Time Length: {nbTime}), (Generator Number: {nbGen}), (PV Number: {nbPV}), (Line Number: {nbLine}), (Bus Number: {nbBus}), (Route Number: {nbRoute}), (Charging Station Number: {nbCS})")
    print(f"Load Data Shape: {bus_active_load.shape}, Solar Data Shape: {pv_power_max.shape}")
    print(f"Schedule Size Shape: {EV_schedules.shape}")

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== variable setup =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    # EV Variables
    EVArcStatus = model.addVars(
        nbScen,
        nbEV,
        nbRoute,
        nbTime-1,
        vtype=gb.GRB.BINARY,
        name="EVArcStatus",
    )

    EVChargeStatus = model.addVars(
        nbScen,
        nbEV,
        nbCS,
        nbTime,
        vtype=gb.GRB.BINARY,
        name="EVChargeStatus",
    )

    EVDischargeStatus = model.addVars(
        nbScen,
        nbEV,
        nbCS,
        nbTime,
        vtype=gb.GRB.BINARY,
        name="EVDischargeStatus",
    )

    EVChargePower = model.addVars(
        nbScen,
        nbEV,
        nbCS,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="EVChargePower",
    )

    EVDischargePower = model.addVars(
        nbScen,
        nbEV,
        nbCS,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="EVDischargePower",
    )

    EVMovePower = model.addVars(
        nbScen,
        nbEV,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="EVMovePower",
    )

    EVEnergy = model.addVars(
        nbScen,
        nbEV,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="EVEnergy",
    )

    # Grid variables
    BusVolt = model.addVars(
        nbScen,
        nbBus,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="BusVoltage",
    )

    ActiveLineFlow = model.addVars(
        nbScen,
        nbLine,
        nbTime,
        lb = -float("inf"),
        ub = float("inf"),
        vtype=gb.GRB.CONTINUOUS,
        name="ActiveLineFlow",
    )

    ReactiveLineFlow = model.addVars(
        nbScen,
        nbLine,
        nbTime,
        lb = -float("inf"),
        ub = float("inf"),
        vtype=gb.GRB.CONTINUOUS,
        name="ReactiveLineFlow",
    )

    # Generator variables
    GenActivePower = model.addVars(
        nbScen,
        nbGen,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="GenActivePower",
    )

    GenReactivePower = model.addVars(
        nbScen,
        nbGen,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="GenReactivePower",
    )

    PVPower = model.addVars(
        nbScen,
        nbPV,
        nbTime,
        vtype=gb.GRB.CONTINUOUS,
        name="PVPower",
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== constraint setup =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    # ========================================================== Routing Constraints ==========================================================

    # Ensures EVs can only choose 1 arc at any time
    model.addConstrs(
        (
            EVArcStatus.sum(sc, k, "*", s) == 1
            for sc in range(nbScen)
            for k in range(nbEV)
            for s in range(nbTime-1)
        ),
        name=f"EVArcSelection0",
    )

    # traffic condition, decides which arc EV can choose depending on traffic condition
    always_arc = []
    normal_arc = []
    congest_arc = []

    for r in range(nbRoute):
        if (EV_routes[r,1] in (normal_nodes+virtual_nodes)) and (EV_routes[r,2] in (normal_nodes+virtual_nodes)) and (EV_routes[r,1] != EV_routes[r,2]):
            normal_arc.append(r)
        elif (EV_routes[r,1] in (normal_nodes+virtual_nodes)) and (EV_routes[r,2] in congest_nodes):
            congest_arc.append(r)
        else:
            always_arc.append(r)

    for s in range(nbTime-1):
        if traffic[s]:
            # with jam
            activate_arc = always_arc + congest_arc
            disable_arc = normal_arc

        else:
            # no jam
            activate_arc = always_arc + normal_arc
            disable_arc = congest_arc

        model.addConstrs(
            (
                EVArcStatus.sum(sc, k, activate_arc, s) == 1
                for sc in range(nbScen)
                for k in range(nbEV)
            ),
            name=f"EVArcTrafficEnable_{s}",
        )
        model.addConstrs(
            (
                EVArcStatus.sum(sc, k, disable_arc, s) == 0
                for sc in range(nbScen)
                for k in range(nbEV)
            ),
            name=f"EVArcTrafficDisable_{s}",
        )

    # Ensures EVs will depart from the last arrived station
    for sc in range(nbScen):
        for k in range(nbEV):
            for s in range(nbTime-2):
                for i in range(nbDestination):
                    from_index = (EV_routes[:,1]==destination_nodes[i]).nonzero()[0]
                    to_index = (EV_routes[:,2]==destination_nodes[i]).nonzero()[0]

                    model.addConstr(
                        (
                            EVArcStatus.sum(sc, k, from_index, s+1) == EVArcStatus.sum(sc, k, to_index, s)
                        ),
                        name=f"EVArcSelection1_{sc}_{k}_{i}_{s}",
                    )

    # Ensure EV reaches the scheduled destination at the right time (starting position and ending positions)
    nbSchedule = EV_schedules.shape[0]
    for sc in range(nbScen):
        for i in range(nbSchedule):
            schedule = EV_schedules[i]  # [EV, destination, time]

            if schedule[2] == 0:
                route_index = (EV_routes[:,1]==schedule[1]).nonzero()[0] 
                Time = 0
            else:
                route_index = (EV_routes[:,2]==schedule[1]).nonzero()[0]
                Time = schedule[2]-1

            model.addConstr(
                (
                    EVArcStatus.sum(sc, schedule[0], route_index, Time) == 1
                ),
                name=f"EVSchedule_{sc}_{i}",
            )
        
    # ========================================================== EV Power Constraints ==========================================================

    # Ensure EV can only connect to grid if it doesnt move during the time span and set the max and min charging/discharging rate
    stationary_index = np.array((EV_routes[:,1] == EV_routes[:,2]).nonzero()[0])
    charging_index = np.array([i for i, e in enumerate(EV_routes[:,1]) if e in charging_station])
    charge_index = [i for i, e in enumerate(stationary_index) if e in charging_index]

    cs_map = [] # maps the corresponding charging station to the variable nbCS
    for i, ii in enumerate(stationary_index[charge_index]):
        cs_map.append(EV_routes[ii][2])

    model.addConstrs(
        (
            EVChargeStatus[sc,k,i,s+1] + EVDischargeStatus[sc,k,i,s+1] <= EVArcStatus[sc,k,ii,s]
            for sc in range(nbScen)
            for k in range(nbEV)
            for i, ii in enumerate(stationary_index[charge_index])
            for s in range(nbTime-1)
        ),
        name=f"EVCharge/DischargeStatus",
    )

    # EV consumption during movement (fixed consumption)
    non_stationary_index = (EV_routes[:,1] != EV_routes[:,2]).nonzero()[0]
    model.addConstrs(
        (
            EVMovePower[sc,k,s+1] == EVArcStatus.sum(sc,k,non_stationary_index,s) * EVTripConsumption
            for sc in range(nbScen)
            for s in range(nbTime-1)
            for k in range(nbEV)
        ),
        name=f"EVMoveConsumption",
    )

    model.addConstrs(
        (
            EVMovePower[sc,k,0] == 0
            for sc in range(nbScen)
            for k in range(nbEV)
        ),
        name=f"EVMoveConsumptionStart",
    )

    # max/min charging rate
    model.addConstrs(
        (
            EVChargePower[sc,k,i,t] <= EV_MaxChargingRate * EVChargeStatus[sc,k,i,t]
            for sc in range(nbScen)
            for t in range(nbTime)
            for k in range(nbEV)
            for i in range(nbCS)
        ),
        name=f"EVChargeMax",
    )

    model.addConstrs(
        (
            0 <= EVChargePower[sc,k,i,t]
            for sc in range(nbScen)
            for t in range(nbTime)
            for k in range(nbEV)
            for i in range(nbCS)
        ),
        name=f"EVChargeMin",
    )

    # max/min discharge rate
    model.addConstrs(
        (
            EVDischargePower[sc,k,i,t] <= EV_MaxChargingRate * EVDischargeStatus[sc,k,i,t]
            for sc in range(nbScen)
            for t in range(nbTime)
            for k in range(nbEV)
            for i in range(nbCS)
        ),
        name=f"EVDischargeMax",
    )

    model.addConstrs(
        (
            0 <= EVDischargePower[sc,k,i,t]
            for sc in range(nbScen)
            for t in range(nbTime)
            for k in range(nbEV)
            for i in range(nbCS)
        ),
        name=f"EVDischargeMin",
    )

    # stop any charging or discharging at start
    for sc in range(nbScen):
        for k in range(nbEV):
            for i in range(nbCS):

                model.addConstr(
                    (
                        EVChargePower[sc,k,i,0] == 0
                    ),
                    name=f"StartTimeEVChargePower_{sc}_{k}_{i}",
                )

                model.addConstr(
                    (
                        EVDischargePower[sc,k,i,0] == 0  
                    ),
                    name=f"StartTimeEVDischargePower_{sc}_{k}_{i}",
                )

    # Set EV to not charge or discharge past certain SOC
    model.addConstrs(
        (
            EVEnergy[sc,k,t] <= EV_MaxCapacity
            for sc in range(nbScen)
            for t in range(nbTime)
            for k in range(nbEV)
        ),
        name=f"EVEnergyMax",
    )

    model.addConstrs(
        (
            EVEnergy[sc,k,t] >= EV_MinCapacity 
            for sc in range(nbScen)
            for t in range(nbTime)
            for k in range(nbEV)
        ),
        name=f"EVEnergyMin",
    )

    model.addConstrs(
        (
            EVEnergy[sc,k,0] == Initial_charge
            for sc in range(nbScen)
            for k in range(nbEV)
        ),
        name=f"EVEnergyStart",
    )

    model.addConstrs(
        (
            EVEnergy[sc,k,nbTime-1] >= Terminal_Charge # -1 because index starts from 0
            for sc in range(nbScen)
            for k in range(nbEV)
        ),
        name=f"EVEnergyEnd",
    )
    # Energy Balance (*1 because time span is 1hr)
    model.addConstrs(
        (
            EVEnergy[sc,k,t] == EVEnergy[sc,k,t-1] + ((1-loss)*EVChargePower.sum(sc,k,"*",t) - (1+loss)*(EVDischargePower.sum(sc,k,"*",t) + EVMovePower[sc,k,t])) * (24/nbTime)
            for sc in range(nbScen)
            for t in range(1,nbTime)
            for k in range(nbEV)
        ),
        name=f"EVEnergyBalance",
    )

    # ========================================================== Generator Constraints ==========================================================
    # add generation limits
    model.addConstrs(
        (
            GenActivePower[sc,u,t] <= gen_max[u]
            for sc in range(nbScen)
            for u in range(nbGen)
            for t in range(nbTime)
        ),
        name=f"GeneratorActivePowerMax",
    )

    model.addConstrs(
        (
            GenActivePower[sc,u,t] >= 0 # gen_min[u] 
            for sc in range(nbScen)
            for u in range(nbGen)
            for t in range(nbTime)
        ),
        name=f"GeneratorActivePowerMin",
    )

    model.addConstrs(
        (
            GenReactivePower[sc,u,t] <= gen_max[u]
            for sc in range(nbScen)
            for u in range(nbGen)
            for t in range(nbTime)
        ),
        name=f"GeneratorReactivePowerMax",
    )

    model.addConstrs(
        (
            GenReactivePower[sc,u,t] >= 0 # gen_min[u] 
            for sc in range(nbScen)
            for u in range(nbGen)
            for t in range(nbTime)
        ),
        name=f"GeneratorReactivePowerMin",
    )

    model.addConstrs(
        (
            PVPower[sc,p,t] <= pv_power_max[sc,p,t]
            for sc in range(nbScen)
            for p in range(nbPV)
            for t in range(nbTime)
        ),
        name=f"PVPowerMax",
    )

    model.addConstrs(
        (
            PVPower[sc,p,t] >= 0
            for sc in range(nbScen)
            for p in range(nbPV)
            for t in range(nbTime)
        ),
        name=f"PVPowerMin",
    )

    # ========================================================== Line Constraints ==========================================================

    # add line constraints
    # add active line flow limits
    model.addConstrs(
        (
            ActiveLineFlow[sc,l,t] <= line_limit_max[l]
            for sc in range(nbScen)
            for l in range(nbLine)
            for t in range(nbTime)
        ),
        name=f"ActiveLinePowerMax",
    )

    model.addConstrs(
        (
            ActiveLineFlow[sc,l,t] >= line_limit_min[l]
            for sc in range(nbScen)
            for l in range(nbLine)
            for t in range(nbTime)
        ),
        name=f"ActiveLinePowerMin",
    )

    # add reactive line flow limits
    model.addConstrs(
        (
            ReactiveLineFlow[sc,l,t] <= line_limit_max[l]
            for sc in range(nbScen)
            for l in range(nbLine)
            for t in range(nbTime)
        ),
        name=f"ReactiveLinePowerMax",
    )

    model.addConstrs(
        (
            ReactiveLineFlow[sc,l,t] >= line_limit_min[l]
            for sc in range(nbScen)
            for l in range(nbLine)
            for t in range(nbTime)
        ),
        name=f"ReactiveLinePowerMin",
    )  

    # add voltage limits
    model.addConstrs(
        (
            BusVolt[sc,l,t] <= v_0*1.05
            for sc in range(nbScen)
            for l in range(nbLine)
            for t in range(nbTime)
        ),
        name=f"BusVoltMax",
    )

    model.addConstrs(
        (
            BusVolt[sc,l,t] >= v_0*0.95
            for sc in range(nbScen)
            for l in range(nbLine)
            for t in range(nbTime)
        ),
        name=f"BusVoltMin",
    )

# ============================================== Line Power and Voltage Calculation ==================================================

    # Calculation of line power by first determining GenPower
    # intialise a counter for each generator
    for sc in range(nbScen):
        c_dg = 0
        c_pv = 0
        c_batt = 0

        for i in range(nbBus):

            downbus = []
            upbus = []
            # determine up and downstream line idx
            for l in range(nbLine):
                if line_params[l,1] == i:
                    upbus.append(l)
                elif line_params[l,0] == i:
                    downbus.append(l)

            downbus = np.array(downbus)
            upbus = np.array(upbus)

            for t in range(nbTime):
                # calculate GenPower
                # if the bus has both dg and pv
                if (i in gen_params[:,0]) and (i in pv_params[:,0]):
                    gap = GenActivePower[sc,c_dg,t] + PVPower[sc,c_pv,t]
                    grp = GenReactivePower[sc,c_dg,t]
                # if the bus only has dg
                elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
                    gap = GenActivePower[sc,c_dg,t]
                    grp = GenReactivePower[sc,c_dg,t]
                # if the bus only has pv
                elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
                    gap = PVPower[sc,c_pv,t]
                    grp = 0
                else:
                    gap = 0
                    grp = 0

                # calculate BusPower
                if (i in cs_map):
                    bap = bus_active_load[i,t] + EVChargePower.sum(sc,"*",cs_map.index(i),t) - EVDischargePower.sum(sc,"*",cs_map.index(i),t)
                    brp = bus_reactive_load[i,t]
                else:
                    bap = bus_active_load[i,t]
                    brp = bus_reactive_load[i,t]

                # set constraint after determining GenPower and BusPower
                # calculate Line power and system power balance here
                if not len(upbus):
                    # no more up stream bus
                    lhs_0 = gap
                    rhs_0 = ActiveLineFlow.sum(sc,downbus,t) + bap
                    lhs_1 = grp
                    rhs_1 = ReactiveLineFlow.sum(sc,downbus,t) + brp

                elif not len(downbus):
                    # no more down stream bus
                    lhs_0 = gap + ActiveLineFlow[sc,upbus[0],t]
                    rhs_0 = bap
                    lhs_1 = grp + ReactiveLineFlow[sc,upbus[0],t]
                    rhs_1 = brp

                else:
                    # for rest of the cases
                    lhs_0 = gap + ActiveLineFlow[sc,upbus[0],t]
                    rhs_0 = ActiveLineFlow.sum(sc,downbus,t) + bap
                    lhs_1 = grp + ReactiveLineFlow[sc,upbus[0],t]
                    rhs_1 = ReactiveLineFlow.sum(sc,downbus,t) + brp

                # add constraint
                model.addConstr(
                    (
                        lhs_0 == rhs_0
                    ),
                    name=f"ActiveBusLinePower_{sc}_{i}_{t}",
                )
                model.addConstr(
                    (
                        lhs_1 == rhs_1
                    ),
                    name=f"ReactiveBusLinePower_{sc}_{i}_{t}",
                )

            # increase generator counters
            if (i in gen_params[:,0]) and (i in pv_params[:,0]):
                c_dg = c_dg + 1
                c_pv = c_pv + 1
            # if the bus only has dg
            elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
                c_dg = c_dg + 1
            # if the bus only has pv
            elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
                c_pv = c_pv + 1


    # calculate voltage
    # add slack bus voltage
    model.addConstrs(
        (
            BusVolt[sc,0,t] == v_0
            for sc in range(nbScen)
            for t in range(nbTime)
        ),
        name=f"SlackVoltage_{t}",
    )

    
    model.addConstrs(
        (
            BusVolt[sc,line_params[l,0],t] - ((ActiveLineFlow[sc,l,t]*line_params[l,3] + ReactiveLineFlow[sc,l,t]*line_params[l,4])/v_0) - BusVolt[sc,line_params[l,1],t] == 0
            for sc in range(nbScen)
            for l in range(nbLine)
            for t in range(nbTime)
        ),
        name=f"VoltageBalance",
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== Objective function setup =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    expr_GenerationCost = gb.LinExpr()
    expr_TravelCost = gb.LinExpr()
    expr_ChargingCost = gb.LinExpr()
    expr_DischargingEarn = gb.LinExpr()
    
    for sc in range(nbScen):
        for u in range(nbGen):
            for t in range(nbTime):
                expr_GenerationCost.add(GenActivePower[sc,u,t] * fuel_cost * prob[sc]) 

    non_stationary_index = (EV_routes[:,1] != EV_routes[:,2]).nonzero()[0]
    for sc in range(nbScen):
        for k in range(nbEV):
            for s in range(nbTime-1):
                for ij in non_stationary_index:
                    expr_TravelCost.add(EVArcStatus[sc,k,ij,s] * EVTravelCost * prob[sc])

    for sc in range(nbScen):
        for k in range(nbEV):
            for i in range(nbCS):
                for t in range(nbTime):
                    expr_ChargingCost.add(EVChargePower[sc,k,i,t] * EV_ChargeCost[t] * prob[sc])

    for sc in range(nbScen):
        for k in range(nbEV):
            for i in range(nbCS):
                for t in range(nbTime):
                    expr_DischargingEarn.add(EVDischargePower[sc,k,i,t] * EV_DischargeEarn[t] * prob[sc])

    model.setObjective(expr_GenerationCost + expr_TravelCost + expr_ChargingCost - expr_DischargingEarn, GRB.MINIMIZE)
    # model.setObjective(expr_TravelCost + expr_ChargingCost - expr_DischargingEarn, GRB.MINIMIZE)

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== solving model =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    # solving model
    model.update()
    model.Params.Threads = 1
    model.Params.PoolSearchMode = 0
    model.Params.DualReductions = 0
    model.reset(0)
    print('optimizing ...')
    model.optimize()

    # if model infeasible print out violated constraints and variables, else print out results for analysing purposes
    if model.Status == GRB.INFEASIBLE:
        print('infeasible')

    else:
        try:
            print("Objective value: ", model.ObjVal)
            print("Model runtime: ", model.Runtime)   
            print("MIP Gap:", model.MIPGap)

            # sys.exit()

            variables = model.getVars()

            # make sure solution are order based on EV number
            # training_binary_test = []
            # training_idx_test = []

            training_binary = []
            training_idx = []

            for sc in range(nbScen):
                for k in range(nbEV):
                    # for each EV get each attribute and order it based on EV number
                    for r in range(nbRoute):
                        for t in range(nbTime-1):
                            var = model.getVarByName(f"EVArcStatus[{sc},{k},{r},{t}]")

                            training_binary.append(var.X)
                            training_idx.append(var.index)

                    for c in range(nbCS):
                        for t in range(nbTime):
                            var = model.getVarByName(f"EVChargeStatus[{sc},{k},{c},{t}]")

                            training_binary.append(var.X)
                            training_idx.append(var.index)
                            
                    for c in range(nbCS):
                        for t in range(nbTime):
                            var = model.getVarByName(f"EVDischargeStatus[{sc},{k},{c},{t}]")

                            training_binary.append(var.X)
                            training_idx.append(var.index)

            print("done")

            dataPoint = {
                "Schedule": EV_schedules,
                "Load": np.concatenate((bus_active_load, bus_reactive_load), axis=0),
                "Solar": pv_power_max,
                "Binary": np.array(training_binary),
                "Indices": np.array(training_idx).astype("int64"),
                "solve_time": model.Runtime,
                "Obj_val": model.ObjVal,
                "model": model_num,
                "nbEV": nbEV,
                "schedule_id": schedule_id
            }

            save_path = f"dataGeneration/feature_target_test_out/dataPoint_{model_num}.pkl"
            data_path = os.path.join(os.getcwd(), save_path)
            with open(data_path, 'wb') as f:
                pickle.dump(dataPoint, f)

            model.reset(0)
            save_path = f"dataGeneration/model_test_out/coordination_{model_num}.mps"

            model.write(os.path.join(os.getcwd(), save_path))

            model_num = model_num + 1

        except:
            continue

        # -----------------------------------------------------------------------------------------------------------------------------------------
        # ========================================================== Perform Testing From Here =========================================================
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # comment out below for verifying line power with ptdf
        # create placeholder for total genpower and total buspower (we only check one timestamp)
        # timestamp = 35
        # ScenNum = 3

        # TotalActiveGenPower = np.zeros((nbBus, 1))
        # TotalActiveBusPower = np.zeros((nbBus, 1))
        # TotalReactiveGenPower = np.zeros((nbBus, 1))
        # TotalReactiveBusPower = np.zeros((nbBus, 1))

        # c_dg = 0
        # c_pv = 0

        # for i in range(nbBus):   
        #     # determine required variable first
        #     var0 = model.getVarByName(f"GenActivePower[{ScenNum},{c_dg},{timestamp}]")
        #     var1 = model.getVarByName(f"PVPower[{ScenNum},{c_pv},{timestamp}]")

        #     charge_sum = []
        #     discharge_sum = [] 
        #     for k in range(nbEV):
        #         if i in cs_map:
        #             var2 = model.getVarByName(f"EVChargePower[{ScenNum},{k},{cs_map.index(i)},{timestamp}]")
        #             var3 = model.getVarByName(f"EVDischargePower[{ScenNum},{k},{cs_map.index(i)},{timestamp}]")

        #             charge_sum.append(var2.X)
        #             discharge_sum.append(var3.X)

        #     charge_sum = np.sum(charge_sum)
        #     discharge_sum = np.sum(discharge_sum)

        #     var5 = model.getVarByName(f"GenReactivePower[{ScenNum},{c_dg},{timestamp}]")
            

        #     # if the bus has both dg and pv
        #     if (i in gen_params[:,0]) and (i in pv_params[:,0]):
        #         gap = var0.X + var1.X
        #         grp = var5.X
        #     # if the bus only has dg
        #     elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
        #         gap = var0.X 
        #         grp = var5.X
        #     # if the bus only has pv
        #     elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
        #         gap = var1.X 
        #         grp = 0
        #     else:
        #         gap = 0
        #         grp = 0

        #     if (i in cs_map):
        #         bap = bus_active_load[i,timestamp] + charge_sum - discharge_sum
        #         brp = bus_reactive_load[i,timestamp]
        #     else:
        #         bap = bus_active_load[i,timestamp]
        #         brp = bus_reactive_load[i,timestamp]

        #     TotalActiveGenPower[i,0] = gap
        #     TotalActiveBusPower[i,0] = bap
        #     TotalReactiveGenPower[i,0] = grp
        #     TotalReactiveBusPower[i,0] = brp

        #     # increase generator counters
        #     if (i in gen_params[:,0]) and (i in pv_params[:,0]):
        #         c_dg = c_dg + 1
        #         c_pv = c_pv + 1
        #     # if the bus only has dg
        #     elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
        #         c_dg = c_dg + 1
        #     # if the bus only has pv
        #     elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
        #         c_pv = c_pv + 1


        # # comment out below to perform other testing 
        # LinePower = np.zeros((nbLine,2))
        # for m in range(nbLine):
        #     var_active = model.getVarByName(f"ActiveLineFlow[{ScenNum},{m},{timestamp}]")
        #     var_reactive = model.getVarByName(f"ReactiveLineFlow[{ScenNum},{m},{timestamp}]")

        #     LinePower[m,0] = var_active.X
        #     LinePower[m,1] = var_reactive.X
            
        # ActiveGenLoad = TotalActiveGenPower - TotalActiveBusPower
        # ReactiveGenLoad = TotalReactiveGenPower - TotalReactiveBusPower
        # linepower = LinePower

        # print(ActiveGenLoad.shape)
        # print(ReactiveGenLoad.shape)
        # print(linepower.shape)

        # gl_df = pd.DataFrame(TotalActiveGenPower)
        # gl_df.to_csv(os.path.join(data_dir, 'ActiveGen.csv'))

        # gl_df = pd.DataFrame(TotalActiveBusPower)
        # gl_df.to_csv(os.path.join(data_dir, 'ActiveLoad.csv'))

        # gl_df = pd.DataFrame(TotalReactiveGenPower)
        # gl_df.to_csv(os.path.join(data_dir, 'ReactiveGen.csv'))

        # gl_df = pd.DataFrame(TotalReactiveBusPower)
        # gl_df.to_csv(os.path.join(data_dir, 'ReactiveLoad.csv'))

        # lp_df = pd.DataFrame(linepower)
        # lp_df.to_csv(os.path.join(data_dir, 'line_power.csv'))


    model.dispose()

    # sys.exit()


        