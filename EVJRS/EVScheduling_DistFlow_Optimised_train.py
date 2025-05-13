import os
import sys
import math
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from utility_serial import *
import time

import gurobipy as gb
from gurobipy import GRB

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== initial setup =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

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
        f.close()
    pv_coefficient_sample = np.array(pv_coefficient_sample).reshape(-1,48)

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================== parameter setup =========================================================
    # -----------------------------------------------------------------------------------------------------------------------------------------

    # combine main and variable params (combine all element at same bus and sort them)
    gen_params = (pd.DataFrame(var_gen_params)).groupby([0], sort=True).sum().reset_index().to_numpy()
    pv_params = (pd.DataFrame(var_pv_params)).groupby([0], sort=True).sum().reset_index().to_numpy()

    nbTime, nbGen, nbPV, nbLine, nbBus, nbRoute = 48, (main_gen_params.shape[0]+np.array(var_gen_params).shape[0]) , pv_params.shape[0], line_params.shape[0], bus_params.shape[0], EV_routes.shape[0]

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
    print(congest_nodes)
    sys.exit()

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
    # provide parameters here
    interval_array = [5,10,15,20]
    min_EV = 20
    max_EV = 100
    total_sample = 800 # total samples required for each interval 

    interval_dict = []

    # build the dict for each interval to determine number of datapoints needed for each interval
    for x in interval_array:

        # create the array of EVs for each interval 
        EV_num = list(range(min_EV, max_EV+1, x))
        if max_EV not in EV_num:
            EV_num.append(max_EV)

        temp_dict = {}
        for e in EV_num:
            if e == EV_num[-1]:
                temp_dict[e] = total_sample - int(total_sample/len(EV_num)) * (len(EV_num) - 1)
            else:
                temp_dict[e] = int(total_sample/len(EV_num))
                
        interval_dict.append(temp_dict)

    # after building the dict for each interval, aggregate them to get the required samples for each EV number
    sample_required_dict = {}
    for x in interval_dict[0]: # get the EV with smallest interval

        sample_required = []
        for i in interval_dict:
            try:
                sample_required.append(i[x])
            except:
                continue

        sample_required_dict[x] = np.max(sample_required)

    # create the array for easy data generation
    EV_num_to_gen = np.array([])
    for id, x in enumerate(sample_required_dict.keys()):
        EV_num_to_gen = np.concatenate((EV_num_to_gen, np.ones((sample_required_dict[x],))*x))

    EV_num_to_gen = EV_num_to_gen.astype("int32")

    # data generation starts here
    model_num = 1552

    n = 1578 # total samples to be generated for 4 different intervals
    print(n)

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
        nbEV = int(EV_num_to_gen[model_num])        

        # select the EV schedule to use
        schedule_id = np.random.randint(0, len(EV_schedules_dir))
        # schedule_id = 0
        EV_schedules = pd.read_csv(os.path.join(os.getcwd(), f"dataGeneration\EV_Schedule\EV_schedules{schedule_id}.csv")).to_numpy()
        EV_schedules = EV_schedules[0:nbEV*4,:]

        # build the load and solar profile using the randomly generated coefficients
        load_coefficient = load_coefficient_samples[load_id,:]
        pv_coefficient = np.transpose(pv_coefficient_sample[pv_id,:])

        bus_active_load = (bus_params[:,2].reshape(-1,1) * load_coefficient.reshape(1,-1)) * 1000
        bus_reactive_load = (bus_params[:,3].reshape(-1,1) * load_coefficient.reshape(1,-1)) *1000
        pv_power_max = (pv_params[:,1].reshape(1,-1,1) * pv_coefficient).squeeze()

        print("=============================================================================================================================================================================================")
        print(f"Building Model Number {model_num}")
        print(f"Load Sample: {load_id}, Solar Sample: {pv_id}, EV Schedule: {schedule_id}")
        print(f"Simulation for (EV number: {nbEV}), (Solar Samples: {pv_coefficient_sample.shape[0]}), (Load Samples: {load_coefficient_samples.shape[0]}), (Schedule Number: {len(EV_schedules_dir)})")
        print(f"Important Parameters (Time Length: {nbTime}), (Generator Number: {nbGen}), (PV Number: {nbPV}), (Line Number: {nbLine}), (Bus Number: {nbBus}), (Route Number: {nbRoute}), (Charging Station Number: {nbCS})")
        print(f"Load Data Shape: {bus_active_load.shape}, Solar Data Shape: {pv_power_max.shape}")

        # -----------------------------------------------------------------------------------------------------------------------------------------
        # ========================================================== variable setup =========================================================
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # EV Variables
        EVArcStatus = model.addVars(
            nbEV,
            nbRoute,
            nbTime-1,
            vtype=gb.GRB.BINARY,
            name="EVArcStatus",
        )

        EVChargeStatus = model.addVars(
            nbEV,
            nbCS,
            nbTime,
            vtype=gb.GRB.BINARY,
            name="EVChargeStatus",
        )

        EVDischargeStatus = model.addVars(
            nbEV,
            nbCS,
            nbTime,
            vtype=gb.GRB.BINARY,
            name="EVDischargeStatus",
        )

        EVChargePower = model.addVars(
            nbEV,
            nbCS,
            nbTime,
            vtype=gb.GRB.CONTINUOUS,
            name="EVChargePower",
        )

        EVDischargePower = model.addVars(
            nbEV,
            nbCS,
            nbTime,
            vtype=gb.GRB.CONTINUOUS,
            name="EVDischargePower",
        )

        EVMovePower = model.addVars(
            nbEV,
            nbTime,
            vtype=gb.GRB.CONTINUOUS,
            name="EVMovePower",
        )

        EVEnergy = model.addVars(
            nbEV,
            nbTime,
            vtype=gb.GRB.CONTINUOUS,
            name="EVEnergy",
        )

        # Grid variables
        BusVolt = model.addVars(
            nbBus,
            nbTime,
            vtype=gb.GRB.CONTINUOUS,
            name="BusVoltage",
        )

        ActiveLineFlow = model.addVars(
            nbLine,
            nbTime,
            lb = -float("inf"),
            ub = float("inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="ActiveLineFlow",
        )

        ReactiveLineFlow = model.addVars(
            nbLine,
            nbTime,
            lb = -float("inf"),
            ub = float("inf"),
            vtype=gb.GRB.CONTINUOUS,
            name="ReactiveLineFlow",
        )

        # Generator variables
        GenActivePower = model.addVars(
            nbGen,
            nbTime,
            vtype=gb.GRB.CONTINUOUS,
            name="GenActivePower",
        )

        GenReactivePower = model.addVars(
            nbGen,
            nbTime,
            vtype=gb.GRB.CONTINUOUS,
            name="GenReactivePower",
        )

        PVPower = model.addVars(
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
                EVArcStatus.sum(k, "*", s) == 1
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
                    EVArcStatus.sum(k, activate_arc, s) == 1
                    for k in range(nbEV)
                ),
                name=f"EVArcTrafficEnable_{s}",
            )
            model.addConstrs(
                (
                    EVArcStatus.sum(k, disable_arc, s) == 0
                    for k in range(nbEV)
                ),
                name=f"EVArcTrafficDisable_{s}",
            )

        # Ensures EVs will depart from the last arrived station
        for k in range(nbEV):
            for s in range(nbTime-2):
                for i in range(nbDestination):
                    from_index = (EV_routes[:,1]==destination_nodes[i]).nonzero()[0]
                    to_index = (EV_routes[:,2]==destination_nodes[i]).nonzero()[0]

                    model.addConstr(
                        (
                            EVArcStatus.sum(k, from_index, s+1) == EVArcStatus.sum(k, to_index, s)
                        ),
                        name=f"EVArcSelection1_{k}_{i}_{s}",
                    )

        # Ensure EV reaches the scheduled destination at the right time (starting position and ending positions)
        nbSchedule = EV_schedules.shape[0]
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
                    EVArcStatus.sum(schedule[0], route_index, Time) == 1
                ),
                name=f"EVSchedule_{i}",
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
                EVChargeStatus[k,i,s+1] + EVDischargeStatus[k,i,s+1] <= EVArcStatus[k,ii,s]
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
                EVMovePower[k,s+1] == EVArcStatus.sum(k,non_stationary_index,s) * EVTripConsumption
                for s in range(nbTime-1)
                for k in range(nbEV)
            ),
            name=f"EVMoveConsumption",
        )

        model.addConstrs(
            (
                EVMovePower[k,0] == 0
                for k in range(nbEV)
            ),
            name=f"EVMoveConsumptionStart",
        )

        # max/min charging rate
        model.addConstrs(
            (
                EVChargePower[k,i,t] <= EV_MaxChargingRate * EVChargeStatus[k,i,t]
                for t in range(nbTime)
                for k in range(nbEV)
                for i in range(nbCS)
            ),
            name=f"EVChargeMax",
        )

        model.addConstrs(
            (
                0 <= EVChargePower[k,i,t]
                for t in range(nbTime)
                for k in range(nbEV)
                for i in range(nbCS)
            ),
            name=f"EVChargeMin",
        )

        # max/min discharge rate
        model.addConstrs(
            (
                EVDischargePower[k,i,t] <= EV_MaxChargingRate * EVDischargeStatus[k,i,t]
                for t in range(nbTime)
                for k in range(nbEV)
                for i in range(nbCS)
            ),
            name=f"EVDischargeMax",
        )

        model.addConstrs(
            (
                0 <= EVDischargePower[k,i,t]
                for t in range(nbTime)
                for k in range(nbEV)
                for i in range(nbCS)
            ),
            name=f"EVDischargeMin",
        )

        # stop any charging or discharging at start
        for k in range(nbEV):
            for i in range(nbCS):

                model.addConstr(
                    (
                        EVChargePower[k,i,0] == 0
                    ),
                    name=f"StartTimeEVChargePower_{k}_{i}",
                )

                model.addConstr(
                    (
                        EVDischargePower[k,i,0] == 0  
                    ),
                    name=f"StartTimeEVDischargePower_{k}_{i}",
                )

        # Set EV to not charge or discharge past certain SOC
        model.addConstrs(
            (
                EVEnergy[k,t] <= EV_MaxCapacity
                for t in range(nbTime)
                for k in range(nbEV)
            ),
            name=f"EVEnergyMax",
        )

        model.addConstrs(
            (
                EVEnergy[k,t] >= EV_MinCapacity 
                for t in range(nbTime)
                for k in range(nbEV)
            ),
            name=f"EVEnergyMin",
        )

        model.addConstrs(
            (
                EVEnergy[k,0] == Initial_charge
                for k in range(nbEV)
            ),
            name=f"EVEnergyStart",
        )

        model.addConstrs(
            (
                EVEnergy[k,nbTime-1] >= Terminal_Charge # -1 because index starts from 0
                for k in range(nbEV)
            ),
            name=f"EVEnergyEnd",
        )
        # Energy Balance (*1 because time span is 1hr)
        model.addConstrs(
            (
                EVEnergy[k,t] == EVEnergy[k,t-1] + ((1-loss)*EVChargePower.sum(k,"*",t) - (1+loss)*(EVDischargePower.sum(k,"*",t) + EVMovePower[k,t])) * (24/nbTime)
                for t in range(1,nbTime)
                for k in range(nbEV)
            ),
            name=f"EVEnergyBalance",
        )

        # ========================================================== Generator Constraints ==========================================================
        # add generation limits
        model.addConstrs(
            (
                GenActivePower[u,t] <= gen_max[u]
                for u in range(nbGen)
                for t in range(nbTime)
            ),
            name=f"GeneratorActivePowerMax",
        )

        model.addConstrs(
            (
                GenActivePower[u,t] >= 0 # gen_min[u] 
                for u in range(nbGen)
                for t in range(nbTime)
            ),
            name=f"GeneratorActivePowerMin",
        )

        model.addConstrs(
            (
                GenReactivePower[u,t] <= gen_max[u]
                for u in range(nbGen)
                for t in range(nbTime)
            ),
            name=f"GeneratorReactivePowerMax",
        )

        model.addConstrs(
            (
                GenReactivePower[u,t] >= 0 # gen_min[u] 
                for u in range(nbGen)
                for t in range(nbTime)
            ),
            name=f"GeneratorReactivePowerMin",
        )

        model.addConstrs(
            (
                PVPower[p,t] <= pv_power_max[p,t]
                for p in range(nbPV)
                for t in range(nbTime)
            ),
            name=f"PVPowerMax",
        )

        model.addConstrs(
            (
                PVPower[p,t] >= 0
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
                ActiveLineFlow[l,t] <= line_limit_max[l]
                for l in range(nbLine)
                for t in range(nbTime)
            ),
            name=f"ActiveLinePowerMax",
        )

        model.addConstrs(
            (
                ActiveLineFlow[l,t] >= line_limit_min[l]
                for l in range(nbLine)
                for t in range(nbTime)
            ),
            name=f"ActiveLinePowerMin",
        )

        # add reactive line flow limits
        model.addConstrs(
            (
                ReactiveLineFlow[l,t] <= line_limit_max[l]
                for l in range(nbLine)
                for t in range(nbTime)
            ),
            name=f"ReactiveLinePowerMax",
        )

        model.addConstrs(
            (
                ReactiveLineFlow[l,t] >= line_limit_min[l]
                for l in range(nbLine)
                for t in range(nbTime)
            ),
            name=f"ReactiveLinePowerMin",
        )  

        # add voltage limits
        model.addConstrs(
            (
                BusVolt[i,t] <= v_0*1.05
                for i in range(1,nbBus)
                for t in range(nbTime)
            ),
            name=f"BusVoltMax",
        )

        model.addConstrs(
            (
                BusVolt[i,t] >= v_0*0.95
                for i in range(1,nbBus)
                for t in range(nbTime)
            ),
            name=f"BusVoltMin",
        )

    # ============================================== Line Power and Voltage Calculation ==================================================

        # Calculation of line power by first determining GenPower
        # intialise a counter for each generator
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
                    gap = GenActivePower[c_dg,t] + PVPower[c_pv,t]
                    grp = GenReactivePower[c_dg,t]
                # if the bus only has dg
                elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
                    gap = GenActivePower[c_dg,t]
                    grp = GenReactivePower[c_dg,t]
                # if the bus only has pv
                elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
                    gap = PVPower[c_pv,t]
                    grp = 0
                else:
                    gap = 0
                    grp = 0

                # calculate BusPower
                if (i in cs_map):
                    bap = bus_active_load[i,t] + EVChargePower.sum("*",cs_map.index(i),t) - EVDischargePower.sum("*",cs_map.index(i),t)
                    brp = bus_reactive_load[i,t]
                else:
                    bap = bus_active_load[i,t]
                    brp = bus_reactive_load[i,t]

                # set constraint after determining GenPower and BusPower
                # calculate Line power and system power balance here
                if not len(upbus):
                    # no more up stream bus
                    lhs_0 = gap
                    rhs_0 = ActiveLineFlow.sum(downbus,t) + bap
                    lhs_1 = grp
                    rhs_1 = ReactiveLineFlow.sum(downbus,t) + brp

                elif not len(downbus):
                    # no more down stream bus
                    lhs_0 = gap + ActiveLineFlow[upbus[0],t]
                    rhs_0 = bap
                    lhs_1 = grp + ReactiveLineFlow[upbus[0],t]
                    rhs_1 = brp

                else:
                    # for rest of the cases
                    lhs_0 = gap + ActiveLineFlow[upbus[0],t]
                    rhs_0 = ActiveLineFlow.sum(downbus,t) + bap
                    lhs_1 = grp + ReactiveLineFlow[upbus[0],t]
                    rhs_1 = ReactiveLineFlow.sum(downbus,t) + brp

                # add constraint
                model.addConstr(
                    (
                        lhs_0 == rhs_0
                    ),
                    name=f"ActiveBusLinePower_{i}_{t}",
                )
                model.addConstr(
                    (
                        lhs_1 == rhs_1
                    ),
                    name=f"ReactiveBusLinePower_{i}_{t}",
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
                BusVolt[0,t] == v_0
                for t in range(nbTime)
            ),
            name=f"SlackVoltage_{t}",
        )

        for l in range(nbLine):
            model.addConstrs(
                (
                    BusVolt[line_params[l,0],t] - ((ActiveLineFlow[l,t]*line_params[l,3] + ReactiveLineFlow[l,t]*line_params[l,4])/v_0) - BusVolt[line_params[l,1],t] == 0
                    for t in range(nbTime)
                ),
                name=f"Voltage_{l}_{t}",
            )

        # -----------------------------------------------------------------------------------------------------------------------------------------
        # ========================================================== Objective function setup =========================================================
        # -----------------------------------------------------------------------------------------------------------------------------------------

        expr_GenerationCost = gb.LinExpr()
        expr_TravelCost = gb.LinExpr()
        expr_ChargingCost = gb.LinExpr()
        expr_DischargingEarn = gb.LinExpr()
        
        for u in range(nbGen):
            for t in range(nbTime):
                expr_GenerationCost.add(GenActivePower[u,t] * fuel_cost) 

        non_stationary_index = (EV_routes[:,1] != EV_routes[:,2]).nonzero()[0]
        for k in range(nbEV):
            for s in range(nbTime-1):
                for ij in non_stationary_index:
                    expr_TravelCost.add(EVArcStatus[k,ij,s] * EVTravelCost)

        for k in range(nbEV):
            for i in range(nbCS):
                for t in range(nbTime):
                    expr_ChargingCost.add(EVChargePower[k,i,t] * EV_ChargeCost[t])

        for k in range(nbEV):
            for i in range(nbCS):
                for t in range(nbTime):
                    expr_DischargingEarn.add(EVDischargePower[k,i,t] * EV_DischargeEarn[t])

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
            # model.computeIIS()

            # for c in model.getConstrs():
            #     if c.IISConstr: print(f'\t{c.constrname}: {model.getRow(c)} {c.Sense} {c.RHS}')

            # for v in model.getVars():
            #     if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
            #     if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')

                # break

        else:
            # try:
                print("Objective value: ", model.ObjVal)
                print("Model runtime: ", model.Runtime)   
                print("MIP Gap:", model.MIPGap)

                variables = model.getVars()

                # make sure solution are order based on EV number
                # training_binary_test = []
                # training_idx_test = []

                training_binary = []
                training_idx = []

                for k in range(nbEV):
                    # for each EV get each attribute and order it based on EV number
                    for r in range(nbRoute):
                        for t in range(nbTime-1):
                            var = model.getVarByName(f"EVArcStatus[{k},{r},{t}]")

                            training_binary.append(var.X)
                            training_idx.append(var.index)

                    for c in range(nbCS):
                        for t in range(nbTime):
                            var = model.getVarByName(f"EVChargeStatus[{k},{c},{t}]")

                            training_binary.append(var.X)
                            training_idx.append(var.index)
                            
                    for c in range(nbCS):
                        for t in range(nbTime):
                            var = model.getVarByName(f"EVDischargeStatus[{k},{c},{t}]")

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

                save_path = f"dataGeneration/feature_target/dataPoint_{model_num}.pkl"
                data_path = os.path.join(os.getcwd(), save_path)
                with open(data_path, 'wb') as f:
                    pickle.dump(dataPoint, f)

                model.reset(0)
                save_path = f"dataGeneration/model/coordination_{model_num}.mps"

                model.write(os.path.join(os.getcwd(), save_path))

                model_num = model_num + 1

        # -----------------------------------------------------------------------------------------------------------------------------------------
        # ========================================================== Perform Testing From Here =========================================================
        # -----------------------------------------------------------------------------------------------------------------------------------------

                # comment out below for verifying line power with ptdf
                # create placeholder for total genpower and total buspower (we only check one timestamp)
                # timestamp = 41

                # TotalActiveGenPower = np.zeros((nbBus, 1))
                # TotalActiveBusPower = np.zeros((nbBus, 1))

                # c_dg = 0
                # c_pv = 0

                # for i in range(nbBus):   
                #     # determine required variable first
                #     var0 = model.getVarByName(f"GenActivePower[{c_dg},{timestamp}]")
                #     var1 = model.getVarByName(f"PVPower[{c_pv},{timestamp}]")

                #     charge_sum = []
                #     discharge_sum = [] 
                #     for k in range(nbEV):
                #         if i in cs_map:
                #             var2 = model.getVarByName(f"EVChargePower[{k},{cs_map.index(i)},{timestamp}]")
                #             var3 = model.getVarByName(f"EVDischargePower[{k},{cs_map.index(i)},{timestamp}]")

                #             charge_sum.append(var2.X)
                #             discharge_sum.append(var3.X)

                #     charge_sum = np.sum(charge_sum)
                #     discharge_sum = np.sum(discharge_sum)
                    

                #     # if the bus has both dg and pv
                #     if (i in gen_params[:,0]) and (i in pv_params[:,0]):
                #         gap = var0.X + var1.X
                #     # if the bus only has dg
                #     elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
                #         gap = var0.X 
                #     # if the bus only has pv
                #     elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
                #         gap = var1.X 
                #     else:
                #         gap = 0

                #     if (i in cs_map):
                #         bap = bus_active_load[i,timestamp] + charge_sum - discharge_sum
                #     else:
                #         bap = bus_active_load[i,timestamp]

                #     TotalActiveGenPower[i,0] = gap
                #     TotalActiveBusPower[i,0] = bap

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
                # LinePower = np.zeros((nbLine,1))
                # for m in range(nbLine):
                #     var = model.getVarByName(f"ActiveLineFlow[{m},{timestamp}]")
                #     LinePower[m,0] = var.X
                    

                # GenLoad = TotalActiveGenPower - TotalActiveBusPower
                # linepower = LinePower

                # print(GenLoad.shape)
                # print(linepower.shape)

                # gl_df = pd.DataFrame(GenLoad)
                # gl_df.to_csv(os.path.join(data_dir, 'GenLoad0.csv'))

                # lp_df = pd.DataFrame(linepower)
                # lp_df.to_csv(os.path.join(data_dir, 'line_power.csv'))

                # comment out below for doing other testing
                # TotalActiveGenPower = np.zeros((nbBus, nbTime))
                # TotalActiveBusPower = np.zeros((nbBus, nbTime))
    # 
                # c_dg = 0
                # c_pv = 0
    # 
                # for i in range(nbBus):   
                    # for t in range(nbTime):
                        # determine required variable first
                        # var0 = model.getVarByName(f"GenActivePower[{c_dg},{t}]")
                        # var1 = model.getVarByName(f"PVPower[{c_pv},{t}]")
    # 
                        # charge_sum = []
                        # discharge_sum = [] 
                        # for k in range(nbEV):
                            # if i in cs_map:
                                # var2 = model.getVarByName(f"EVChargePower[{k},{cs_map.index(i)},{t}]")
                                # var3 = model.getVarByName(f"EVDischargePower[{k},{cs_map.index(i)},{t}]")
    # 
                                # charge_sum.append(var2.X)
                                # discharge_sum.append(var3.X)
    # 
                        # charge_sum = np.sum(charge_sum)
                        # discharge_sum = np.sum(discharge_sum)
                        # 
    # 
                        # if the bus has both dg and pv
                        # if (i in gen_params[:,0]) and (i in pv_params[:,0]):
                            # gap = var0.X + var1.X
                        # if the bus only has dg
                        # elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
                            # gap = var0.X 
                        # if the bus only has pv
                        # elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
                            # gap = var1.X 
                        # else:
                            # gap = 0
    # 
                        # if (i in cs_map):
                            # bap = bus_active_load[i,t] + charge_sum - discharge_sum
                        # else:
                            # bap = bus_active_load[i,t]
    # 
                        # TotalActiveGenPower[i,t] = gap
                        # TotalActiveBusPower[i,t] = bap
    # 
                    # increase generator counters
                    # if (i in gen_params[:,0]) and (i in pv_params[:,0]):
                        # c_dg = c_dg + 1
                        # c_pv = c_pv + 1
                    # if the bus only has dg
                    # elif (i in gen_params[:,0]) and not(i in pv_params[:,0]):
                        # c_dg = c_dg + 1
                    # if the bus only has pv
                    # elif not(i in gen_params[:,0]) and (i in pv_params[:,0]):
                        # c_pv = c_pv + 1
                # 
                # for m in range(nbBus):
                    # print(m, np.sum(np.abs(TotalActiveBusPower[m,:]-bus_active_load[m,:])))
            # except:
            #     continue

        model.dispose()

        # sys.exit()


        