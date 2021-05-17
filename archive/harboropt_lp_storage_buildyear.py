import numpy as np # numerical library
import matplotlib.pyplot as plt # plotting library
import datetime as dt
import pandas as pd

from ortools.linear_solver import pywraplp

import utils

##### TO-DO: 
##### 2.Incorporate timespan of storage and apply discount factor to replacement capacity costs (ie. every 15 years). 
##### 3. Require storage to only charge from renewable sources or during off-peak times? See if value of avoided emissions incentivizes this.
##### 4. Split up results functions into specific functions (ex. total_gen, storage_net_source, gen_fractions, curtailment).
        
class LinearProgram(object):
    
    def __init__(self, initial_state_of_charge = 0, storage_life = 15, timespan = 30,
                 gas_fuel_cost=8, discount_rate = 0.06, cost=1, build_years = 1, storage_resilience_incentive_per_kwh = 1000, resilient_storage_grid_fraction = 0.7):
        
        self.initial_state_of_charge = initial_state_of_charge
        self.timespan = timespan
        self.build_years = build_years
        self.resilience_incentive_per_mwh = storage_resilience_incentive_per_kwh * 1000
        self.resilient_storage_grid_fraction = resilient_storage_grid_fraction
        
        #Rate that money decays per year.
        self.discount_rate = discount_rate
        
        self.gas_fuel_cost = gas_fuel_cost
        self.cost = cost
        self.solver = pywraplp.Solver('HarborOptimization',
                         pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        self.resources = self._setup_resources()
        self.capacity_vars = self._initialize_capacity_by_resource(build_years)
        
        self.storage = self._setup_storage()
        self.storage_capacity_vars = self._initialize_storage_capacity_vars(build_years)

        self.disp = self.resources.loc[self.resources['dispatchable'] == 'y']
        self.nondisp = self.resources.loc[self.resources['dispatchable'] == 'n']
        
        self.discounting_factor = self.discount_factor_from_cost(self.cost, self.discount_rate, self.build_years)
        
        
         #Create a dictionary to hold a list for each dispatchable resource that keeps track of its hourly generation variables.
        self.disp_gen = {}
        for resource in self.disp.index:
            self.disp_gen[resource] = []
            
        #Create a dictionary to hold a list for each storage resource that keeps track of its hourly charge variables.
        self.storage_charge_vars = {}
        for resource in self.storage.index:
            self.storage_charge_vars[resource] = []
            
        #Create a dictionary to hold a list for each storage resource that keeps track of its hourly discharge variables.
        self.storage_discharge_vars = {}
        for resource in self.storage.index:
            self.storage_discharge_vars[resource] = []
            
        #Create a dictionary to hold a list for each storage resource that keeps track of its hourly state of charge variables.
        self.storage_state_of_charge_vars = {}
        for resource in self.storage.index:
            self.storage_state_of_charge_vars[resource] = []
        
        
        self.objective = self._add_constraints_and_costs() 
        
             
        
    def _add_constraints_and_costs(self):
        
        #Initialize objective function.
        objective = self.solver.Objective()
        
        #Read in demand and nondispatchable resource profiles.
        profiles = pd.read_csv('data/doscoe_profiles.csv')
        
        for year in range(self.build_years):

            #Initialize hydro energy limit constraint: hydro resources cannot exceed the following energy supply limit in each year.
            hydro_energy_limit = self.solver.Constraint(0, 13808000)

            # Loop through every hour in demand, creating:
            # 1) hourly gen variables for each disp resource 
            # 2) hourly constraints
            # 3) adding variable cost coefficients to each hourly generation variable.
            for ind in profiles.index:

                #Initialize fulfill demand constraint: summed generation from all resources must be equal or greater to demand in all hours.
                #Constraint only applies in the last year of build.
                if year == (self.build_years-1):
                    fulfill_demand = self.solver.Constraint(profiles.loc[ind,'DEMAND'], self.solver.infinity())
                else:
                    fulfill_demand = self.solver.Constraint(0, self.solver.infinity())

                #Initialize hydro power limit constraint: hydro resources cannot exceed the following power supply limit in each hour.
                hydro_power_limit = self.solver.Constraint(0, 9594.8)

                #Create hourly charge and discharge variables for each storage resource and store in respective dictionaries. 
                for resource in self.storage.index:

                    storage_duration = self.storage.loc[resource, 'storage_duration (hrs)']
                    efficiency = self.storage.loc[resource, 'efficiency']    

                    #Create hourly charge and discharge variables for each storage resource in each build year.
                    charge= self.solver.NumVar(0, self.solver.infinity(), '_charge_year'+ str(year) + '_hour' + str(ind))
                    discharge= self.solver.NumVar(0, self.solver.infinity(), '_discharge_year'+ str(year) + '_hour' + str(ind))

                    #Add variable cost of charging (and monetized emissions if uncommented) to objective function.
                    variable_cost = self.storage.loc[resource,'variable ($/MWh)'] #+whole_grid_emissions.loc[ind,'TOTAL/MWH']
                    objective.SetCoefficient(charge, variable_cost)

                    #Limit hourly charge and discharge variables to storage max power (MW). 
                    #Sum storage capacity from previous and current build years to set max power.
                    max_charge= self.solver.Constraint(0, self.solver.infinity())
                    storage_capacity_cumulative = self.storage_capacity_vars[resource][0:year+1]
                    for i, var in enumerate(storage_capacity_cumulative):
                        if self.storage.loc[resource, 'resilient'] == 'y':
                            #For resilient storage, limit max charge to the fraction of capacity set aside for the grid.
                            max_charge.SetCoefficient(var, self.resilient_storage_grid_fraction)
                        else: 
                            max_charge.SetCoefficient(var, 1)
                    max_charge.SetCoefficient(charge, -1)

                    if year == 0 and ind == 0:
                        max_discharge= self.solver.Constraint(0, self.solver.infinity())
                        var = self.storage_capacity_vars[resource][0]
                        if self.storage.loc[resource, 'resilient'] == 'y':
                            #For resilient storage, limit max discharge to the fraction of capacity set aside for the grid.
                            max_discharge.SetCoefficient(var, self.resilient_storage_grid_fraction)
                        else: 
                            max_discharge.SetCoefficient(var, 1)
                        max_discharge.SetCoefficient(discharge, -1)
                        
#                     elif year > 0 and ind == 0:
#                         max_discharge= self.solver.Constraint(0, self.solver.infinity())
#                         storage_capacity_previous_years = self.storage_capacity_vars[resource][0:year]
#                         for i, var in enumerate(storage_capacity_previous_years):
#                             max_discharge.SetCoefficient(var, 1)
#                         max_discharge.SetCoefficient(discharge, -1)

                    elif ind > 0:
                        max_discharge= self.solver.Constraint(0, self.solver.infinity())
                        storage_capacity_cumulative = self.storage_capacity_vars[resource][0:year+1]
                        for i, var in enumerate(storage_capacity_cumulative):
                            if self.storage.loc[resource, 'resilient'] == 'y':
                            #For resilient storage, limit max charge and discharge to the fraction of capacity set aside for the grid.
                                max_discharge.SetCoefficient(var, self.resilient_storage_grid_fraction)
                            else: 
                                max_discharge.SetCoefficient(var, 1)
                        max_discharge.SetCoefficient(discharge, -1)
                    
                        
                    #Keep track of hourly charge and discharge variables by appending to lists for each storage resource.
                    self.storage_charge_vars[resource].append(charge)
                    self.storage_discharge_vars[resource].append(discharge)

                    #Hourly discharge variables of storage resources are incorporated into the fulfill demand constraint. If storage can only charge from portfolio resources, include the charge variable in this constraint.
                    fulfill_demand.SetCoefficient(discharge, efficiency)
                    #Include the line below if storage can only charge from portfolio resources.
                    fulfill_demand.SetCoefficient(charge, -1)

                    #Creates hourly state of charge variable, representing the state of charge at the end of each timestep. 
                    state_of_charge= self.solver.NumVar(0, self.solver.infinity(), 'state_of_charge_year'+ str(year) + '_hour' + str(ind))

                    #Temporal coupling of storage state of charge.
                    if ind > 0:
                        state_of_charge_constraint= self.solver.Constraint(0, 0)
                        state_of_charge_constraint.SetCoefficient(state_of_charge, -1)
                        state_of_charge_constraint.SetCoefficient(discharge, -1)
                        #To-Do: Should coefficient here be "efficiency" to represent lost power during charging?
                        state_of_charge_constraint.SetCoefficient(charge, 1)

                        #Get the state of charge from previous timestep to include in the state_of_charge_constraint.
                        previous_state = self.storage_state_of_charge_vars[resource][-1]
                        if year == 3 and ind == 0:
                            print('previous_state', previous_state)
                        state_of_charge_constraint.SetCoefficient(previous_state, 1)
                        
                    else: 
                        state_of_charge_constraint= self.solver.Constraint(self.initial_state_of_charge, self.initial_state_of_charge)
                        state_of_charge_constraint.SetCoefficient(state_of_charge, 1)
                        state_of_charge_constraint.SetCoefficient(discharge, 1)
                        #To-Do: Should coefficient here be "efficiency" to represent lost power during charging?
                        state_of_charge_constraint.SetCoefficient(charge, -1)

                    #Add hourly state of charge variable to corresponding list for each storage resource.
                    self.storage_state_of_charge_vars[resource].append(state_of_charge)

                    #Creates constraint setting max state of charge to: storage capacity * storage duration.
                    max_storage= self.solver.Constraint(0, self.solver.infinity())
                    storage_capacity_cumulative = self.storage_capacity_vars[resource][0:year+1]
                    for i, var in enumerate(storage_capacity_cumulative):
                        if self.storage.loc[resource, 'resilient'] == 'y':
                            #For resilient storage, limit max storage to the fraction of capacity set aside for the grid * storage_duration.
                            max_storage.SetCoefficient(var, self.resilient_storage_grid_fraction*storage_duration)
                        else: 
                            max_storage.SetCoefficient(var, storage_duration)
                    max_storage.SetCoefficient(state_of_charge, -1)

                    #Creates constraint ensuring that no net energy is supplied by storage (ending state of charge is equal to initial state of charge).
                    #if harborgen.index.get_loc(ind) == len(harborgen)-1:
                    if year == (self.build_years-1) and ind == (len(profiles)-1):
                        ending_state = self.solver.Constraint(self.initial_state_of_charge, self.initial_state_of_charge)
                        ending_state.SetCoefficient(state_of_charge, 1)


                #Loop through dispatchable resources.
                for resource in self.disp.index:

                    #Create generation variable for each dispatchable resource for every hour. 
                    gen = self.solver.NumVar(0, self.solver.infinity(), '_gen_year_'+ str(year) + '_hour' + str(ind))

                    #Append hourly gen variable to the list for that resource, located in the disp_gen dictionary.
                    self.disp_gen[resource].append(gen)

            #         if resource == 'outofbasin':
            #             # TODO: Incorporate transmission cost into variable cost for outofbasin option.
            #             variable_cost = outofbasin_emissions.loc[ind,'TOTAL/MWH']+ disp.loc[resource,'variable']
            #             objective.SetCoefficient(gen, variable_cost)

                    #Calculate variable cost for each dispatchable resource and extrapolate cost to total timespan, accounting for discount rate.
                    if 'NG' in resource:
                        variable_cost = self.disp.loc[resource,'variable']+ (self.disp.loc[resource,'heat_rate']* self.gas_fuel_cost)
                        variable_cost_extrapolated = variable_cost * self.discounting_factor[year]
                    else:
                        variable_cost = self.disp.loc[resource,'variable']
                        variable_cost_extrapolated = variable_cost * self.discounting_factor[year]

                    #Incorporate extrapolated variable cost of hourly gen for each disp resource into objective function.
                    objective.SetCoefficient(gen, variable_cost_extrapolated)

                    #Add hourly gen variables for disp resources to the fulfill_demand constraint.
                    fulfill_demand.SetCoefficient(gen, 1)

                    #For hydro resource, add hourly generation to power limit constraint (resets every hour) and energy limit constraint.
                    if resource in ['HYDROPOWER']:
                        hydro_power_limit.SetCoefficient(gen, 1)
                        hydro_energy_limit.SetCoefficient(gen, 1)

                    #Initialize max_gen constraint: hourly gen must be less than or equal to capacity for each dispatchable resource.
                    max_gen = self.solver.Constraint(0, self.solver.infinity())
                    disp_capacity_cumulative = self.capacity_vars[resource][0:year+1]
                    for i, var in enumerate(disp_capacity_cumulative):
                        max_gen.SetCoefficient(var, 1)
                    max_gen.SetCoefficient(gen, -1)

                #Nondispatchable resources can only generate their hourly profile scaled by nameplate capacity to help fulfill demand.   
                for resource in self.nondisp.index:
                    profile_max = max(profiles[resource])
                    scaling_coefficient = profiles.loc[ind, resource] / profile_max
                    nondisp_capacity_cumulative = self.capacity_vars[resource][0:year+1]

                    for i, var in enumerate(nondisp_capacity_cumulative):
                        fulfill_demand.SetCoefficient(var, scaling_coefficient)

            #Outside of hourly loop, add capex costs to objective function for every disp resource.         
            for resource in self.disp.index:
                capacity = self.capacity_vars[resource][year]
                capex_initial = self.resources.loc[resource, 'capex']
                capex_decline = self.resources.loc[resource, 'annual capex decline']
                capex_now = capex_initial* pow((1-capex_decline), year)
                fixed = self.resources.loc[resource, 'fixed'] * self.discounting_factor[year]
                capex_fixed = capex_now + fixed
                objective.SetCoefficient(capacity, capex_fixed)

            #Outside of hourly loop, add capex costs to objective function for every storage resource.
            for resource in self.storage.index:
                capex_decline = self.storage.loc[resource, 'annual capex decline']
                
                #For resilient storage, subtract resilience incentive from capex cost.
                if self.storage.loc[resource, 'resilient'] == 'y':
                    storage_duration = self.storage.loc[resource, 'storage_duration (hrs)']
                    incentive_per_mwh = self.resilience_incentive_per_mwh
                    incentive_per_mw = incentive_per_mwh * storage_duration
                    
                    #Subtract resilience incentive from storage capex cost.
                    capex_initial = self.storage.loc[resource, 'capex ($/MW)']
                    
                    #Calculate present capex cost based on year and capex decline.
                    capex_now = (capex_initial* pow((1-capex_decline), year)) -  incentive_per_mw
                    if capex_now < 0:
                        capex_now = 0
                    fixed = self.storage.loc[resource, 'fixed ($/MW-year)'] * self.discounting_factor[year]
                    
                else:
                    #Calculate present capex cost based on year and capex decline.
                    capex_initial = self.storage.loc[resource, 'capex ($/MW)']
                    capex_now = capex_initial* pow((1-capex_decline), year)
                    fixed = self.storage.loc[resource, 'fixed ($/MW-year)'] * self.discounting_factor[year]

                #Set capex cost for storage built in this year.
                objective.SetCoefficient(self.storage_capacity_vars[resource][year], capex_now + fixed)
                  

            #Outside of hourly loop, add capex and extrapolated variable costs to obj function for each nondisp resource. 
            for resource in self.nondisp.index: 
                capacity = self.capacity_vars[resource][year]
                fixed = self.nondisp.loc[resource, 'fixed'] * self.discounting_factor[year]
                capex_initial = self.resources.loc[resource, 'capex']
                capex_decline = self.resources.loc[resource, 'annual capex decline']
                capex_now = capex_initial* pow((1-capex_decline), year) 
                capex_fixed = capex_now + fixed

                profile_max = max(profiles[resource])
                profile = profiles[resource] / profile_max
                profile_sum = sum(profile)

                #Sum annual generation for each unit of capacity. Extrapolate to timespan, accounting for discount rate.
                annual_sum_var_cost = self.nondisp.loc[resource,'variable'] * profile_sum
                annual_sum_var_cost_extrapolated = self.discounting_factor[year] * annual_sum_var_cost

                #Add extrapolated variable cost to capex cost for each nondisp resource.
                total_cost_coefficient = annual_sum_var_cost_extrapolated + capex_fixed

                #Add total cost coefficient to nondisp capacity variable in objective function.
                objective.SetCoefficient(capacity, total_cost_coefficient)

        return objective


    def discount_factor_from_cost(self, cost, discount_rate, build_years):
        growth_rate = 1.0 + discount_rate
        
        discount_factor = []       
        for year in range(build_years):

            value_decay_1 = pow(growth_rate, -(self.timespan-year))
            value_decay_2 = pow(growth_rate, -1)
            try:
                extrapolate = cost * (1.0 - value_decay_1) / (1.0-value_decay_2)
            except ZeroDivisionError:
                extrapolate = cost
            discount_factor.append(extrapolate)
        
        return discount_factor
    
    
    def _setup_resources(self):
        resources = pd.read_csv('data/doscoe_resources.csv')
#         exclusion_str = []
#         for i in range(3):
#             if not i == self.cost:
#                 exclusion_str.append("_{}".format(i))
#         exclusion_str = "|".join(exclusion_str)
        
#         resources = resources[resources['resource'].str.contains(exclusion_str) == False]
        resources = resources.set_index('resource')     
#         resources.index = [resource.replace('_'+str(self.cost),'') for resource in resources.index]
        return resources

        
    def _setup_storage(self):
        storage = pd.read_csv('data/storage.csv')
        num_columns = storage.columns[3:]
        storage[num_columns] = storage[num_columns].astype(float)
        storage = storage.set_index('resource')
        
        return storage
    
    def _initialize_capacity_by_resource(self, build_years):
        capacity_by_resource = {}
        for resource in self.resources.index:
            
            capacity_by_build_year = []
            if self.resources.loc[str(resource)]['legacy'] == 'n':
                #Create list of capacity variables for each year of build.
                for year in range(build_years):
                    capacity = self.solver.NumVar(0, self.solver.infinity(), str(resource)+ '_' + str(year))
                    capacity_by_build_year.append(capacity)
                capacity_by_resource[resource] = capacity_by_build_year
            
            #Need to fix code below to deal with legacy resources.
            else:
                existing_mw = self.resources.loc[str(resource)]['existing_mw']
                capacity = self.solver.NumVar(existing_mw, self.solver.infinity(), str(resource))
                capacity_by_resource[resource] = capacity
                
        return capacity_by_resource
    
        
    def _initialize_storage_capacity_vars(self, build_years):
        storage_capacity_vars = {}
        for resource in self.storage.index:
            
            storage_capacity_by_build_year = []
            if self.storage.loc[str(resource)]['legacy'] == 'n':
                #Create list of capacity variables for each year of build.
                for year in range(build_years):
                    capacity = self.solver.NumVar(0, self.solver.infinity(), str(resource)+ '_' + str(year))
                    storage_capacity_by_build_year.append(capacity)
                storage_capacity_vars[resource] = storage_capacity_by_build_year

        return storage_capacity_vars

    def solve(self):
        self.objective.SetMinimization()
        status = self.solver.Solve()
        if status == self.solver.OPTIMAL:
            print("Solver found optimal solution.")
        else:
            print("Solver exited with error code {}".format(status))
            
                   
    def capacity_results(self):
        
        capacity_fractions = {}
        total_capacity = 0
        for resource in self.capacity_vars:
            total_capacity = total_capacity + self.capacity_vars[resource].solution_value()
        for resource in self.capacity_vars:
            fraction_capacity = self.capacity_vars[resource].solution_value() / total_capacity
            capacity_fractions[resource] = fraction_capacity
        
        return capacity_fractions
            
    def gen_results(self):

        profiles = pd.read_csv('data/doscoe_profiles.csv')
        #Sum total annual generation across all resources.
        total_gen = 0
        for resource in self.disp.index:
            summed_gen = 0
            for i_gen in self.disp_gen[str(resource)]:
                summed_gen += i_gen.solution_value()
            total_gen = total_gen + summed_gen

        for resource in self.nondisp.index:
            profile_max = max(profiles[resource])
            summed_gen = sum(profiles[resource]) / profile_max
            capacity = self.capacity_vars[resource].solution_value()
            gen = summed_gen * capacity
            total_gen = total_gen + gen
        
        #If storage can charge from sources outside portfolio, then net supply from storage should be counted towards total generation.
#         for resource in self.storage_capacity_vars:
#             summed_storage_gen = 0
#             for i,hour in enumerate(self.storage_charge_vars[resource]):
#                 charge_var = self.storage_charge_vars[resource][i].solution_value()
#                 discharge_var = self.storage_discharge_vars[resource][i].solution_value()
#                 net = discharge_var - charge_var
#                 summed_storage_gen += net    
#             total_gen = total_gen + summed_storage_gen
                
            
        gen_fractions = {}
        for resource in self.disp.index:
            summed_gen = 0
            for i_gen in self.disp_gen[str(resource)]:
                summed_gen += i_gen.solution_value()
            fraction_generation = summed_gen / total_gen
            gen_fractions[resource] = fraction_generation

        for resource in self.nondisp.index:
            profile_max = max(profiles[resource])
            summed_gen = sum(profiles[resource]) / profile_max
            capacity = self.capacity_vars[resource].solution_value()
            gen = summed_gen * capacity
            fraction_generation = gen / total_gen
            gen_fractions[resource] = fraction_generation
        
        return total_gen #gen_fractions

#Could add other keys to storage results (ex. hourly charge, hourly state of charge, hourly discharge).
    def storage_results():
        
        storage_results = {}
        for resource in self.storage_capacity_vars:
            resource_results_dict = {}
            
            storage_capacity = self.storage_capacity_vars[resource].solution_value()
            resource_results_dict['capacity']= storage_capacity

            #Get hourly net charge values and add to resource_results_dict.
            storage_hourly_net = []
            storage_hourly_charge = []
            storage_hourly_discharge = []
            efficiency = self.storage.loc[resource, 'efficiency']
            for i,hour in enumerate(self.storage_charge_vars[resource]):
                
                charge_var = self.storage_charge_vars[resource][i].solution_value()
                discharge_var = self.storage_discharge_vars[resource][i].solution_value() * efficiency
                
                net = discharge_var - charge_var
                storage_hourly_net.append(net)
                
                storage_hourly_charge.append(charge_var)
                storage_hourly_discharge.append(discharge_var)
            
            resource_results_dict['hourly_net_source']= storage_hourly_net    
            resource_results_dict['hourly_charge']= storage_hourly_charge
            resource_results_dict['hourly_discharge']= storage_hourly_discharge
            
            
            storage_results[resource]=resource_results_dict
        
        return storage_results