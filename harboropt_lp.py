import numpy as np # numerical library
import matplotlib.pyplot as plt # plotting library
import datetime as dt
import pandas as pd

from ortools.linear_solver import pywraplp

import utils
        
class DOSCOE(object):
    
    def __init__(self, battery_duration = 4, initial_state_of_charge = 0, timespan = 30,
                 gas_fuel_cost=4, discount_rate = 0.06, cost=1):
        
        self.battery_duration = battery_duration
        self.initial_state_of_charge = initial_state_of_charge
        self.timespan = timespan
        
        #Rate that money decays per year.
        self.discount_rate = discount_rate
        
        self.gas_fuel_cost = gas_fuel_cost
        self.cost = cost
        self.solver = pywraplp.Solver('HarborOptimization',
                         pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        self.resources = self._setup_resources()
        self.capacity_vars = self._initialize_capacity_vars()
        
        self.disp = self.resources.loc[self.resources['dispatchable'] == 'y']
        self.nondisp = self.resources.loc[self.resources['dispatchable'] == 'n']
        
        #Create a dictionary to hold a list for each dispatchable resource that keeps track of its hourly generation variables.
        self.disp_gen = {}
        for resource in self.disp.index:
            self.disp_gen[resource] = []
            
        self.discounting_factor = self.discount_factor_from_cost(self.cost, self.discount_rate)
        
        self.objective = self._add_constraints_and_costs()  
        
    def _add_constraints_and_costs(self):
        
        #Initialize objective function.
        objective = self.solver.Objective()
        
        #Read in demand and nondispatchable resource profiles.
        profiles = pd.read_csv('data/doscoe_profiles.csv')
        
        #Initialize hydro energy limit constraint: hydro resources cannot exceed the following energy supply limit in each year.
        hydro_energy_limit = self.solver.Constraint(0, 13808000)

        # Loop through every hour in demand, creating 1) hourly generation variables for each dispatchable resource, 2) hourly constraints, and 3) adding variable cost coefficients to each hourly generation variable.
        for ind in profiles.index:
            
            #Initialize fulfill demand constraint: summed generation from all resources must be equal or greater to demand in all hours.
            fulfill_demand = self.solver.Constraint(profiles.loc[ind,'DEMAND'], self.solver.infinity())
            
            #Initialize hydro power limit constraint: hydro resources cannot exceed the following power supply limit in each hour.
            hydro_power_limit = self.solver.Constraint(0, 9594.8)


            #Loop through dispatchable resources.
            for resource in self.disp.index:
                
                #Create generation variable for each dispatchable resource for every hour. 
                gen = self.solver.NumVar(0, self.solver.infinity(), '_gen_hour_'+ str(ind))
                
                #Append hourly gen variable to the list for that resource, located in the disp_gen dictionary.
                self.disp_gen[resource].append(gen)
                
        #         if resource == 'outofbasin':
        #             # TODO: Incorporate transmission cost into variable cost for outofbasin option.
        #             variable_cost = outofbasin_emissions.loc[ind,'TOTAL/MWH']+ disp.loc[resource,'variable']
        #             objective.SetCoefficient(gen, variable_cost)
                
                #Calculate variable cost for each dispatchable resource and extrapolate cost to total timespan, accounting for discount rate.
                if 'NG' in resource:
                    variable_cost = self.disp.loc[resource,'variable']+ (self.disp.loc[resource,'heat_rate']* self.gas_fuel_cost)
                    variable_cost_extrapolated = variable_cost * self.discounting_factor
                else:
                    variable_cost = self.disp.loc[resource,'variable']
                    variable_cost_extrapolated = variable_cost * self.discounting_factor
                
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
                capacity = self.capacity_vars[resource]
                max_gen.SetCoefficient(capacity, 1)
                max_gen.SetCoefficient(gen, -1)
            
            #Nondispatchable resources can only generate their hourly profile scaled by nameplate capacity to help fulfill demand.   
            for resource in self.nondisp.index:
                capacity = self.capacity_vars[resource]
                profile_max = max(profiles[resource])
                scaling_coefficient = profiles.loc[ind, resource] / profile_max
                
                fulfill_demand.SetCoefficient(capacity, scaling_coefficient)
                 
        #Outside of hourly loop, add capex costs to objective function for every disp resource.       
        for resource in self.disp.index:
            capacity = self.capacity_vars[resource]
            capex = self.resources.loc[resource, 'capex']
            #fixed = self.resources.loc[resource, 'fixed'] * self.discounting_factor
            objective.SetCoefficient(capacity, capex) #+ fixed)
            
        #Outside of hourly loop, add capex and extrapolated variable costs to obj function for each nondisp resource. 
        for resource in self.nondisp.index: 
            capacity = self.capacity_vars[resource]
            capex = self.resources.loc[resource, 'capex']
            
            profile_max = max(profiles[resource])
            profile = profiles[resource] / profile_max
            profile_sum = sum(profile)
            
            #Sum annual generation for each unit of capacity. Extrapolate to timespan, accounting for discount rate.
            annual_sum_var_cost = self.nondisp.loc[resource,'variable'] * profile_sum
            annual_sum_var_cost_extrapolated = self.discounting_factor * annual_sum_var_cost
            
            #Add extrapolated variable cost to capex cost for each nondisp resource.
            total_cost_coefficient = annual_sum_var_cost_extrapolated + capex

            #Add total cost coefficient to nondisp capacity variable in objective function.
            objective.SetCoefficient(capacity, total_cost_coefficient)
        
        return objective


    def discount_factor_from_cost(self, cost, discount_rate):
        growth_rate = 1.0 + discount_rate
        value_decay_1 = pow(growth_rate, -self.timespan)
        value_decay_2 = pow(growth_rate, -1)
        try:
            return cost * (1.0 - value_decay_1) / (1.0-value_decay_2)
        except ZeroDivisionError:
            return cost
    
    
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
    
    def _initialize_capacity_vars(self):
        capacity_vars = {}
        for resource in self.resources.index:
            if self.resources.loc[str(resource)]['legacy'] == 'n':
                capacity = self.solver.NumVar(0, self.solver.infinity(), str(resource))
                capacity_vars[resource] = capacity
            else:
                existing_mw = self.resources.loc[str(resource)]['existing_mw']
                capacity = self.solver.NumVar(existing_mw, self.solver.infinity(), str(resource))
                capacity_vars[resource] = capacity
                
        return capacity_vars

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
        
        return gen_fractions