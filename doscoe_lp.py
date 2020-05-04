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
        self.discount_rate = discount_rate
        self.gas_fuel_cost = gas_fuel_cost
        self.cost = cost
        self.solver = pywraplp.Solver('HarborOptimization',
                         pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        self.resources = self._setup_resources()
        #Introduce objective object so we can refer to it in the for loop.
        self.capacity_vars = self._initialize_capacity_vars()
        
        self.disp = self.resources.loc[self.resources['dispatchable'] == 'y']
        self.nondisp = self.resources.loc[self.resources['dispatchable'] == 'n']
        #Create a dictionary to hold a list for each dispatchable resource that keeps track of its hourly generation variables.
        self.disp_gen = {}
        for resource in self.disp.index:
            self.disp_gen[resource] = []
            
        self.discounting_factor = self.discount_factor_from_cost(self.discount_rate)
        self.objective = self._add_constraints_and_costs()
        
    def _add_constraints_and_costs(self):
        objective = self.solver.Objective()
        profiles = pd.read_csv('data/doscoe_profiles.csv')
        hydro_limit = self.solver.Constraint(0, 13808000)

        # Loop through every hour, creating 1) hourly generation variables for each 
        # dispatchable resource, 2) hourly constraints, and 3) adding variable cost 
        # coefficients to each hourly generation variable.
        for ind in profiles.index:
            # Summed generation from all resources must be equal or greater to demand in all hours.
            fulfill_demand = self.solver.Constraint(profiles.loc[ind,'DEMAND'], self.solver.infinity())

            # Create generation variable for each dispatchable resource for every hour. Append hourly gen variable to the list for that resource, located in the disp_gen dictionary.
            #Create constraint that generation must be less than or equal to capacity for each dispatchable resource for all hours.
            for resource in self.disp.index:

                gen = self.solver.NumVar(0, self.solver.infinity(), '_gen'+ str(ind))
                self.disp_gen[resource].append(gen)
        #         if resource == 'outofbasin':
        #             # TODO: Incorporate transmission cost into variable cost for outofbasin option.
        #             variable_cost = outofbasin_emissions.loc[ind,'TOTAL/MWH']+ disp.loc[resource,'variable']
        #             objective.SetCoefficient(gen, variable_cost)
                if 'NG' in resource:
                    variable_cost = (self.disp.loc[resource,'variable']+ (self.disp.loc[resource,'heat_rate']* self.gas_fuel_cost)) * self.discounting_factor
                else:
                    variable_cost = self.disp.loc[resource,'variable'] * self.discounting_factor
                objective.SetCoefficient(gen, variable_cost)
                #Set coefficients for the hourly gen variables for the fulfill_demand constraint.
                fulfill_demand.SetCoefficient(gen, 1)
                #Set coefficients for dispatchable capacity variables and hourly gen variables for the max_gen = capacity constraint. 
                #For legacy resources, contrains maximum hourly generation to existing capacity.
                max_gen = self.solver.Constraint(0, self.solver.infinity())
                capacity = self.capacity_vars[resource]
                max_gen.SetCoefficient(capacity, 1)
                max_gen.SetCoefficient(gen, -1)

                if 'HYDRO' in resource:
                    hydro_limit.SetCoefficient(gen, 1)
    
            #For each nondispatchable resource, set the coefficient of the capacity variable to its generation profile scaling factor. **Make sure units are aligned here (kw vs. mw capacities)
            for resource in self.nondisp.index: 
                capacity = self.capacity_vars[resource]
                profile_max = max(profiles[resource])
                coefficient = profiles.loc[ind, resource] / profile_max
                fulfill_demand.SetCoefficient(capacity, coefficient)

                variable_cost = self.nondisp.loc[resource,'variable'] * self.discounting_factor
                objective.SetCoefficient(capacity, coefficient * variable_cost)
        for resource in self.resources.index:
            capex = self.resources.loc[resource, 'capex']
            fixed = self.resources.loc[resource, 'fixed'] * self.discounting_factor
            objective.SetCoefficient(self.capacity_vars[resource], capex + fixed)
        return objective


    def discount_factor_from_cost(self, discount_rate):
        growth_rate = 1.0 + discount_rate
        value_decay_1 = pow(growth_rate, -self.timespan)
        value_decay_2 = pow(growth_rate, -1)
        discounting_factor = (1.0 - value_decay_1) / (1.0-value_decay_2)
        return discounting_factor
    
    def _setup_resources(self):
        resources = pd.read_csv('data/doscoe_resources.csv')
        exclusion_str = []
        for i in range(3):
            if not i == self.cost:
                exclusion_str.append("_{}".format(i))
        exclusion_str = "|".join(exclusion_str)
        
        resources = resources[resources['resource'].str.contains(exclusion_str) == False]
        resources = resources.set_index('resource')     
        resources.index = [resource.replace('_'+str(self.cost),'') for resource in resources.index]
        return resources
    
    def _initialize_capacity_vars(self):
        capacity_vars = {}
        for resource in self.resources.index:
            if self.resources.loc[str(resource)]['legacy'] == 'n':
                capacity = self.solver.NumVar(0, self.solver.infinity(), str(resource))
                capacity_vars[resource] = capacity
            else:
                max_hydro = self.resources.loc[str(resource)]['existing_mw']
                capacity = self.solver.NumVar(0, max_hydro, str(resource))
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