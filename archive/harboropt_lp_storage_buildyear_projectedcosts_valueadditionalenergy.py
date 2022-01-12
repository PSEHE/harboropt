import numpy as np # numerical library
import matplotlib.pyplot as plt # plotting library
import datetime as dt
import pandas as pd

from ortools.linear_solver import pywraplp

import utils

##### TO-DO: 

##### 2. Make changes to model based on conversation with Patrick. Upper limits on solar & EE based on how they would meet the model alone. Currently upper limit on all resources is 400 MW. Change this method to Elena's method of a constraint in hour of highest EE savings or solar production.
##### 3. For EE and solar and nondisp resources: Incorporate avoided transmission and distribution costs for energy generated to meet Harbor.
##### 6. Incorporate demand response.
##### Change storage grid-average emissions to marginal emissions.
##### 3. Resource potential constraints.
##### 4. Retirement of Harbor — cannot continue to produce electricity for 30 years. Need to incorporate this deadline. Merge the Harbor & new gas plant option as a replacement cost for Harbor?
##### 1. Inflation —— make sure all costs are in correct dollar year ($2018). NREL ATB projected costs are in 2018 real dollars. Need to incorporate inflation rate for EE costs. Need to confirm that inflation costs are already incorporated into projected fixed and variable costs from NREL. Need to inflate costs that are inputs (ex. marginal generation energy cost).
##### 3. Incorporate paired resilient solar+storage systems based on REopt apartment building discussion.
##### 4. Diesel genset emissions: change code to read in diesel genset parameters rather than set as inputs. See notes from conversation with Patrick. Use updated dataset from Lisa Ramos. Need yearly fuel costs. Run REOpt using reference building to figure out paired solar + storage for replacing diesel gensets.



#Nice-to-have:
##### 2. Confirm that transmission cost should be scaled with capacity of outofbasin resources. 
##### 5. Resources are currently valued for the generation they provide over a 30-year timespan. Value the residual energy they generate after this window? Ex. if something is built in build-year 10, value the residual energy generated it generates for the 10 years after the portfolio timespan is complete?
##### 1. Fix storage cost — should variable cost include cost of electricity if storage charges from grid?
##### 4. Incorporate cost of PM2.5 and PM10 to hourly grid emissions and Harbor emissions. Also make sure that the same pollutants are evaluated across resources. 
##### 9. Make sure "existing_mw" for legacy resources is incorporated correctly.
##### 10. If cost_projection assumption is anything other than 'moderate', need to find "advanced" and "conservative" cost projetion estimates for EE resources.

        
class LinearProgram(object):
    
    def __init__(self, selected_resource = 'all', initial_state_of_charge = 0, storage_lifespan = 15, portfolio_timespan = 30, storage_can_charge_from_grid = False, discount_rate = 0.03, cost=1, build_years = 1, health_cost_range = 'HIGH', cost_projections = 'moderate', build_start_year =2018, ee_cost_type = 'total_cost', transmission_capex_cost_per_mw = 72138, transmission_annual_cost_per_mw = 8223, storage_resilience_incentive_per_kwh = 1000, resilient_storage_grid_fraction = 0.7, social_cost_carbon_short_ton = 46.3, avoided_marginal_generation_cost_per_mwh = 36.60, diesel_genset_carbon_per_mw = 0, diesel_genset_pm25_per_mw = 0, diesel_genset_nox_per_mw = 0, diesel_genset_so2_per_mw = 0, diesel_genset_pm10_per_mw = 0, diesel_genset_fixed_cost_per_mw_year = 35000, diesel_genset_mmbtu_per_mwh =0, diesel_genset_cost_per_mmbtu = 20, diesel_genset_run_hours_per_year = 24):#,gas_fuel_cost=4):
        
        self.selected_resource = selected_resource
        self.initial_state_of_charge = initial_state_of_charge
        self.storage_can_charge_from_grid = storage_can_charge_from_grid
        self.timespan = portfolio_timespan
        self.storage_lifespan = storage_lifespan
        self.build_years = build_years
        self.build_start_year = build_start_year
        self.ee_cost_type = ee_cost_type
        self.transmission_capex_cost_per_mw = transmission_capex_cost_per_mw
        self.transmission_annual_cost_per_mw = transmission_annual_cost_per_mw
        
        self.resilience_incentive_per_mwh = storage_resilience_incentive_per_kwh * 1000
        self.resilient_storage_grid_fraction = resilient_storage_grid_fraction
        
        self.health_costs = health_cost_range
        self.cost_projections = cost_projections
        
        self.social_cost_carbon_short_ton = social_cost_carbon_short_ton
        
        self.avoided_marginal_generation_cost_per_mwh = avoided_marginal_generation_cost_per_mwh
#         self.avoided_marginal_transmission_cost_per_mwh = avoided_marginal_transmission_cost_per_mwh
#         self.avoided_marginal_distribution_cost_per_mwh = avoided_marginal_distribution_cost_per_mwh
        
        
#Yearly emissions per diesel genset mw-equivalent.
        self.diesel_genset_carbon_per_mw = diesel_genset_carbon_per_mw
        self.diesel_genset_pm25_per_mw = diesel_genset_pm25_per_mw
        self.diesel_genset_nox_per_mw = diesel_genset_nox_per_mw
        self.diesel_genset_so2_per_mw = diesel_genset_so2_per_mw
        self.diesel_genset_pm10_per_mw = diesel_genset_pm10_per_mw
        self.diesel_genset_fixed_cost_per_mw_year = diesel_genset_fixed_cost_per_mw_year

#These need to be per mw values
        self.diesel_genset_mmbtu_per_mwh = diesel_genset_mmbtu_per_mwh
        self.diesel_genset_cost_per_mmbtu = diesel_genset_cost_per_mmbtu
        self.diesel_genset_run_hours_per_year = diesel_genset_run_hours_per_year

        
        #Rate that money decays per year.
        self.discount_rate = discount_rate
        self.growth_rate = 1 + discount_rate
        #Abbrevation for discount rate, used to align health cost discount rate with
        self.discount_rate_abbrev = str(self.discount_rate)[-1:]+'pct'
        
        #self.gas_fuel_cost = gas_fuel_cost
        self.cost = cost
        self.solver = pywraplp.Solver('HarborOptimization',
                         pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        self.resources = self._setup_resources()
        self.resource_costs = self._setup_resource_costs()
        
        self.capacity_vars = self._initialize_capacity_by_resource(build_years)
        
        self.storage = self._setup_storage()
        self.storage_capacity_vars = self._initialize_storage_capacity_vars(build_years)

        self.disp = self.resources.loc[self.resources['dispatchable'] == 'y']
        self.nondisp = self.resources.loc[self.resources['dispatchable'] == 'n']
        
        self.wholegrid_emissions = self._setup_wholegrid_emissions()
        self.health_costs_emissions_la = self._setup_health_costs_emissions_la()
        
        #Set up health cost of pollutants emitted in LA.
        
        discount_rate_inds = self.health_costs_emissions_la['discount_rate'] == self.discount_rate
        la_inds = self.health_costs_emissions_la['county'] == 'LA'
        pm25_inds = self.health_costs_emissions_la['pollutant'] == 'PM2.5' 
        so2_inds = self.health_costs_emissions_la['pollutant'] == 'SO2'
        nox_inds = self.health_costs_emissions_la['pollutant'] == 'NOx'
        
        if self.health_costs == 'HIGH':
            self.pm25_cost_short_ton_la = self.health_costs_emissions_la[discount_rate_inds & la_inds & pm25_inds]['US_HIGH_annual ($/ton)'].iloc[0]*-1

            self.so2_cost_short_ton_la = self.health_costs_emissions_la[discount_rate_inds & la_inds & so2_inds]['US_HIGH_annual ($/ton)'].iloc[0]*-1

            self.nox_cost_short_ton_la = self.health_costs_emissions_la[discount_rate_inds & la_inds & nox_inds]['US_HIGH_annual ($/ton)'].iloc[0]*-1

            
        if self.health_costs == 'LOW':
            self.pm25_cost_short_ton_la = self.health_costs_emissions_la[discount_rate_inds & la_inds & pm25_inds]['US_LOW_annual ($/ton)'].iloc[0]*-1

            self.so2_cost_short_ton_la = self.health_costs_emissions_la[discount_rate_inds & la_inds & so2_inds]['US_LOW_annual ($/ton)'].iloc[0]*-1

            self.nox_cost_short_ton_la = self.health_costs_emissions_la[discount_rate_inds & la_inds & nox_inds]['US_LOW_annual ($/ton)'].iloc[0]*-1
  
            
        #self.outofbasin_emissions = self._setup_outofbasin_emissions()
        
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
            
        #Creates nested dictionary to hold cost information for each resource in order to calculate LCOE.
        self.lcoe_dict = {}
        
        #Creates nested dictionary to hold cost information (w/ co-benefits) for each resource in order to calculate LCOE.
        self.lcoe_dict_w_cobenefits = {}
        
        self.objective = self._add_constraints_and_costs() 
        
              
    def _add_constraints_and_costs(self):
        
        #Initialize objective function.
        objective = self.solver.Objective()
        
        #Read in Harbor hourly generation and emissions profile.
        harbor = pd.read_csv('data/harbor_hourly_gen_emissions_2019.csv')
        harbor = harbor.fillna(0)
        
        #Read in nondispatchable resource profiles.
        profiles = pd.read_csv('data/gen_profiles.csv')
        
        for year in range(self.build_years):
            
            cost_year = self.build_start_year + year
            self.lcoe_dict[year]={}
            self.lcoe_dict_w_cobenefits[year]={}
            
            #Outside of hourly loop, add capex costs and extrapolated fixed costs to obj function for each nondisp resource. 
            for resource in self.nondisp.index: 
                self.lcoe_dict[year][resource]={}
                self.lcoe_dict_w_cobenefits[year][resource]={}
                
                capacity = self.capacity_vars[resource][year]
                
                resource_inds = self.resource_costs['resource']==resource
                capex_cost_inds = self.resource_costs['cost_type']=='capex_per_kw'
                fixed_cost_inds = self.resource_costs['cost_type']=='fixed_per_kw_year'
                              
                #Extrapolate fixed costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                discount_factor = pow(self.growth_rate, -(year))
                fixed_mw_extrapolated = self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000 * self.discounting_factor[year] * discount_factor
                
                if resource == 'utility_solar_outofbasin':
                    #*** Need to apply inflation rate to this transmission cost.
                    discount_factor = pow(self.growth_rate, -(year))
                    transmission_annual_extrapolated = self.transmission_annual_cost_per_mw * self.discounting_factor[year] * discount_factor
                    fixed_mw_extrapolated = fixed_mw_extrapolated + transmission_annual_extrapolated
                    
                capex_mw = self.resource_costs.loc[capex_cost_inds & resource_inds, str(cost_year)].item()*1000 * discount_factor

                weighted_avg_eul_inds = self.resource_costs['cost_type']=='weighted_avg_eul'
                resource_weighted_avg_eul_inds = self.resource_costs.loc[weighted_avg_eul_inds & resource_inds]
                
                #If nondisp resource is one with an effective useful life in the resource_projected_costs df, incorporate replacement costs.
                if not resource_weighted_avg_eul_inds.empty:
                    
                    weighted_avg_eul = round(self.resource_costs.loc[weighted_avg_eul_inds & resource_inds, str(cost_year)].item())
                    number_of_replacements = int((self.timespan -1 -year)/ weighted_avg_eul)

                    ## Calculate replacement capex costs in future years, apply discount rate, and add to original capex cost.
                    for i in range(number_of_replacements):

                        replacement_year = int(((i+1) * weighted_avg_eul) + cost_year)
                        
                        replacement_capex = self.resource_costs.loc[capex_cost_inds & resource_inds, str(replacement_year)].item()*1000

                        #Calculate discounting factor to apply to capex in the given replacement year.
                        discount_factor = pow(self.growth_rate, -(replacement_year-self.build_start_year))
                        replacement_capex_discounted =  replacement_capex * discount_factor

                        capex_mw = capex_mw + replacement_capex_discounted
                    
               
                #Add capex cost for given build year to lcoe dictionary.
                self.lcoe_dict[year][resource]['capex']=capex_mw
                self.lcoe_dict_w_cobenefits[year][resource]['capex']=capex_mw
                
                #Add fixed costs extrapolated over portfolio timespan to lcoe dictionary.
                self.lcoe_dict[year][resource]['fixed_extrapolated']=fixed_mw_extrapolated
                self.lcoe_dict_w_cobenefits[year][resource]['fixed_extrapolated']=fixed_mw_extrapolated
                
                capex_fixed = capex_mw + fixed_mw_extrapolated
                
                if resource == 'utility_solar_outofbasin':
                    #*** Need to apply inflation rate to this transmission cost.
                    discount_factor = pow(self.growth_rate, -(year))
                    transmission_cost = self.transmission_capex_cost_per_mw * discount_factor
                    capex_fixed = capex_fixed + transmission_cost 

                #Add total cost coefficient to nondisp capacity variable in objective function.
                objective.SetCoefficient(capacity, capex_fixed)
                
        
            #Within build year loop but outside of hourly loop, add capex and fixed costs to objective function for every disp resource.         
            for resource in self.disp.index:
                
                self.lcoe_dict[year][resource]={}
                self.lcoe_dict_w_cobenefits[year][resource]={}
                capacity = self.capacity_vars[resource][year]
                
                resource_inds = self.resource_costs['resource']==resource
                capex_cost_inds = self.resource_costs['cost_type']=='capex_per_kw'
                
                capex_cost_per_mw = self.resource_costs.loc[capex_cost_inds & resource_inds, str(cost_year)].item()*1000 * discount_factor
                
                #Add capex cost for the given build year to lcoe dictionary.
                self.lcoe_dict[year][resource]['capex']=capex_cost_per_mw
                self.lcoe_dict_w_cobenefits[year][resource]['capex']=capex_cost_per_mw
                
                fixed_cost_inds = self.resource_costs['cost_type']=='fixed_per_kw_year'
                
                #Extrapolate fixed costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                discount_factor = pow(self.growth_rate, -(year))
                fixed_mw_extrapolated = self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000 * self.discounting_factor[year] * discount_factor
                
                #Add fixed costs accumulated over chosen timespan to lcoe dictionary.
                self.lcoe_dict[year][resource]['fixed_extrapolated']=fixed_mw_extrapolated
                self.lcoe_dict_w_cobenefits[year][resource]['fixed_extrapolated']=fixed_mw_extrapolated
                
                capex_fixed = capex_cost_per_mw + fixed_mw_extrapolated
                objective.SetCoefficient(capacity, capex_fixed)

            #Within build year loop but outside of hourly loop, add capex costs to objective function for every storage resource.
            for resource in self.storage.index:
                
                resource_inds = self.resource_costs['resource']==resource
                capex_cost_inds = self.resource_costs['cost_type']=='capex_per_kw'
                fixed_cost_inds = self.resource_costs['cost_type']=='fixed_per_kw_year'
                
                discount_factor = pow(self.growth_rate, -(year))
                
                #For resilient storage, subtract resilience incentive from capex cost.
                if self.storage.loc[resource, 'resilient'] == 'y':
                    
                    storage_duration = self.storage.loc[resource, 'storage_duration (hrs)']
                    incentive_per_mwh = self.resilience_incentive_per_mwh
                    incentive_per_mw = incentive_per_mwh * storage_duration
                    
                    #Calculate capex cost minus resilience incentive.
                    capex_per_mw = self.resource_costs.loc[capex_cost_inds & resource_inds, str(cost_year)].item()*1000
    
                    capex_per_mw = (capex_per_mw -  incentive_per_mw) * discount_factor
                    if capex_per_mw < 0:
                         capex_per_mw = 0
                    
                    #Extrapolate fixed costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                    fixed_mw_extrapolated = self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000 * self.discounting_factor[year] * discount_factor
                    
                else:
                    capex_per_mw = self.resource_costs.loc[capex_cost_inds & resource_inds, str(cost_year)].item()*1000 * discount_factor
                    fixed_mw_extrapolated = self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000 * self.discounting_factor[year] * discount_factor
                    

                ## Uncomment the following block once diesel emissions are figured out.
                if resource == 'diesel_genset_replacement_storage_4hr':
                    diesel_genset_monetized_emissions_yearly = (self.diesel_genset_carbon_per_mw * self.social_cost_carbon_short_ton) + self.diesel_genset_pm25_per_mw * self.pm25_cost_short_ton_la + self.diesel_genset_nox_per_mw * self.nox_cost_short_ton_la + self.diesel_genset_so2_per_mw * self.so2_cost_short_ton_la #+ self.diesel_genset_pm10_per_mw * self.pm10_cost_per_ton 
                    
                    diesel_genset_fixed_cost_extrapolated = self.diesel_genset_fixed_cost_per_mw_year * self.discounting_factor[year]* discount_factor
        
                    diesel_genset_fuel_cost_yearly = self.diesel_genset_mmbtu_per_mwh * self.diesel_genset_cost_per_mmbtu * self.diesel_genset_run_hours_per_year
                    
                    monetized_emissions_saved_extrapolated = diesel_genset_monetized_emissions_yearly * self.discounting_factor[year] * discount_factor
                    fuel_costs_saved_extrapolated = diesel_genset_fuel_cost_yearly * self.discounting_factor[year]*discount_factor
                    
                    capex_per_mw =  capex_per_mw - monetized_emissions_saved_extrapolated - fuel_costs_saved_extrapolated
                    fixed_mw_extrapolated = fixed_mw_extrapolated - diesel_genset_fixed_cost_extrapolated
                    
                
                #For all storage resources, incorporate replacement capex costs after storage lifespan.
                number_of_storage_replacements = int((self.timespan -1 -year)/ self.storage_lifespan)

                ## Calculate replacement capex costs in future years, apply discount rate, and add to original capex cost.
                for i in range(number_of_storage_replacements):
                    
                    storage_replacement_year = int(((i+1) * self.storage_lifespan) + cost_year)

                    storage_replacement_capex = self.resource_costs.loc[capex_cost_inds & resource_inds, str(storage_replacement_year)].item()*1000

                    #Calculate discounting factor to apply to capex in the given replacement year.
                    discount_factor = pow(self.growth_rate, -(storage_replacement_year-self.build_start_year))
                    storage_replacement_capex_discounted =  storage_replacement_capex * discount_factor

                    capex_per_mw = capex_per_mw + storage_replacement_capex_discounted
                    

                #Set capex cost for storage built in this year.
                objective.SetCoefficient(self.storage_capacity_vars[resource][year], capex_per_mw + fixed_mw_extrapolated)
            
                
            # Loop through every hour in demand, creating:
            # 1) hourly gen variables for each disp resource 
            # 2) hourly constraints
            # 3) adding variable cost coefficients to each hourly generation variable.
            for ind in profiles.index:

                #Initialize fulfill demand constraint: summed generation from all resources must be equal or greater to demand in all hours.
                
                #Include the line below if generation only needs to meet demand in the last year of build.
                if year == (self.build_years-1):
                    fulfill_demand = self.solver.Constraint(harbor.loc[ind,'load_mw'], self.solver.infinity())
                else:
                    fulfill_demand = self.solver.Constraint(0, self.solver.infinity())
                    
                #If storage can only charge from portfolio of resources, initialize constraint that storage can only charge from dispatchable generation or solar (not energy efficiency).
                if self.storage_can_charge_from_grid == False:
                    storage_charge = self.solver.Constraint(0, self.solver.infinity())
                    
                
                #Calculate monetized grid emissions for each hour of the year — health impacts for NOx and SO2 and social cost of carbon for CO2.
                if self.health_costs == 'LOW':
                    grid_monetized_emissions_per_mwh = self.wholegrid_emissions.loc[ind, 'so2_cost_per_mwh_LOW_'+self.discount_rate_abbrev] + self.wholegrid_emissions.loc[ind, 'nox_cost_per_mwh_LOW_'+ self.discount_rate_abbrev] + (self.wholegrid_emissions.loc[ind,'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton)
                elif self.health_costs == 'HIGH':
                    grid_monetized_emissions_per_mwh = self.wholegrid_emissions.loc[ind, 'so2_cost_per_mwh_HIGH_'+self.discount_rate_abbrev] + self.wholegrid_emissions.loc[ind, 'nox_cost_per_mwh_HIGH_'+self.discount_rate_abbrev]+ (self.wholegrid_emissions.loc[ind,'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton)
                    
                
                #Calculate value of additional energy generated in this hour of the year. This includes avoided generation, transmission, and distribution costs. 
                #*** Need to apply inflation rate to this value?
                value_additional_energy_mwh = self.avoided_marginal_generation_cost_per_mwh + grid_monetized_emissions_per_mwh
                
                #Create dummy variable so that a scalar coefficient can be added to the objective function in each hour. This is part of the equation incorporating the value of additional energy generated.
                dummy_variable= self.solver.NumVar(1, 1, 'dummy_variable'+str(year)+str(ind))
                dummy_variable_coeff = value_additional_energy_mwh * harbor.loc[ind,'load_mw']

                objective.SetCoefficient(dummy_variable, dummy_variable_coeff)
                
                #Create hourly charge and discharge variables for each storage resource and store in respective dictionaries. 
                for resource in self.storage.index:

                    storage_duration = self.storage.loc[resource, 'storage_duration (hrs)']
                    efficiency = self.storage.loc[resource, 'efficiency']    

                    #Create hourly charge and discharge variables for each storage resource in each build year.
                    charge= self.solver.NumVar(0, self.solver.infinity(), resource + '_charge_year'+ str(year) + '_hour' + str(ind))
                    discharge= self.solver.NumVar(0, self.solver.infinity(), resource + '_discharge_year'+ str(year) + '_hour' + str(ind))

                    if self.storage_can_charge_from_grid == False:
                        storage_charge.SetCoefficient(charge, -1)
                    
                    #Add variable cost of charging to objective function. If storage charges from grid, add monetized grid emissions to variable cost. 
                    cost_type_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                    resource_inds = self.resource_costs['resource']==resource
                    
                    if self.storage_can_charge_from_grid:
                        variable_cost = self.resource_costs.loc[cost_type_inds & resource_inds, str(cost_year)].item() + grid_monetized_emissions_per_mwh
                    else:
                        variable_cost = self.resource_costs.loc[cost_type_inds & resource_inds, str(cost_year)].item()

                    variable_cost_with_value_additional_energy = variable_cost + value_additional_energy_mwh
                    
                    #If not in the last build_year, don't extrapolate variable costs. Just discount back to build_start_year. If in the last build year, extrapolate variable costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                    discount_factor = pow(self.growth_rate, -(year))
                    if year < (self.build_years-1):
                        variable_cost_with_value_additional_energy = variable_cost_with_value_additional_energy * discount_factor
                    else:
                        variable_cost_with_value_additional_energy = variable_cost_with_value_additional_energy * self.discounting_factor[year] * discount_factor
                    
                    
                    objective.SetCoefficient(charge, variable_cost_with_value_additional_energy)
                    objective.SetCoefficient(discharge, -value_additional_energy_mwh)

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
                    
                    #Include the line below if storage cannot charge from grid (and can only charge from portfolio resources).
                    if self.storage_can_charge_from_grid == False:
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
                    if year == (self.build_years-1) and ind == (len(profiles)-1):
                        ending_state = self.solver.Constraint(self.initial_state_of_charge, self.initial_state_of_charge)
                        ending_state.SetCoefficient(state_of_charge, 1)


                #Loop through dispatchable resources.
                for resource in self.disp.index:
                    
                    resource_inds = self.resource_costs['resource']==resource

                    #Create generation variable for each dispatchable resource for every hour. 
                    gen = self.solver.NumVar(0, self.solver.infinity(), '_gen_year_'+ str(year) + '_hour' + str(ind))
                    
                    if self.storage_can_charge_from_grid == False:
                        storage_charge.SetCoefficient(gen, 1)

                    #Append hourly gen variable to the list for that resource, located in the disp_gen dictionary.
                    self.disp_gen[resource].append(gen)

                    #Calculate monetized emissions for given resource in selected hour.
                    #*** Need to apply inflation rate to this value for future years? 
                    resource_monetized_emissions_mwh = (self.disp.loc[resource, 'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton) + self.disp.loc[resource, 'nox_lbs_per_mwh']/2000*self.nox_cost_short_ton_la + self.disp.loc[resource, 'so2_lbs_per_mwh']/2000*self.so2_cost_short_ton_la + self.disp.loc[resource, 'pm25_lbs_per_mwh']/2000*self.pm25_cost_short_ton_la
                    
                    
                    #Calculate variable costs extrapolated over portfolio timespan.
                    if 'gas' in resource:
                        variable_om_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                        variable_fuel_cost_inds = self.resource_costs['cost_type']=='fuel_costs_per_mmbtu'
                        heat_rate_inds = self.resource_costs['cost_type']=='heat_rate_mmbtu_per_mwh'
                        variable_om_cost = self.resource_costs.loc[variable_om_inds & resource_inds,str(cost_year)].item() 
                        variable_fuel_cost = self.resource_costs.loc[variable_fuel_cost_inds & resource_inds, str(cost_year)].item()
                        variable_heat_rate = self.resource_costs.loc[heat_rate_inds & resource_inds, str(cost_year)].item()

                        variable_cost = variable_om_cost + (variable_fuel_cost*variable_heat_rate) + resource_monetized_emissions_mwh
                        

                    else:
                        variable_cost_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                        variable_cost = self.resource_costs.loc[variable_cost_inds & resource_inds,str(cost_year)].item()+ resource_monetized_emissions_mwh
                    
                    #Incorporate coefficient for the value of additional energy into variable cost.
                    variable_cost = variable_cost - value_additional_energy_mwh
                    
                    
                    #If not in the last build_year, don't extrapolate variable costs. Just discount back to build_start_year. If in the last build year, extrapolate variable costs to the end of portfolio timespan and then discount total extrapolated costs back to build_start_year.
                    discount_factor = pow(self.growth_rate, -(year))
                    if year < (self.build_years-1):
                        variable_cost_discounted = variable_cost * discount_factor
                    else:
                        variable_cost_discounted = variable_cost_with_value_additional_energy * self.discounting_factor[year] * discount_factor
                    
                    #Incorporate extrapolated variable cost of hourly gen for each disp resource into objective function.
                    objective.SetCoefficient(gen, variable_cost_discounted)

                    #Add hourly gen variables for disp resources to the fulfill_demand constraint.
                    fulfill_demand.SetCoefficient(gen, 1)

                    #Initialize max_gen constraint: hourly gen must be less than or equal to capacity for each dispatchable resource.
                    max_gen = self.solver.Constraint(0, self.solver.infinity())
                    disp_capacity_cumulative = self.capacity_vars[resource][0:year+1]
                    for i, var in enumerate(disp_capacity_cumulative):
                        max_gen.SetCoefficient(var, 1)
                    max_gen.SetCoefficient(gen, -1)

                #Within hourly for loop, loop through nondispatchable resources.   
                for resource in self.nondisp.index:
                    
                    #Nondispatchable resources can only generate their hourly profile scaled by nameplate capacity to help fulfill demand. 
                    profile_max = max(profiles[resource])
                    scaling_coefficient = profiles.loc[ind, resource] / profile_max
                    
                    nondisp_capacity_cumulative = self.capacity_vars[resource][0:year+1]

                    for i, var in enumerate(nondisp_capacity_cumulative):
                        fulfill_demand.SetCoefficient(var, scaling_coefficient)
                        if self.storage_can_charge_from_grid == False:
                            if 'solar' in resource:
                                storage_charge.SetCoefficient(var, scaling_coefficient)
                    
                    #Get the coefficient of capacity variable and change coefficient to incorporate value of additional energy generated in this hour.
                    capacity_variable_current_build_year = self.capacity_vars[resource][-1]
                    
                    existing_coeff = objective.GetCoefficient(var=capacity_variable_current_build_year)

                    #Calculate variable cost including monetized emissions.
                    resource_monetized_emissions_mwh = (self.nondisp.loc[resource, 'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton) + self.nondisp.loc[resource, 'nox_lbs_per_mwh']/2000*self.nox_cost_short_ton_la + self.nondisp.loc[resource, 'so2_lbs_per_mwh']/2000*self.so2_cost_short_ton_la #+ self.nondisp.loc[resource, 'pm10_lbs_per_mwh']*self.pm10_cost_per_ton + self.nondisp.loc[resource, 'pm25_lbs_per_mwh']*self.pm25_cost_per_ton
                    
                    if resource_monetized_emissions_mwh > 0:
                        print(resource, 'resource_monetized_emissions_mwh',resource_monetized_emissions_mwh)
                    
                    variable_cost_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                    resource_inds = self.resource_costs['resource']==resource
                    variable_cost = self.resource_costs.loc[variable_cost_inds & resource_inds, str(cost_year)].item()

                    variable_cost_monetized_emissions = variable_cost + resource_monetized_emissions_mwh
                    
                    #Incorporates value of additional energy generated.
                    variable_cost_monetized_emissions_additional_energy = (variable_cost_monetized_emissions - value_additional_energy_mwh) * scaling_coefficient
                                         
                    #If not in the last build_year, don't extrapolate variable costs. Just discount back to build_start_year. If in the last build year, extrapolate variable costs to the end of portfolio timespan and then discount total extrapolated costs back to build_start_year.
                    discount_factor = pow(self.growth_rate, -(year))
                    if year < (self.build_years-1):
                        coefficient_adjustment = variable_cost_monetized_emissions_additional_energy * discount_factor
                    else:
                        coefficient_adjustment = variable_cost_monetized_emissions_additional_energy * self.discounting_factor[year] * discount_factor

                    new_coefficient = existing_coeff + coefficient_adjustment
                    
                    #Adjust coefficient on capacity variable to include the value of additional energy.
                    objective.SetCoefficient(capacity_variable_current_build_year, new_coefficient)


        return objective


    def discount_factor_from_cost(self, cost, discount_rate, build_years):
        self.growth_rate = 1.0 + discount_rate
        
        discount_factor = []       
        for year in range(build_years):

            value_decay_1 = pow(self.growth_rate, -(self.timespan)) #include if only want to evaluate resources across porftolio timespan: -year))
            value_decay_2 = pow(self.growth_rate, -1)
            try:
                extrapolate = cost * (1.0 - value_decay_1) / (1.0-value_decay_2)
            except ZeroDivisionError:
                extrapolate = cost
            discount_factor.append(extrapolate)
        
        return discount_factor
    
    
    def _setup_resources(self):
        resources = pd.read_csv('data/resources.csv')
        if self.selected_resource != 'all':
            resources = resources[resources['resource']==self.selected_resource]
            
        resources = resources.set_index('resource')
        
        return resources
    
    def _setup_resource_costs(self):
        if self.ee_cost_type == 'utility_side':
            resource_costs = pd.read_csv('data/resource_projected_costs_ee_utilitycosts.csv')
        elif self.ee_cost_type == 'total_cost':
            resource_costs = pd.read_csv('data/resource_projected_costs_ee_totalcosts.csv')
            
        resource_costs = resource_costs[resource_costs['cost_decline_assumption']==self.cost_projections]
        
        
        return resource_costs

        
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
                    capacity = self.solver.NumVar(0, 5, str(resource)+ '_' + str(year))
                    capacity_by_build_year.append(capacity)
                capacity_by_resource[resource] = capacity_by_build_year
            else:
                #If resource is legacy resource, capacity "built" in year 0 of build years must be less than or equal to existing capacity. Built capacity in subsequent build years must be 0.
                existing_mw = self.resources.loc[str(resource)]['existing_mw']
                for year in range(build_years):
                    if year == 0:
                        capacity = self.solver.NumVar(0, existing_mw, str(resource)+ '_' + str(year))
                    else:
                        capacity = self.solver.NumVar(0, 0, str(resource)+ '_' + str(year))
                    capacity_by_build_year.append(capacity)
                capacity_by_resource[resource] = capacity_by_build_year
                
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
    
#     def _setup_outofbasin_emissions(self):
#         outofbasin_emissions = pd.read_csv('data/outofbasin_emissions_template.csv')
#         #outofbasin_emissions.insert(0, 'datetime', harborgen.index)
#         #outofbasin_emissions = outofbasin_emissions.set_index('datetime')
        
#         return outofbasin_emissions
    
    def _setup_wholegrid_emissions(self):
        wholegrid_emissions = pd.read_csv('data/grid_emissions/ladwp_hourly_grid_noxSO2co2.csv')
        wholegrid_emissions = wholegrid_emissions.fillna(0)
        
        return wholegrid_emissions
    
    def _setup_health_costs_emissions_la(self):
        health_costs_emissions_la = pd.read_csv('data/pollutant_health_impacts/COBRA_LADWPplants_healthCosts.csv')
        
        return health_costs_emissions_la
    

    def solve(self):
        self.objective.SetMinimization()
        status = self.solver.Solve()
        if status == self.solver.OPTIMAL:
            print("Solver found optimal solution.")
        elif status == self.solver.FEASIBLE:
            # No optimal solution was found.
            print('A potentially suboptimal solution was found.')
        elif status == self.solver.INFEASIBLE:
            print('The solver could not find a feasible solution.')
        elif status == self.solver.UNBOUNDED:
            print('The linear program is unbounded.')
        elif status == self.solver.ABNORMAL:
            print('The linear program is abnormal.')
        else:
            print("Solver exited with error code {}".format(status))
        return status
            
                   
    def get_lcoe_per_mwh(self):
        
        #Change code to write a csv of LCOE for each resource in each build year.
        
        lcoe_per_mwh_by_resource_df = pd.DataFrame()
        
        lcoe_per_mwh_w_cobenefits_by_resource_df = pd.DataFrame()
        
        
        lcoe_per_mwh_by_resource = {}
        profiles = pd.read_csv('data/gen_profiles.csv')
        
        
        for i,resource in enumerate(self.capacity_vars.keys()):
            
            lcoe_per_mwh_by_resource_df.loc[i,'resource']=resource
            lcoe_per_mwh_w_cobenefits_by_resource_df.loc[i,'resource']=resource
            
            lcoe_per_mwh_by_resource[resource]={}
            
            for build_year in range(self.build_years):
        
                resource_inds = self.resource_costs['resource']==resource
                cost_year = self.build_start_year + build_year

                capex = self.lcoe_dict[build_year][resource]['capex']
                fixed_extrapolated = self.lcoe_dict[build_year][resource]['fixed_extrapolated']

                #Get range of hours where demand is met in order to index correctly into the list of generation variables.
                demand_year = self.build_years-1
                demand_start_hour = (self.build_years-1)*8760
                demand_end_hour = (self.build_years-1)*8760 + 8760

                #For dispatchable resources, calculate extrapolated variable cost and annual generation per mw of capacity.
                if resource in self.disp.index:
                    
                    resource_monetized_emissions_mwh = (self.disp.loc[resource, 'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton) + self.disp.loc[resource, 'nox_lbs_per_mwh']/2000*self.nox_cost_short_ton_la + self.disp.loc[resource, 'so2_lbs_per_mwh']/2000*self.so2_cost_short_ton_la + self.disp.loc[resource, 'pm25_lbs_per_mwh']/2000*self.pm25_cost_short_ton_la

                    #Calculated extrapolated variable cost and add to lcoe dictionary. Excludes health impacts of monetized emissions.
                    if 'gas' in resource:
                        variable_om_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                        variable_fuel_cost_inds = self.resource_costs['cost_type']=='fuel_costs_per_mmbtu'
                        heat_rate_inds = self.resource_costs['cost_type']=='heat_rate_mmbtu_per_mwh'

                        variable_om_cost = self.resource_costs.loc[variable_om_inds & resource_inds,str(cost_year)].item() 
                        variable_fuel_cost = self.resource_costs.loc[variable_fuel_cost_inds & resource_inds, str(cost_year)].item()
                        variable_heat_rate = self.resource_costs.loc[heat_rate_inds & resource_inds, str(cost_year)].item()

                        variable_cost = variable_om_cost + (variable_fuel_cost*variable_heat_rate)
                        
                        variable_cost_w_cobenefits = variable_om_cost + (variable_fuel_cost*variable_heat_rate) + resource_monetized_emissions_mwh

                    else:
                        variable_cost_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                        variable_cost = self.resource_costs.loc[variable_cost_inds & resource_inds,str(cost_year)].item()
                        
                        variable_cost_w_cobenefits = variable_om_cost + (variable_fuel_cost*variable_heat_rate) + resource_monetized_emissions_mwh
                    
                    variable_cost_extrapolated = variable_cost * self.discounting_factor[build_year]
                    
                    variable_cost_w_cobenefits_extrapolated = variable_cost_w_cobenefits * self.discounting_factor[build_year]

                    self.lcoe_dict[build_year][resource]['variable_extrapolated'] = variable_cost_extrapolated


                    summed_gen = 0
                    #Currently sums generation in the year in which demand must be met with supply (the last build year).
                    for i_gen in self.disp_gen[str(resource)][demand_start_hour:demand_end_hour]:
                        summed_gen += i_gen.solution_value()

                    capacity_cumulative = self.capacity_vars[resource][0:self.build_years+1]
                    capacity_total = 0
                    for i, var in enumerate(capacity_cumulative):
                        capacity_total += var.solution_value()

                    if capacity_total >0:
                        mwh_per_mw = summed_gen / capacity_total
                        self.lcoe_dict[build_year][resource]['annual_generation_per_mw'] = mwh_per_mw
                    else:
                        print(resource,': Resource is not selected in optimization solution. Cannot calculate LCOE because annual generation is N/A.')

                elif resource in self.nondisp.index:
                    
                    resource_monetized_emissions_mwh = (self.nondisp.loc[resource, 'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton) + self.nondisp.loc[resource, 'nox_lbs_per_mwh']/2000*self.nox_cost_short_ton_la + self.nondisp.loc[resource, 'so2_lbs_per_mwh']/2000*self.so2_cost_short_ton_la + self.nondisp.loc[resource, 'pm25_lbs_per_mwh']/2000*self.pm25_cost_short_ton_la

                    profile_max = max(profiles[resource])
                    summed_gen = sum(profiles[resource] / profile_max)
                    
                    #Add variable costs extrapolated over portfolio timespan to lcoe dictionary.
                    self.lcoe_dict[build_year][resource]['variable_extrapolated']=self.resource_costs.loc[variable_cost_inds & resource_inds, str(cost_year)].item() * summed_gen * self.discounting_factor[build_year]

                    #Add annual mwh generated per mw capacity to lcoe dictionary.
                    self.lcoe_dict[build_year][resource]['annual_generation_per_mw']=summed_gen
                    
                    variable_cost_extrapolated = self.lcoe_dict[build_year][resource]['variable_extrapolated']
                    mwh_per_mw = self.lcoe_dict[build_year][resource]['annual_generation_per_mw']


                if 'annual_generation_per_mw' in self.lcoe_dict[build_year][resource].keys():
                    #Calculate lcoe_per_mwh without co-benefits.
                    variable_costs = variable_cost_extrapolated * mwh_per_mw
                    lcoe_per_mw = capex + fixed_extrapolated + variable_costs
                    lcoe_per_mwh = lcoe_per_mw / (mwh_per_mw*self.timespan)
                    
                    lcoe_per_mwh_by_resource[resource][build_year] = lcoe_per_mwh
                    
                    #Calculate lcoe_per_mwh with co-benefits.
                    variable_costs = variable_cost_extrapolated * mwh_per_mw
                    lcoe_per_mw = capex + fixed_extrapolated + variable_costs
                    lcoe_per_mwh = lcoe_per_mw / (mwh_per_mw*self.timespan)

                    lcoe_per_mwh_by_resource[resource][build_year] = lcoe_per_mwh
                    
                    lcoe_per_mwh_by_resource_df.loc[i, build_year] = lcoe_per_mwh
                    lcoe_per_mwh_w_cobenefits_by_resource_df.loc[i, build_year] = lcoe_per_mwh
                
            lcoe_per_mwh_by_resource_df['lceo_type'] = 'no_cobenefits'
            lcoe_per_mwh_w_cobenefits_by_resource_df['lceo_type'] = 'w_cobenefits'
            
            lcoe_per_mwh_by_resource_df = pd.concat([lcoe_per_mwh_by_resource_df,lcoe_per_mwh_w_cobenefits_by_resource_df], ignore_index=True)
                
        return lcoe_per_mwh_by_resource_df
        
                
                    
    def get_capacities_mw(self, build_year):
        
        capacities = {}
        for resource in self.capacity_vars:
            capacity = self.capacity_vars[resource][build_year].solution_value()
            capacities[resource] = capacity
        
        return capacities
    
    
    def get_capacity_fractions(self, build_year):
        
        capacity_fractions = {}
        total_capacity = 0
        for resource in self.capacity_vars:
            total_capacity = total_capacity + self.capacity_vars[resource][build_year].solution_value()
        
        if total_capacity > 0:
            for resource in self.capacity_vars:
                fraction_capacity = self.capacity_vars[resource][build_year].solution_value() / total_capacity
                capacity_fractions[resource] = fraction_capacity
        
        if total_capacity == 0:
            print('No resource capacity built in year '+ str(build_year))
    
        return capacity_fractions
            
    
    def get_generation_mwh(self, build_year):

        generation_by_resource = {}
        profiles = pd.read_csv('data/gen_profiles.csv')
        
        #Get range of hours in selected build_year to index correctly into the list of generation variables.
        start_hour = build_year*8760
        end_hour = build_year*8760 + 8760
        
        for resource in self.disp.index:
            summed_gen = 0
            #Sums generation in the selected build_year.
            for i_gen in self.disp_gen[str(resource)][start_hour:end_hour]:
                summed_gen += i_gen.solution_value()
            generation_by_resource[resource] = summed_gen

        for resource in self.nondisp.index:
            profile_max = max(profiles[resource])
            summed_gen = sum(profiles[resource]) / profile_max
            capacity = self.capacity_vars[resource][build_year].solution_value()
            gen = summed_gen * capacity
            generation_by_resource[resource] = gen
            
        #If storage can charge from sources outside portfolio, then net supply from storage should be counted towards total generation.
        for resource in self.storage_capacity_vars:
            summed_storage_gen = 0
            for i,hour in enumerate(self.storage_charge_vars[resource][start_hour:end_hour]):
                charge_var = self.storage_charge_vars[resource][i].solution_value()
                discharge_var = self.storage_discharge_vars[resource][i].solution_value()
                net = discharge_var - charge_var
                summed_storage_gen += net    
            generation_by_resource[resource] = summed_storage_gen
        
        return generation_by_resource   
        
                
            
# Change the code below to create a function that gets the generation fraction for each resource in a given year.    
#         profiles = pd.read_csv('data/gen_profiles.csv')
#         #Sum total annual generation across all resources.
        
#         total_gen = 0
#         gen_fractions = {}
#         for resource in self.disp.index:
#             summed_gen = 0
#             for i_gen in self.disp_gen[str(resource)]:
#                 summed_gen += i_gen.solution_value()
#             fraction_generation = summed_gen / total_gen
#             gen_fractions[resource] = fraction_generation

#         for resource in self.nondisp.index:
#             profile_max = max(profiles[resource])
#             summed_gen = sum(profiles[resource]) / profile_max
#             capacity = self.capacity_vars[resource].solution_value()
#             gen = summed_gen * capacity
#             fraction_generation = gen / total_gen
#             gen_fractions[resource] = fraction_generation
            
#             total_gen = total_gen + gen
        
#         return #gen_fractions



# #Could add other keys to storage results (ex. hourly charge, hourly state of charge, hourly discharge).
#     def storage_results():
        
#         storage_results = {}
#         for resource in self.storage_capacity_vars:
#             resource_results_dict = {}
            
#             storage_capacity = self.storage_capacity_vars[resource].solution_value()
#             resource_results_dict['capacity']= storage_capacity

#             #Get hourly net charge values and add to resource_results_dict.
#             storage_hourly_net = []
#             storage_hourly_charge = []
#             storage_hourly_discharge = []
#             efficiency = self.storage.loc[resource, 'efficiency']
#             for i,hour in enumerate(self.storage_charge_vars[resource]):
                
#                 charge_var = self.storage_charge_vars[resource][i].solution_value()
#                 discharge_var = self.storage_discharge_vars[resource][i].solution_value() * efficiency
                
#                 net = discharge_var - charge_var
#                 storage_hourly_net.append(net)
                
#                 storage_hourly_charge.append(charge_var)
#                 storage_hourly_discharge.append(discharge_var)
            
#             resource_results_dict['hourly_net_source']= storage_hourly_net    
#             resource_results_dict['hourly_charge']= storage_hourly_charge
#             resource_results_dict['hourly_discharge']= storage_hourly_discharge
            
            
#             storage_results[resource]=resource_results_dict
        
#         return storage_results