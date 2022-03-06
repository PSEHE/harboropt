import numpy as np # numerical library
import matplotlib.pyplot as plt # plotting library
import datetime as dt
import pandas as pd
import os

from ortools.linear_solver import pywraplp

import utils

    
class LinearProgram(object):
    
    def __init__(self, solver_type = 'GLOP', demand_profile = pd.DataFrame(), max_excess_energy = pd.DataFrame(), caiso_lmp = pd.DataFrame(), marginal_healthdamages = pd.DataFrame(), marginal_co2 = pd.DataFrame(), profiles = pd.DataFrame(), ee_resource_potential = pd.DataFrame(), resources = pd.DataFrame(), storage = pd.DataFrame(), resource_potential = pd.DataFrame(), health_cost_emissions_la = pd.DataFrame(), res_hourly_tou_retail_rate = pd.DataFrame(), comm_hourly_tou_retail_rate = pd.DataFrame(), EE_comm_annual_demand_charge_savings = pd.DataFrame(), EE_mf_annual_demand_charge_savings = pd.DataFrame(), selected_resource = 'all', initial_state_of_charge = 0, storage_lifespan = 15, portfolio_timespan = 30, storage_can_charge_from_grid = True, discount_rate = 0.03, cost=1, health_cost_range = 'HIGH', cost_projections = 'moderate', build_start_year =2020, harbor_retirement_year = 2029, ee_cost_type = 'utility', bill_savings = False, demand_charge_savings_annual_comm_solar_plus_storage_per_mw_solar = 60000, demand_charge_savings_annual_comm_solarOnly_per_mw_solar = 30000, demand_charge_savings_annual_comm_storageOnly_per_mw_storage = 148000, ratio_solar_to_storage_comm = 5, demand_charge_savings_annual_mf_solar_plus_storage_per_mw_solar = 43500, demand_charge_savings_annual_mf_solarOnly_per_mw_solar = 8700, demand_charge_savings_annual_mf_storageOnly_per_mw_storage = 148000, ratio_solar_to_storage_mf = 4, ratio_solar_to_storage_sf = 1, a1_comm_weight_by_roofsqft = 0.14, a2_comm_weight_by_roofsqft = 0.86, a1_comm_weight_by_floorsqft = 0.01, a2_comm_weight_by_floorsqft = 0.99, transmission_capex_cost_per_mw = 72138, transmission_annual_cost_per_mw = 8223, storage_resilience_incentive_per_kwh = 1000, resilient_storage_grid_fraction = 0.7, social_cost_carbon_short_ton = 46, diesel_genset_carbon_per_mw = 0, diesel_genset_pm25_per_mw = 0, diesel_genset_nox_per_mw = 0, diesel_genset_so2_per_mw = 0, diesel_genset_pm10_per_mw = 0, diesel_genset_fixed_cost_per_mw_year = 0, diesel_genset_mmbtu_per_mwh =0, diesel_genset_cost_per_mmbtu = 0, diesel_genset_run_hours_per_year = 0):
        
        #Test out each constraint.
        self.fulfill_demand_constraint = True
        self.max_charge_constraint = True
        self.residential_storage_tied_to_solar_SF_constraint=True
        self.residential_storage_tied_to_solar_MF_constraint=True
        self.commercial_storage_tied_to_solar_constraint=True
        self.dr_shed_yearly_limit_constraint = True
        self.dr_shed_daily_limit_constraint = True
        self.storage_charge_constraint = True
        self.discharge_zero_constraint = True
        self.gen_zero_constraint = True
        self.utility_storage_limit_constraint = True
        
        self.demand_profile = demand_profile
        self.max_excess_energy = max_excess_energy
        self.caiso_lmp = caiso_lmp
        self.profiles = profiles
        self.ee_resource_potential = ee_resource_potential
        self.resource_potential = resource_potential
        self.resources = resources
        self.res_hourly_tou_retail_rate = res_hourly_tou_retail_rate
        self.comm_hourly_tou_rate = comm_hourly_tou_retail_rate
        self.EE_comm_annual_demand_charge_savings = EE_comm_annual_demand_charge_savings
        self.EE_mf_annual_demand_charge_savings = EE_mf_annual_demand_charge_savings
        self.selected_resource = selected_resource
        if self.selected_resource != 'all':
            self.resources = self.resources[self.resources['resource']==self.selected_resource]
        
        self.selected_resource = selected_resource
        self.initial_state_of_charge = initial_state_of_charge
        self.storage_can_charge_from_grid = storage_can_charge_from_grid
        self.bill_savings = bill_savings
        
        self.a1_comm_weight_by_roofsqft = a1_comm_weight_by_roofsqft
        self.a2_comm_weight_by_roofsqft = a2_comm_weight_by_roofsqft
        self.a1_comm_weight_by_floorsqft = a1_comm_weight_by_floorsqft
        self.a2_comm_weight_by_floorsqft = a2_comm_weight_by_floorsqft
        
        self.portfolio_timespan = portfolio_timespan
        self.storage_lifespan = storage_lifespan
        
        self.build_start_year = build_start_year
        self.harbor_retirement_year = harbor_retirement_year
        #The first year of build_years is the build_start_year. The last year is the Harbor retirement year. 
        self.build_years = self.harbor_retirement_year+1 - self.build_start_year
        
        self.demand_charge_savings_annual_comm_solar_plus_storage_per_mw_solar = demand_charge_savings_annual_comm_solar_plus_storage_per_mw_solar
        self.demand_charge_savings_annual_comm_solarOnly_per_mw_solar = demand_charge_savings_annual_comm_solarOnly_per_mw_solar
        self.demand_charge_savings_annual_comm_storageOnly_per_mw_storage = demand_charge_savings_annual_comm_storageOnly_per_mw_storage
        
        self.demand_charge_savings_annual_mf_solar_plus_storage_per_mw_solar = demand_charge_savings_annual_mf_solar_plus_storage_per_mw_solar
        self.demand_charge_savings_annual_mf_solarOnly_per_mw_solar = demand_charge_savings_annual_mf_solarOnly_per_mw_solar
        self.demand_charge_savings_annual_mf_storageOnly_per_mw_storage = demand_charge_savings_annual_mf_storageOnly_per_mw_storage
        
        self.ratio_solar_to_storage_comm = ratio_solar_to_storage_comm
        self.ratio_solar_to_storage_mf = ratio_solar_to_storage_mf
        self.ratio_solar_to_storage_sf = ratio_solar_to_storage_sf
        
        self.transmission_capex_cost_per_mw = transmission_capex_cost_per_mw
        self.transmission_annual_cost_per_mw = transmission_annual_cost_per_mw
        self.ee_cost_type = ee_cost_type
        
        self.storage_resilience_incentive_per_kwh = storage_resilience_incentive_per_kwh
        self.resilience_incentive_per_mwh = storage_resilience_incentive_per_kwh * 1000
        self.resilient_storage_grid_fraction = resilient_storage_grid_fraction
        
        self.health_cost_range = health_cost_range
        self.cost_projections = cost_projections
        
        self.social_cost_carbon_short_ton = social_cost_carbon_short_ton
        
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
        
        self.solver_type = solver_type
        if self.solver_type == 'GLOP':
            self.solver = pywraplp.Solver('HarborOptimization', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        elif self.solver_type == 'CLP':
            self.solver = pywraplp.Solver('HarborOptimization', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
        elif self.solver_type == 'GLPK':
            self.solver = pywraplp.Solver('HarborOptimization', pywraplp.Solver.GLPK_LINEAR_PROGRAMMING)
        elif self.solver_type == 'GUROBI':
            self.solver = pywraplp.Solver('HarborOptimization', pywraplp.Solver.GUROBI_LINEAR_PROGRAMMING)
        elif self.solver_type == 'SCIP':
            self.solver = pywraplp.Solver('HarborOptimization', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
        elif self.solver_type == 'CBC':
            self.solver = pywraplp.Solver('HarborOptimization', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        self.resources = resources
        self.resource_costs = self._setup_resource_costs()
        
        #Create constraints tying residential storage (SF and MF) and commercial storage built to residential solar (SF and MF) and commercial solar built for solar+storage systems.
        if self.residential_storage_tied_to_solar_SF_constraint == True:
            self.residential_storage_tied_to_solar_SF = self.solver.Constraint(0, 0)
        if self.residential_storage_tied_to_solar_MF_constraint == True:
            self.residential_storage_tied_to_solar_MF = self.solver.Constraint(0, 0)
        if self.commercial_storage_tied_to_solar_constraint == True:
            self.commercial_storage_tied_to_solar = self.solver.Constraint(0, 0)


        self.capacity_vars = self._initialize_capacity_by_resource(self.build_years)
        
        self.storage = storage
        self.storage_capacity_vars = self._initialize_storage_capacity_vars(self.build_years)
        

        self.disp = self.resources.loc[self.resources['dispatchable'] == 'y']
        self.nondisp = self.resources.loc[self.resources['dispatchable'] == 'n']
        
        self.ladwp_marginal_co2 = marginal_co2
        #self.ladwp_marginal_co2 = self._setup_ladwp_marginal_co2()
        self.ladwp_marginal_healthdamages = marginal_healthdamages
        #self.ladwp_marginal_healthdamages = self._setup_ladwp_marginal_healthdamages()
        self.health_cost_emissions_la = health_cost_emissions_la
        
        #Set up health cost of pollutants emitted in LA. ** Replace with $/lb values for each pollutant from WattTime. See email from Henry.
        discount_rate_inds = self.health_cost_emissions_la['discount_rate'] == self.discount_rate
        la_inds = self.health_cost_emissions_la['county'] == 'LA'
        pm25_inds = self.health_cost_emissions_la['pollutant'] == 'PM2.5' 
        so2_inds = self.health_cost_emissions_la['pollutant'] == 'SO2'
        nox_inds = self.health_cost_emissions_la['pollutant'] == 'NOx'
        
        if self.health_cost_range == 'HIGH':
            self.pm25_cost_short_ton_la = self.health_cost_emissions_la[discount_rate_inds & la_inds & pm25_inds]['US_HIGH_annual ($/ton)'].iloc[0]*-1

            self.so2_cost_short_ton_la = self.health_cost_emissions_la[discount_rate_inds & la_inds & so2_inds]['US_HIGH_annual ($/ton)'].iloc[0]*-1

            self.nox_cost_short_ton_la = self.health_cost_emissions_la[discount_rate_inds & la_inds & nox_inds]['US_HIGH_annual ($/ton)'].iloc[0]*-1

            
        if self.health_cost_range == 'LOW':
            self.pm25_cost_short_ton_la = self.health_cost_emissions_la[discount_rate_inds & la_inds & pm25_inds]['US_LOW_annual ($/ton)'].iloc[0]*-1

            self.so2_cost_short_ton_la = self.health_cost_emissions_la[discount_rate_inds & la_inds & so2_inds]['US_LOW_annual ($/ton)'].iloc[0]*-1

            self.nox_cost_short_ton_la = self.health_cost_emissions_la[discount_rate_inds & la_inds & nox_inds]['US_LOW_annual ($/ton)'].iloc[0]*-1
  
        
        self.extrapolate_then_discount = self.discount_factor_from_cost(self.cost, self.discount_rate, self.build_years)
        
        
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
        
        self.objective = self._add_constraints_and_costs() 
        
              
    def _add_constraints_and_costs(self):
        
        #Initialize objective function.
        objective = self.solver.Objective()
        
        
        for year in range(self.build_years):
            
            print('year',year)
            print('build_years',self.build_years)
            
            cost_year = self.build_start_year + year
            self.lcoe_dict[year]={}
            
            #Initialize constraint limiting DR shed resource to 48 hours/year.
            if self.dr_shed_yearly_limit_constraint == True:
                dr_shed_yearly_limit = self.solver.Constraint(0, self.solver.infinity())
            
            #Outside of hourly loop, add capex costs and extrapolated fixed costs to obj function for each nondisp resource. 
            for resource in self.nondisp.index: 
                if resource in self.capacity_vars.keys():
                    self.lcoe_dict[year][resource]={}

                    capacity = self.capacity_vars[resource][year]

                    resource_inds = self.resource_costs['resource']==resource

                    capex_cost_inds = self.resource_costs['cost_type']=='capex_per_kw'
                    fixed_cost_inds = self.resource_costs['cost_type']=='fixed_per_kw_year'

                    #Extrapolate fixed costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                    discount_factor = pow(self.growth_rate, -(year))
                    fixed_mw= self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000 
                    
                    #If bill_savings is set to True, incorporate bill savings for solar, storage, and EE. 
                    if self.bill_savings == True:
                        
                        #Incorporate demand charge savings into fixed costs for commercial EE commercial cooling and refrigeration measures, and for EE residential multifamily lighting, cooling, and refrigerator measures (comm ventilation does not achieve demand charge savings according to Patrick). If additional EE measures are included in the model run, need to incorporate demand charge savings ratios for them.
                        
                        
                        #Small commercial buildings only incur a monthly demand charge based on yearly peak. For these buildings, get yearly demand charge savings (summing savings from yearly peak reductions) for cooling and refrigeration EE measures. Subtract this value from fixed costs ($/MW) before extrapolating costs. **Technically savings do not accrue in the first year that EE measure is built for demand charge savings based on yearly peak (only monthly peak) —— need to change code to account for this.                
                        small_commercial_buildings = ['FCZ7.Commercial.Restaurant.Cooling',
                              'FCZ7.Commercial.Restaurant.Refrigeration',
                              'FCZ7.Commercial.Miscellaneous.Cooling']
                        
                        if resource == 'FCZ7.Commercial.Restaurant.Cooling' or 'FCZ7.Commercial.Miscellaneous.Cooling':
                            demand_charge_savings_mw_year = self.EE_comm_annual_demand_charge_savings[self.EE_comm_annual_demand_charge_savings['resource_peakperiod'].str.contains('hvac_yearly_peak')].sum()['demand_charge_savings_per_mw']
                            fixed_mw = fixed_mw - demand_charge_savings_mw_year
                            
                        if resource == 'FCZ7.Commercial.Restaurant.Refrigeration':
                            demand_charge_savings_mw_year = self.EE_comm_annual_demand_charge_savings[self.EE_comm_annual_demand_charge_savings['resource_peakperiod'].str.contains('refrigeration_yearly_peak')].sum()['demand_charge_savings_per_mw']
                            fixed_mw = fixed_mw - demand_charge_savings_mw_year
                                        
                        #Large commercial buildings incur monthly demand charges based on yearly peak, monthly peak in high peak pricing period, and monthly peak in low peak pricing period. Get yearly demand charge savings (summing savings from yearly and monthly peak reductions) for large comm cooling and refrigeration measures. Subtract this value from fixed costs ($/MW) before extrapolating costs.  
                        comm_cooling_strings = ['FCZ7','Commercial','Cooling']
                        comm_refrig_strings = ['FCZ7','Commercial','Refrigeration']
                        
                        if all(x in resource for x in comm_cooling_strings):
                            if not resource in small_commercial_buildings:
                                demand_charge_savings_mw_year = self.EE_comm_annual_demand_charge_savings[self.EE_comm_annual_demand_charge_savings['resource_peakperiod'].str.contains('hvac')].sum()['demand_charge_savings_per_mw']
                                fixed_mw = fixed_mw - demand_charge_savings_mw_year
                            
                        if all(x in resource for x in comm_refrig_strings):
                            if not resource in small_commercial_buildings:
                                demand_charge_savings_mw_year = self.EE_comm_annual_demand_charge_savings[self.EE_comm_annual_demand_charge_savings['resource_peakperiod'].str.contains('refrigeration')].sum()['demand_charge_savings_per_mw']
                                fixed_mw = fixed_mw - demand_charge_savings_mw_year
                        
                        #Residential MF buildings incur monthly demand charges based on yearly peak and monthly peak. Get yearly demand charge savings (summing savings from yearly and monthly peak reductions) for res MF cooling, refrigeration, and lighting EE measures. Subtract this value from fixed costs ($/MW) before extrapolating costs. 
                
                        mf_lighting_strings = ['FCZ7','MULTIFAMILY','ResLightingEff']
                        mf_cooling_strings = ['FCZ7','MULTIFAMILY','Cooling']
                        mf_refrig_strings = ['FCZ7','MULTIFAMILY','Refrigerator']

                        if all(x in resource for x in mf_lighting_strings):
                            demand_charge_savings_mw_year = self.EE_mf_annual_demand_charge_savings[self.EE_mf_annual_demand_charge_savings['resource_peakperiod'].str.contains('lighting')].sum()['demand_charge_savings_per_mw']
                            fixed_mw = fixed_mw - demand_charge_savings_mw_year
                        
                        if all(x in resource for x in mf_cooling_strings):
                            demand_charge_savings_mw_year = self.EE_mf_annual_demand_charge_savings[self.EE_mf_annual_demand_charge_savings['resource_peakperiod'].str.contains('hvac')].sum()['demand_charge_savings_per_mw']
                            fixed_mw = fixed_mw - demand_charge_savings_mw_year
                            
                        if all(x in resource for x in mf_refrig_strings):
                            demand_charge_savings_mw_year = self.EE_mf_annual_demand_charge_savings[self.EE_mf_annual_demand_charge_savings['resource_peakperiod'].str.contains('refrigeration')].sum()['demand_charge_savings_per_mw']
                            fixed_mw = fixed_mw - demand_charge_savings_mw_year
                            
                            
                        #Incorporate demand charge savings for comm and res multi-family solar+storage systems (savings per mw solar built of solar+storage system) and for comm solar without storage.
                        if resource == 'solar_rooftop_ci_solarStorage':
                            bill_savings_mw_year = self.demand_charge_savings_annual_comm_solar_plus_storage_per_mw_solar
                            fixed_mw = fixed_mw - bill_savings_mw_year
                        if resource == 'solar_rooftop_ci_solarOnly':
                            bill_savings_mw_year = self.demand_charge_savings_annual_comm_solarOnly_per_mw_solar
                            fixed_mw = fixed_mw - bill_savings_mw_year
                            
                        if resource == 'solar_rooftop_residentialMF_solarStorage':
                            bill_savings_mw_year = self.demand_charge_savings_annual_mf_solar_plus_storage_per_mw_solar
                            fixed_mw = fixed_mw - bill_savings_mw_year
                        if resource == 'solar_rooftop_residentialMF_solarOnly':
                            bill_savings_mw_year = self.demand_charge_savings_annual_mf_solarOnly_per_mw_solar
                            fixed_mw = fixed_mw - bill_savings_mw_year
                            
                                                   
                    #Extrapolate fixed costs out to end of portfolio timespan and discount back to build_start_year.
                    fixed_mw_extrapolated = fixed_mw * self.extrapolate_then_discount[year] * discount_factor
                    
                    if resource == 'utility_solar_outofbasin':
                        discount_factor = pow(self.growth_rate, -(year))
                        transmission_annual_extrapolated = self.transmission_annual_cost_per_mw * self.extrapolate_then_discount[year] * discount_factor
                        fixed_mw_extrapolated = fixed_mw_extrapolated + transmission_annual_extrapolated

                    capex_mw = self.resource_costs.loc[capex_cost_inds & resource_inds, str(cost_year)].item()*1000 * discount_factor

                    weighted_avg_eul_inds = self.resource_costs['cost_type']=='weighted_avg_eul'
                    resource_weighted_avg_eul_inds = self.resource_costs.loc[weighted_avg_eul_inds & resource_inds]

                    #If nondisp resource is one with an effective useful life in the resource_projected_costs df, incorporate replacement costs.
                    if not resource_weighted_avg_eul_inds.empty:

                        weighted_avg_eul = round(self.resource_costs.loc[weighted_avg_eul_inds & resource_inds, str(cost_year)].item())
                        number_of_replacements = int((self.portfolio_timespan -1 -year)/ weighted_avg_eul)

                        ## Calculate replacement capex costs in future years, apply discount rate, and add to original capex cost.
                        for i in range(number_of_replacements):

                            replacement_year = int(((i+1) * weighted_avg_eul) + cost_year)

                            if replacement_year > 2050:
                                replacement_capex = self.resource_costs.loc[capex_cost_inds & resource_inds, str(2050)].item()*1000
                            else:
                                replacement_capex = self.resource_costs.loc[capex_cost_inds & resource_inds, str(replacement_year)].item()*1000

                            #Calculate discounting factor to apply to capex in the given replacement year.
                            discount_factor = pow(self.growth_rate, -(replacement_year-self.build_start_year))
                            replacement_capex_discounted =  replacement_capex * discount_factor

                            capex_mw = capex_mw + replacement_capex_discounted


                    #Add capex cost for given build year to lcoe dictionary.
                    self.lcoe_dict[year][resource]['capex']=capex_mw

                    #Add fixed costs extrapolated over portfolio timespan to lcoe dictionary.
                    self.lcoe_dict[year][resource]['fixed_extrapolated']=fixed_mw_extrapolated

                    capex_fixed = capex_mw + fixed_mw_extrapolated

                    if resource == 'utility_solar_outofbasin':
                        #*** Need to apply inflation rate to this transmission cost.
                        discount_factor = pow(self.growth_rate, -(year))
                        transmission_cost = self.transmission_capex_cost_per_mw * discount_factor
                        capex_fixed = capex_fixed + transmission_cost 

                    #Add total cost coefficient to nondisp capacity variable in objective function.
                    objective.SetCoefficient(capacity, capex_fixed)


            #Within build year loop but outside of hourly loop, add capex and fixed costs to objective function for every dispatchable resource.         
            for resource in self.disp.index:
                
                self.lcoe_dict[year][resource]={}
                capacity = self.capacity_vars[resource][year]
                
                #In each year, set constraint limiting CII DR total shed (dispatch) to less than or equal to 48 hours * cumulative DR capacity.
                if resource == 'ci_shed':
                    dr_capacity_cumulative = self.capacity_vars[resource][0:year+1]
                    if self.dr_shed_yearly_limit_constraint == True:
                        for i, var in enumerate(dr_capacity_cumulative):
                            dr_shed_yearly_limit.SetCoefficient(var, 48)

                
                resource_inds = self.resource_costs['resource']==resource
                capex_cost_inds = self.resource_costs['cost_type']=='capex_per_kw'
                
                capex_cost_per_mw = self.resource_costs.loc[capex_cost_inds & resource_inds, str(cost_year)].item()*1000 * discount_factor
                
                #Add capex cost for the given build year to lcoe dictionary.
                self.lcoe_dict[year][resource]['capex']=capex_cost_per_mw
                
                fixed_cost_inds = self.resource_costs['cost_type']=='fixed_per_kw_year'
                
                #Extrapolate fixed costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                discount_factor = pow(self.growth_rate, -(year))
                fixed_mw_extrapolated = self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000 * self.extrapolate_then_discount[year] * discount_factor
                
                #Add fixed costs accumulated over chosen timespan to lcoe dictionary.
                self.lcoe_dict[year][resource]['fixed_extrapolated']=fixed_mw_extrapolated
                
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
                    
                    fixed_mw = self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000
                    #Extrapolate fixed costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                    fixed_mw_extrapolated = fixed_mw * self.extrapolate_then_discount[year] * discount_factor
                    
                else:
                    capex_per_mw = self.resource_costs.loc[capex_cost_inds & resource_inds, str(cost_year)].item()*1000 * discount_factor
                    fixed_mw = self.resource_costs.loc[fixed_cost_inds & resource_inds, str(cost_year)].item()*1000
                    
                    #If bill_savings is set to True, incorporate demand charge savings into fixed costs for commercial storage.
                    if self.bill_savings == True:
                        
                        #Get yearly demand charge and energy savings for commercial storage per mw storage. Subtract this value from fixed costs ($/MW) before extrapolating costs.  
                        if resource == 'storage_ci_StorageOnly':
                            bill_savings_mw_year = self.demand_charge_savings_annual_comm_storageOnly_per_mw_storage
                            print(resource, bill_savings_mw_year)
                            fixed_mw = fixed_mw - bill_savings_mw_year
                    
                    fixed_mw_extrapolated = fixed_mw * self.extrapolate_then_discount[year] * discount_factor
                    

                ## Subtract avoided emissions from cost of resilient storage replacing diesel gensets.
                if resource == 'diesel_genset_replacement_storage_4hr':
                    diesel_genset_monetized_emissions_yearly = (self.diesel_genset_carbon_per_mw * self.social_cost_carbon_short_ton) + self.diesel_genset_pm25_per_mw * self.pm25_cost_short_ton_la + self.diesel_genset_nox_per_mw * self.nox_cost_short_ton_la + self.diesel_genset_so2_per_mw * self.so2_cost_short_ton_la #+ self.diesel_genset_pm10_per_mw * self.pm10_cost_per_ton 
                    
                    diesel_genset_fixed_cost_extrapolated = self.diesel_genset_fixed_cost_per_mw_year * self.extrapolate_then_discount[year]* discount_factor
        
                    diesel_genset_fuel_cost_yearly = self.diesel_genset_mmbtu_per_mwh * self.diesel_genset_cost_per_mmbtu * self.diesel_genset_run_hours_per_year
                    
                    monetized_emissions_saved_extrapolated = diesel_genset_monetized_emissions_yearly * self.extrapolate_then_discount[year] * discount_factor
                    fuel_costs_saved_extrapolated = diesel_genset_fuel_cost_yearly * self.extrapolate_then_discount[year]*discount_factor
                    
                    capex_per_mw =  capex_per_mw - monetized_emissions_saved_extrapolated - fuel_costs_saved_extrapolated
                    fixed_mw_extrapolated = fixed_mw_extrapolated - diesel_genset_fixed_cost_extrapolated
                    
                
                #For all storage resources, incorporate replacement capex costs after storage lifespan.
                number_of_storage_replacements = int((self.portfolio_timespan -1 -year)/ self.storage_lifespan)

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
            # 1) hourly generation variables for each dispatchable resource 
            # 2) hourly constraints
            # 3) adding variable cost coefficients to each hourly generation variable.
            for ind in self.demand_profile.index:

                #Initialize fulfill demand constraint: summed generation from all resources must be equal or greater to demand in all hours.
                if self.fulfill_demand_constraint == True:
                    fulfill_demand = self.solver.Constraint(self.demand_profile.loc[ind,'load_mw'], self.solver.infinity())

                #Every 24 hours, initialize new daily DR shed limit (4 hours).
                if ind%24 == 0:
                    if self.dr_shed_daily_limit_constraint == True:
                        dr_shed_daily_limit = self.solver.Constraint(0, self.solver.infinity())
                        for i, var in enumerate(dr_capacity_cumulative):
                            dr_shed_daily_limit.SetCoefficient(var, 4)
                
                #If storage can only charge from portfolio of resources, initialize constraint that storage can only charge from dispatchable generation or solar (not energy efficiency).
                if self.storage_can_charge_from_grid == False:
                    if self.storage_charge_constraint == True:
                        storage_charge = self.solver.Constraint(0, self.solver.infinity())
                    
                #Get grid hourly marginal health damages and CO2 costs.
                grid_monetized_emissions_per_mwh = self.ladwp_marginal_healthdamages.loc[ind, 'healthdamage_moer'] + self.ladwp_marginal_co2.loc[ind,'moer (lbs CO2/MWh)']/2000*self.social_cost_carbon_short_ton

                #Calculate value of additional energy generated in this hour of the year. This includes avoided LMP (locational marginal price) and avoided marginal grid emissions. 
                lmp = self.caiso_lmp.loc[ind,'LMP_$/MWh']
                value_additional_energy_mwh = lmp + grid_monetized_emissions_per_mwh
                
                #Get max excess energy value and demand value in this hour.
                max_excess_energy_mwh = self.max_excess_energy.loc[ind,'max_excess_energy (mwh)']
                demand = self.demand_profile.loc[ind,'load_mw']
                
                #Create constraint for valuation of excess energy.
                value_excess_energy_constraint = self.solver.Constraint(0, self.solver.infinity())
                
                #Create floating variable for valuation of excess energy (floating variable minus demand is excess energy). 
                #Floating variable has to be equal to or more than demand, and cannot exceed demand+max excess energy.
                floating_variable = self.solver.NumVar(0, max_excess_energy_mwh, 'floating_variable'+str(year)+str(ind))
                
                #Coefficient on floating variable is negative value of excess energy.
                discount_factor = pow(self.growth_rate, -(year))
                if year < (self.build_years-1):
                    floating_variable_coeff = - (value_additional_energy_mwh * discount_factor)
                else:
                    floating_variable_coeff = - (value_additional_energy_mwh * self.extrapolate_then_discount[year] * discount_factor)
                
                #Add floating variable to objective function.
                objective.SetCoefficient(floating_variable, floating_variable_coeff)
                
                value_excess_energy_constraint.SetCoefficient(floating_variable, -1)
            
                #Create dummy variable for valuation of excess energy (to add demand*value scalar to objective function) 
                dummy_variable = self.solver.NumVar(1, 1, 'dummy_variable'+str(year)+str(ind))
                
                if year < (self.build_years-1):
                    dummy_variable_coeff = value_additional_energy_mwh * demand * discount_factor
                else:
                    dummy_variable_coeff = value_additional_energy_mwh * demand * self.extrapolate_then_discount[year] * discount_factor

                objective.SetCoefficient(dummy_variable, dummy_variable_coeff)

                
                #Within hourly for loop, loop through nondispatchable resources.   
                for resource in self.nondisp.index:
                    if resource in self.capacity_vars.keys():
                        
                        #Nondispatchable resources can only generate their hourly profile scaled by nameplate capacity to help fulfill demand. 
                        profile_max = max(self.profiles[resource])
                        scaling_coefficient = self.profiles.loc[ind, resource] / profile_max

                        nondisp_capacity_cumulative = self.capacity_vars[resource][0:year+1]

                        for i, var in enumerate(nondisp_capacity_cumulative):
                            if self.fulfill_demand_constraint == True:
                                fulfill_demand.SetCoefficient(var, scaling_coefficient)

                                value_excess_energy_constraint.SetCoefficient(var, scaling_coefficient)


                            if self.storage_can_charge_from_grid == False:
                                if 'solar' in resource:
                                    if self.storage_charge_constraint == True:
                                        storage_charge.SetCoefficient(var, scaling_coefficient)

                        #Get the coefficient of capacity variable and change coefficient to incorporate variable costs.
                        #If EE costs are set to "total_cost," subtract residential bill savings from variable costs.
                        capacity_variable_current_build_year = self.capacity_vars[resource][-1]

                        existing_coeff = objective.GetCoefficient(var=capacity_variable_current_build_year)

                        #Calculate variable cost including monetized emissions from that resource.
                        resource_monetized_emissions_mwh = (self.nondisp.loc[resource, 'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton) + self.nondisp.loc[resource, 'nox_lbs_per_mwh']/2000*self.nox_cost_short_ton_la + self.nondisp.loc[resource, 'so2_lbs_per_mwh']/2000*self.so2_cost_short_ton_la + self.nondisp.loc[resource, 'pm25_lbs_per_mwh']/2000*self.pm25_cost_short_ton_la

                        #Error check— nondispatchable resources should have 0 emissions in our model.
                        if resource_monetized_emissions_mwh > 0:
                            print('ERROR: ',resource, ' monetized emissions (per mwh) =', resource_monetized_emissions_mwh)

                        variable_cost_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                        resource_inds = self.resource_costs['resource']==resource
                        variable_cost = self.resource_costs.loc[variable_cost_inds & resource_inds, str(cost_year)].item()
                        
        
                        #Incorporate energy cost bill savings if self.bill_savings is set to True.
                        if self.bill_savings == True:
                            #End-user bill savings are retail rate minus wholesale price of electricity (LMP). To avoid double-counting savings across bill savings and value of excess energy.
                            small_commercial_buildings = ['FCZ7.Commercial.Restaurant.Cooling',
                                                          'FCZ7.Commercial.Restaurant.Refrigeration',
                                                          'FCZ7.Commercial.Miscellaneous.Cooling']

                            if 'MULTIFAMILY' in resource:
                                bill_savings = self.comm_hourly_tou_rate.loc[ind, 'rate_mf'].item()
                                variable_cost = variable_cost - (bill_savings - lmp)
                            elif 'SINGLEFAMILY' in resource:
                                bill_savings = self.res_hourly_tou_retail_rate.loc[ind, 'rate_sf']
                                variable_cost = variable_cost - (bill_savings - lmp)
                            elif 'FCZ7.Commercial' in resource:
                                if resource in small_commercial_buildings:
                                    bill_savings = self.comm_hourly_tou_rate.loc[ind, 'rate_small_comm']
                                else:
                                    bill_savings = self.comm_hourly_tou_rate.loc[ind, 'rate_large_comm']
                                variable_cost = variable_cost - (bill_savings - lmp)
                            
                            elif 'solar_rooftop_ci' in resource:
                                #Weight comm buildings into a1 and a2 by roofsqft.
                                bill_savings = self.a1_comm_weight_by_roofsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_small_comm'] + self.a2_comm_weight_by_roofsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_large_comm']
                                variable_cost = variable_cost - (bill_savings - lmp)
                            elif 'solar_rooftop_residentialSF' in resource:
                                bill_savings = self.res_hourly_tou_retail_rate.loc[ind, 'rate_sf']
                                variable_cost = variable_cost - (bill_savings - lmp)
                            elif 'solar_rooftop_residentialMF' in resource:
                                bill_savings = self.comm_hourly_tou_rate.loc[ind, 'rate_mf']
                                variable_cost = variable_cost - (bill_savings - lmp)
                                
                                
                                
                        variable_cost_monetized_emissions = variable_cost + resource_monetized_emissions_mwh

                        #Incorporate value of additional energy generated into coefficient.
                        variable_cost_monetized_emissions_coeff = variable_cost_monetized_emissions * scaling_coefficient

                        #If not in the last build_year, don't extrapolate variable costs. Just discount back to build_start_year. If in the last build year, extrapolate variable costs to the end of portfolio timespan and then discount total extrapolated costs back to build_start_year.
                        discount_factor = pow(self.growth_rate, -(year))
                        if year < (self.build_years-1):
                            coefficient_adjustment = variable_cost_monetized_emissions_coeff * discount_factor
                        else:
                            coefficient_adjustment = variable_cost_monetized_emissions_coeff * self.extrapolate_then_discount[year] * discount_factor

                        #Adjust coefficient on capacity variable to include variable costs.
                        new_coefficient = existing_coeff + coefficient_adjustment
                        objective.SetCoefficient(capacity_variable_current_build_year, new_coefficient)
                    
                #Create hourly charge and discharge variables for each storage resource and store in respective dictionaries. 
                for resource in self.storage.index:

                    storage_duration = self.storage.loc[resource, 'storage_duration (hrs)']
                    efficiency = self.storage.loc[resource, 'efficiency']

                    #Create hourly charge and discharge variables for each storage resource in each build year.
                    charge= self.solver.NumVar(0, self.solver.infinity(), resource + '_charge_year'+ str(year) + '_hour' + str(ind))
                    discharge= self.solver.NumVar(0, self.solver.infinity(), resource + '_discharge_year'+ str(year) + '_hour' + str(ind))
                    
                    value_excess_energy_constraint.SetCoefficient(charge, -1)
                    value_excess_energy_constraint.SetCoefficient(discharge, 1)

                    if self.storage_can_charge_from_grid == False:
                        if self.storage_charge_constraint ==True:
                            storage_charge.SetCoefficient(charge, -1)

                    #Add variable cost of charging to objective function. If storage charges from grid, add monetized grid emissions and cost of electricity (wholesale or retail) to variable cost.
                    cost_type_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                    resource_inds = self.resource_costs['resource']==resource
                    
                    if self.storage_can_charge_from_grid:
                        #If storage can charge from grid, adjust variable cost of charging storage to include health+env costs of marginal grid emissions to charge and wholesale cost of electricity to charge.
                        variable_cost = self.resource_costs.loc[cost_type_inds & resource_inds, str(cost_year)].item() + grid_monetized_emissions_per_mwh + lmp

                        #If self.bill_savings is set to True, incorporate retail cost of charging for BTM storage (minus wholesale cost of charging).
                        if self.bill_savings == True:
                            if 'storage_ci' in resource:
                                btm_cost_to_charge = self.a1_comm_weight_by_roofsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_small_comm'] + self.a2_comm_weight_by_roofsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_large_comm']
                                variable_cost = variable_cost + (btm_cost_to_charge - lmp)
                            elif 'storage_residentialSF' in resource:
                                btm_cost_to_charge = self.res_hourly_tou_retail_rate.loc[ind, 'rate_sf']
                                variable_cost = variable_cost + (btm_cost_to_charge - lmp)
                            elif 'storage_residentialMF' in resource:
                                btm_cost_to_charge = self.comm_hourly_tou_rate.loc[ind, 'rate_mf']
                                variable_cost = variable_cost + (btm_cost_to_charge - lmp)
                        
                    #If storage cannot charge from grid (and can only charge from portfolio resources), cost of charging is already incorporated into variable generation costs of other resources in portfolio.
                    else:
                        variable_cost = self.resource_costs.loc[cost_type_inds & resource_inds, str(cost_year)].item()
                        
                    #If not in the last build_year, don't extrapolate variable costs. Just discount back to build_start_year. If in the last build year, extrapolate variable costs to the end of portfolio timespan and then discount extrapolated costs back to build_start_year.
                    discount_factor = pow(self.growth_rate, -(year))
                    if year < (self.build_years-1):
                        variable_cost = variable_cost * discount_factor
                    else:
                        variable_cost = variable_cost * self.extrapolate_then_discount[year] * discount_factor
                    
                    objective.SetCoefficient(charge, variable_cost)
                                   
                        
                    #If self.bill_savings is set to True, incorporate bill savings from discharging BTM storage (avoided retail rate in these hours, minus wholesale rate). This should be included whether or not storage can charge from grid.
                    if self.bill_savings == True:
                        if 'storage_ci_solarStorage' in resource:
                            #Weight buildings by A1 and A2 rates by roof sqft.
                            btm_bill_savings = (self.a1_comm_weight_by_roofsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_small_comm'] + self.a2_comm_weight_by_roofsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_large_comm'])-lmp
                            
                            #If not in the last build_year, don't extrapolate variable bill savings. Just discount back to build_start_year. If in the last build year, extrapolate variable bill savings to the end of portfolio timespan and then discount extrapolated savings back to build_start_year.
                            discount_factor = pow(self.growth_rate, -(year))
                            if year < (self.build_years-1):
                                btm_bill_savings = btm_bill_savings * discount_factor
                            else:
                                btm_bill_savings = btm_bill_savings * self.extrapolate_then_discount[year] * discount_factor
                            objective.SetCoefficient(discharge, -btm_bill_savings)

                        elif 'storage_ci_storageOnly' in resource:
                            #Weight buildings by A1 and A2 rates by floor sqft.
                            btm_bill_savings = (self.a1_comm_weight_by_floorsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_small_comm'] + self.a2_comm_weight_by_floorsqft * self.comm_hourly_tou_rate.loc[ind, 'rate_large_comm'])-lmp
                            
                            #If not in the last build_year, don't extrapolate variable bill savings. Just discount back to build_start_year. If in the last build year, extrapolate variable bill savings to the end of portfolio timespan and then discount extrapolated savings back to build_start_year.
                            discount_factor = pow(self.growth_rate, -(year))
                            if year < (self.build_years-1):
                                btm_bill_savings = btm_bill_savings * discount_factor
                            else:
                                btm_bill_savings = btm_bill_savings * self.extrapolate_then_discount[year] * discount_factor
                            objective.SetCoefficient(discharge, -btm_bill_savings)
                            
                            
                        elif 'storage_residentialSF' in resource:
                            btm_bill_savings = (self.res_hourly_tou_retail_rate.loc[ind, 'rate_sf'])-lmp
                            
                            #If not in the last build_year, don't extrapolate variable bill savings. Just discount back to build_start_year. If in the last build year, extrapolate variable bill savings to the end of portfolio timespan and then discount extrapolated savings back to build_start_year.
                            discount_factor = pow(self.growth_rate, -(year))
                            if year < (self.build_years-1):
                                btm_bill_savings = btm_bill_savings * discount_factor
                            else:
                                btm_bill_savings = btm_bill_savings * self.extrapolate_then_discount[year] * discount_factor
                            objective.SetCoefficient(discharge, -btm_bill_savings)
                        
                        elif 'storage_residentialMF' in resource:
                            btm_bill_savings = (self.comm_hourly_tou_rate.loc[ind, 'rate_mf'])-lmp
                            
                            #If not in the last build_year, don't extrapolate variable bill savings. Just discount back to build_start_year. If in the last build year, extrapolate variable bill savings to the end of portfolio timespan and then discount extrapolated savings back to build_start_year.
                            discount_factor = pow(self.growth_rate, -(year))
                            if year < (self.build_years-1):
                                btm_bill_savings = btm_bill_savings * discount_factor
                            else:
                                btm_bill_savings = btm_bill_savings * self.extrapolate_then_discount[year] * discount_factor
                            objective.SetCoefficient(discharge, -btm_bill_savings)
                        

                    #Limit hourly charge and discharge variables to storage max power (MW). 
                    #Sum storage capacity from previous and current build years to set max power.
                    if self.max_charge_constraint == True:
                        max_charge= self.solver.Constraint(0, self.solver.infinity())
                    storage_capacity_cumulative = self.storage_capacity_vars[resource][0:year+1]
                    for i, var in enumerate(storage_capacity_cumulative):
                        if self.storage.loc[resource, 'resilient'] == 'y':
                            #For resilient storage, limit max charge to the fraction of capacity set aside for the grid.
                            max_charge.SetCoefficient(var, self.resilient_storage_grid_fraction)

                        else: 
                            max_charge.SetCoefficient(var, 1)
                            
                    
                    max_charge.SetCoefficient(charge, -1)
                    
                    #*** Is discharge supposed to be constrained to zero here?
                    if year == 0 and ind == self.demand_profile.index[0]:  #self.demand_hour_start:
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

                    elif ind > self.demand_profile.index[0]:#self.demand_hour_start:
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
                    if self.fulfill_demand_constraint == True:
                        fulfill_demand.SetCoefficient(discharge, 1)
                               
                    
                    #Include the line below if storage cannot charge from grid (and can only charge from portfolio resources).
                    if self.storage_can_charge_from_grid == False:
                        if self.fulfill_demand_constraint == True:
                            fulfill_demand.SetCoefficient(charge, -1)


                    #Creates hourly state of charge variable, representing the state of charge at the end of each timestep. 
                    state_of_charge= self.solver.NumVar(0, self.solver.infinity(), 'state_of_charge_year'+ str(year) + '_hour' + str(ind))

                    #Temporal coupling of storage state of charge.
                    if ind > self.demand_profile.index[0]:
                        state_of_charge_constraint= self.solver.Constraint(0, 0)
                        state_of_charge_constraint.SetCoefficient(state_of_charge, -1)
                        
                        state_of_charge_constraint.SetCoefficient(discharge, -1/efficiency)
                        
                        state_of_charge_constraint.SetCoefficient(charge, 1)
                        
                        #Get the state of charge from previous timestep to include in the state_of_charge_constraint.
                        previous_state = self.storage_state_of_charge_vars[resource][-1]
                        state_of_charge_constraint.SetCoefficient(previous_state, 1)
                        
                        
                    else: 
                        state_of_charge_constraint= self.solver.Constraint(self.initial_state_of_charge, self.initial_state_of_charge)
                        state_of_charge_constraint.SetCoefficient(state_of_charge, 1)
                            
                        state_of_charge_constraint.SetCoefficient(discharge, 1/efficiency)
                        
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
                    if year == (self.build_years-1) and ind == len(self.demand_profile)-1:
                        ending_state = self.solver.Constraint(self.initial_state_of_charge, self.initial_state_of_charge)
                        ending_state.SetCoefficient(state_of_charge, 1)
                        
                        
                #Loop through dispatchable resources.
                for resource in self.disp.index:
                    
                    resource_inds = self.resource_costs['resource']==resource

                    #Create generation variable for each dispatchable resource for every hour. 
                    if self.demand_profile.loc[ind, 'load_mw'] > 0:
                        gen = self.solver.NumVar(0, self.solver.infinity(), '_gen_year_'+ str(year) + '_hour' + str(ind))
                    else:
                        gen = self.solver.NumVar(0, 0, '_gen_year_'+ str(year) + '_hour' + str(ind))
                        if self.gen_zero_constraint == True:
                            gen_zero_constraint = self.solver.Constraint(0, 0)
                            gen_zero_constraint.SetCoefficient(gen, 1)

                    value_excess_energy_constraint.SetCoefficient(gen, 1)
                    
                    if resource == 'ci_shed':
                        if self.dr_shed_yearly_limit_constraint == True:
                            dr_shed_yearly_limit.SetCoefficient(gen, -1)
                            
                            
                        if self.dr_shed_daily_limit_constraint == True:
                            
                            if ind%24 == 0:
                                dr_shed_daily_limit.SetCoefficient(gen, -1)

                    
                    if self.storage_can_charge_from_grid == False:
                        if self.storage_charge_constraint == True:
                            storage_charge.SetCoefficient(gen, 1)


                    #Append hourly gen variable to the list for that resource, located in the disp_gen dictionary.
                    self.disp_gen[resource].append(gen)

                    #Calculate monetized emissions for given resource in selected hour.
                    resource_monetized_emissions = (self.disp.loc[resource, 'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton) + self.disp.loc[resource, 'nox_lbs_per_mwh']/2000*self.nox_cost_short_ton_la + self.disp.loc[resource, 'so2_lbs_per_mwh']/2000*self.so2_cost_short_ton_la + self.disp.loc[resource, 'pm25_lbs_per_mwh']/2000*self.pm25_cost_short_ton_la
                        
                    
                    #Calculate variable costs extrapolated over portfolio timespan.
                    if 'gas' in resource:
                        variable_om_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                        variable_fuel_cost_inds = self.resource_costs['cost_type']=='fuel_costs_per_mmbtu'
                        heat_rate_inds = self.resource_costs['cost_type']=='heat_rate_mmbtu_per_mwh'
                        variable_om_cost = self.resource_costs.loc[variable_om_inds & resource_inds,str(cost_year)].item() 
                        variable_fuel_cost = self.resource_costs.loc[variable_fuel_cost_inds & resource_inds, str(cost_year)].item()
                        variable_heat_rate = self.resource_costs.loc[heat_rate_inds & resource_inds, str(cost_year)].item()

                        variable_cost = variable_om_cost + (variable_fuel_cost*variable_heat_rate) + resource_monetized_emissions
                        

                    else:
                        variable_cost_inds = self.resource_costs['cost_type']=='variable_per_mwh'
                        variable_cost = self.resource_costs.loc[variable_cost_inds & resource_inds,str(cost_year)].item()+ resource_monetized_emissions
                    
                  
                    #If not in the last build_year, don't extrapolate variable costs. Just discount back to build_start_year. If in the last build year, extrapolate variable costs to the end of portfolio timespan and then discount total extrapolated costs back to build_start_year.
                    discount_factor = pow(self.growth_rate, -(year))
                    if year < (self.build_years-1):
                        variable_cost_discounted = variable_cost * discount_factor
                    else:
                        variable_cost_discounted = variable_cost * self.extrapolate_then_discount[year] * discount_factor
                    
                    #Incorporate extrapolated variable cost of hourly gen for each disp resource into objective function.
                    objective.SetCoefficient(gen, variable_cost_discounted)


                    #Add hourly gen variables for disp resources to the fulfill_demand constraint.
                    if self.fulfill_demand_constraint == True:
                        fulfill_demand.SetCoefficient(gen, 1)


                    #Initialize max_gen constraint: hourly gen must be less than or equal to capacity for each dispatchable resource.
                    max_gen = self.solver.Constraint(0, self.solver.infinity())
                    disp_capacity_cumulative = self.capacity_vars[resource][0:year+1]
                    #If in Harbor retirement year, limit Harbor generation to 0. 
                    if resource == 'gas_harbor' and year == self.build_years-1:
                
                        harbor_capacity = self.capacity_vars[resource][year]
                        max_gen.SetCoefficient(harbor_capacity, 1)
                        
                    else:
                        for i, var in enumerate(disp_capacity_cumulative):
                            max_gen.SetCoefficient(var, 1)

                            
                    max_gen.SetCoefficient(gen, -1)
                    
                
        
        return objective


    def discount_factor_from_cost(self, cost, discount_rate, build_years):
        self.growth_rate = 1.0 + discount_rate
        
        discount_factor = []       
        for year in range(build_years):

            value_decay_1 = pow(self.growth_rate, -(self.portfolio_timespan-year))
            value_decay_2 = pow(self.growth_rate, -1)
            try:
                extrapolate = cost * (1.0 - value_decay_1) / (1.0-value_decay_2)
            except ZeroDivisionError:
                extrapolate = cost
            discount_factor.append(extrapolate)
        
        return discount_factor
    
    
#     def _setup_resources(self):
#         resources = pd.read_csv('data/resources.csv')
    
#         if self.selected_resource != 'all':
#             resources = resources[resources['resource']==self.selected_resource]
            
#         resources = resources.set_index('resource')
        
#         return resources
    
    def _setup_resource_costs(self):
        if self.ee_cost_type == 'utility':
            resource_costs = pd.read_csv('data/resource_projected_costs_ee_utilitycosts.csv')
        elif self.ee_cost_type == 'total':
            resource_costs = pd.read_csv('data/resource_projected_costs_ee_totalcosts.csv')

        resource_costs = resource_costs[resource_costs['cost_decline_assumption']==self.cost_projections]
        
        return resource_costs

        
#     def _setup_storage(self):
#         storage = pd.read_csv('data/storage.csv')
#         num_columns = storage.columns[3:]
#         storage[num_columns] = storage[num_columns].astype(float)
#         storage = storage.set_index('resource')
        
#         return storage
    
    
    def _initialize_capacity_by_resource(self, build_years):
        
        capacity_by_resource = {}
        resource_max_constraints = {}
        
        for resource in self.resources.index:
            
            resource_max_constraints[resource]={}
            
            capacity_by_build_year = []
            
            if self.resources.loc[str(resource)]['legacy'] == 'n':
                #Create list of capacity variables for each year of build.
                
                if 'FCZ7' in resource:
                    resource_inds = self.ee_resource_potential['resource']==resource
                    if resource_inds.sum()>0:
                        ee_resource = self.ee_resource_potential[resource_inds]
                        if ee_resource.iloc[:,1:13].sum(axis=1).item() > 0:
                        
                            for year in range(build_years):

                                calendar_year = self.build_start_year + year
                                ee_max_mw = self.ee_resource_potential.loc[resource_inds, str(calendar_year)].item()

                                capacity = self.solver.NumVar(0, self.solver.infinity(), str(resource)+ '_' + str(year))
                                capacity_by_build_year.append(capacity)

                                #Capacity built up through this year must be less than or equal to the ee_max_mw limit in this year.
                                ee_capacity_constraint = self.solver.Constraint(0, ee_max_mw)
                                for i, var in enumerate(capacity_by_build_year):
                                    ee_capacity_constraint.SetCoefficient(var, 1)

                            capacity_by_resource[resource] = capacity_by_build_year
                        
                else:
                    resource_inds = self.resource_potential['resource']==resource
                    
                    #Need to connect both solar+storage systems and solar only systems to the same solar resource potential estimates (the sum of their capacities should be limited to resource potential max in that year).
                    if 'solar_rooftop_ci' in resource:
                        resource_inds = self.resource_potential['resource']=='solar_rooftop_ci'
                    if 'solar_rooftop_residentialSF' in resource:
                        resource_inds = self.resource_potential['resource']=='solar_rooftop_residentialSF'
                    if 'solar_rooftop_residentialMF' in resource:
                        resource_inds = self.resource_potential['resource']=='solar_rooftop_residentialMF'
                    
                    for year in range(build_years):
                    
                        calendar_year = self.build_start_year + year
                        capacity = self.solver.NumVar(0, self.solver.infinity(), str(resource)+ '_' + str(year))
                        capacity_by_build_year.append(capacity)
                        
                        if resource_inds.sum()>0:
                            resource_max_mw = self.resource_potential.loc[resource_inds, str(calendar_year)].item()
                            
                            #For BTM solar resources, add the capacity of solar only and solar+storage systems within each building type to the same resource potential constraint in each year. Ex. Standalone solar and solar+storage on commercial buildings should together be limited by the max resource potential for commercial solar in a given year.
                            if 'solar_rooftop_ci' in resource:
                                resource_dict = [val for key,val in resource_max_constraints.items() if 'solar_rooftop_ci' in key]
                                if any(resource_dict[0].keys()):
                                    
                                    constraint = resource_dict[0][year]
                                    for i, var in enumerate(capacity_by_build_year):
                                        constraint.SetCoefficient(var, 1)
                                else:
                                    resource_capacity_constraint = self.solver.Constraint(0, resource_max_mw)
                                    resource_max_constraints[resource][year] = resource_capacity_constraint
                                    
                            elif 'solar_rooftop_residentialSF' in resource:
                                resource_dict = [val for key,val in resource_max_constraints.items() if 'solar_rooftop_residentialSF' in key]
                                
                                if any(resource_dict[0].keys()):
                                    constraint = resource_dict[0][year]
                                    for i, var in enumerate(capacity_by_build_year):
                                        constraint.SetCoefficient(var, 1)
                                else:
                                    resource_capacity_constraint = self.solver.Constraint(0, resource_max_mw)
                                    resource_max_constraints[resource][year] = resource_capacity_constraint
                                 
                             
                            elif 'solar_rooftop_residentialMF' in resource:
                                resource_dict = [val for key,val in resource_max_constraints.items() if 'solar_rooftop_residentialMF' in key]
                                
                                if any(resource_dict[0].keys()):
                                    constraint = resource_dict[0][year]
                                    for i, var in enumerate(capacity_by_build_year):
                                        constraint.SetCoefficient(var, 1)
                                else:
                                    resource_capacity_constraint = self.solver.Constraint(0, resource_max_mw)
                                    resource_max_constraints[resource][year] = resource_capacity_constraint
                             
                            else:
                                resource_capacity_constraint = self.solver.Constraint(0, resource_max_mw)
                                resource_max_constraints[resource][year] = resource_capacity_constraint
                            

                        #Tie amount of storage built to amount of solar built for solar+storage systems.
                        #*** These constraints are initialized only once in code above (not every year). So it only ties solar and storage together across all years, not in individual years. May consider rewriting code to include a constraint in every year that ties solar+storage together in each year.
                        if resource == 'solar_rooftop_residentialSF_solarStorage':
                            if self.residential_storage_tied_to_solar_SF_constraint == True:
                                self.residential_storage_tied_to_solar_SF.SetCoefficient(capacity, 1)

                        if resource == 'solar_rooftop_residentialMF_solarStorage':
                            if self.residential_storage_tied_to_solar_MF_constraint == True:
                                self.residential_storage_tied_to_solar_MF.SetCoefficient(capacity, 1)

                        if resource == 'solar_rooftop_ci_solarStorage':
                            if self.commercial_storage_tied_to_solar_constraint == True:
                                self.commercial_storage_tied_to_solar.SetCoefficient(capacity, 1)
                            
                    capacity_by_resource[resource] = capacity_by_build_year
            
            else:
                #If resource is legacy resource, capacity "built" in year 0 of build years must be less than or equal to existing capacity. If build_start_year is the same as Harbor retirement year, capacity must be 0. Built capacity in subsequent build years must be 0.
                
                existing_mw = self.resources.loc[str(resource)]['existing_mw']
                for year in range(build_years):
                    if year == 0:
                        if year+self.build_start_year != self.harbor_retirement_year:
                            
                            capacity = self.solver.NumVar(0, existing_mw, str(resource)+ '_' + str(year))
                            
                        else:
                            capacity = self.solver.NumVar(0, 0, str(resource)+ '_' + str(year))
                    else:
                        capacity = self.solver.NumVar(0, 0, str(resource)+ '_' + str(year))
                    capacity_by_build_year.append(capacity)
                capacity_by_resource[resource] = capacity_by_build_year
                
        return capacity_by_resource
    
        
    def _initialize_storage_capacity_vars(self, build_years):
        storage_capacity_vars = {}
        
        #Constrain total utility-scale storage to Harbor's capacity.
        if self.utility_storage_limit_constraint == True:
            utility_storage_limit = self.solver.Constraint(0, 452)
        
        for resource in self.storage.index:
            
            storage_capacity_by_build_year = []
            if self.storage.loc[str(resource)]['legacy'] == 'n':
                #Create list of capacity variables for each year of build.
                for year in range(build_years):
                    capacity = self.solver.NumVar(0, self.solver.infinity(), str(resource)+ '_' + str(year))
                    storage_capacity_by_build_year.append(capacity)
                    
                    #If storage is utility-scale, add to total utility storage capacity constraint.
                    storage_utility_resources = ['storage_utility_2hr','storage_utility_4hr','storage_utility_6hr']
                    if resource in storage_utility_resources:
                        if self.utility_storage_limit_constraint == True:
                            utility_storage_limit.SetCoefficient(capacity, 1)
                     
                    elif resource == 'storage_residentialSF_4hr_solarStorage':
                        if self.residential_storage_tied_to_solar_SF_constraint == True:
                            self.residential_storage_tied_to_solar_SF.SetCoefficient(capacity, -1/self.ratio_solar_to_storage_sf)
                         
                    elif resource == 'storage_residentialMF_4hr_solarStorage':
                        if self.residential_storage_tied_to_solar_MF_constraint == True:
                            self.residential_storage_tied_to_solar_MF.SetCoefficient(capacity, -1/self.ratio_solar_to_storage_mf)
                            
                    elif resource == 'storage_ci_4hr_solarStorage':
                        if self.commercial_storage_tied_to_solar_constraint == True:
                            self.commercial_storage_tied_to_solar.SetCoefficient(capacity, -1/self.ratio_solar_to_storage_comm)
                            
                    
                storage_capacity_vars[resource] = storage_capacity_by_build_year

        return storage_capacity_vars
    
    
    #Set up WattTime health damages and CO2 costs.
    
#     def _setup_ladwp_marginal_co2(self):
#         ladwp_marginal_co2 = pd.read_csv('data/WattTime_MOER/LDWP_MOERv3_CO2_2019.csv')
#         ladwp_marginal_co2['date'] = pd.to_datetime(ladwp_marginal_co2['timestamp_utc']).dt.date
#         ladwp_marginal_co2['hour'] = pd.to_datetime(ladwp_marginal_co2['timestamp_utc']).dt.hour
#         ladwp_marginal_co2 = ladwp_marginal_co2.groupby(['date','hour']).mean(['moer (lbs CO2/MWh)']).reset_index()
        
#         return ladwp_marginal_co2
    
#     def _setup_ladwp_marginal_healthdamages(self):
#         ladwp_marginal_healthdamages = pd.read_csv('data/WattTime_MOER/LDWP_healthdamage_moer.csv')
#         index_list = ladwp_marginal_healthdamages[ladwp_marginal_healthdamages['healthdamage_moer'].isna()].index.tolist()
#         for index in index_list:
#             previous_index = index-1
#             ladwp_marginal_healthdamages.loc[index, 'healthdamage_moer']= ladwp_marginal_healthdamages.loc[previous_index, 'healthdamage_moer']
        
#         if (ladwp_marginal_healthdamages['healthdamage_moer'].isna()).sum()>0:
#             print('Nan values in LADWP marginal healthdamages file.')
        
#         return ladwp_marginal_healthdamages
    
#     def _setup_health_costs_emissions_la(self):
#         health_costs_emissions_la = pd.read_csv('data/pollutant_health_impacts/COBRA_LADWPplants_healthCosts.csv')
        
#         return health_costs_emissions_la
    

    def solve(self):
        self.objective.SetMinimization()
        status = self.solver.Solve()
        
        if status == self.solver.OPTIMAL:
            print("Solver found optimal solution.")
            print('Problem solved in %f milliseconds' % self.solver.wall_time())
            print('Problem solved in %d iterations' % self.solver.iterations())
            
            print('Number of variables =', self.solver.NumVariables())
            print('Number of constraints =', self.solver.NumConstraints())
            
            print('Solution:')
            print('Objective value (total cost in 2018$) =', self.solver.Objective().Value())

            
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
            
    def results_to_csv(self, today_date, string_to_identify_lp_run):
        
        input_param_list = [
            'solver_type',
            'selected_resource', 
            'initial_state_of_charge', 
            'storage_lifespan', 
            'portfolio_timespan',
            'storage_can_charge_from_grid', 
            'wholesale_cost_electricity_mwh',
            'discount_rate', 
            'cost', 
            'health_cost_range',
            'cost_projections',
            'build_start_year', 
            'harbor_retirement_year',
            'ee_cost_type', 
            'transmission_capex_cost_per_mw',
            'transmission_annual_cost_per_mw', 
            'storage_resilience_incentive_per_kwh', 
            'resilient_storage_grid_fraction', 
            'social_cost_carbon_short_ton', 
            #'avoided_marginal_generation_cost_per_mwh', 
            'diesel_genset_carbon_per_mw',
            'diesel_genset_pm25_per_mw', 
            'diesel_genset_nox_per_mw', 
            'diesel_genset_so2_per_mw', 
            'diesel_genset_pm10_per_mw', 
            'diesel_genset_fixed_cost_per_mw_year', 
            'diesel_genset_mmbtu_per_mwh', 
            'diesel_genset_cost_per_mmbtu', 
            'diesel_genset_run_hours_per_year',
        ]
        
        input_df_list = [
            'demand_profile',
            'ladwp_marginal_co2',
            'ladwp_marginal_healthdamages',
            'profiles',
            'ee_resource_potential',
            'resource_potential',
            'max_excess_energy',
            'caiso_lmp',
            'resources',
            'storage',
            'health_cost_emissions_la',
        ]


        results_dict = self.__dict__

        solver_type = results_dict['solver_type']
        harbor_retirement_year = str(results_dict['harbor_retirement_year'])
        if self.solve() == 0:
            solution_result = 'OPTIMAL'
        if self.solve() == 1:
            solution_result = 'FEASIBLE'
        if self.solve() == 2:
            solution_result = 'INFEASIBLE'
        if self.solve() == 3:
            solution_result = 'UNBOUNDED'
        if self.solve() == 4:
            solution_result = 'ABNORMAL'

        folder_name = solver_type + str(string_to_identify_lp_run) +'_retirement'+ harbor_retirement_year + '_' + solution_result +'_' + str(today_date) + str(np.random.rand(1,1).item())
        path = os.path.join('model_run_results', folder_name)
        os.mkdir(path)

        model_inputs_df = pd.DataFrame({'input':input_param_list})

        for input_name in input_param_list:
            input_value = results_dict[input_name]
            input_name_inds = model_inputs_df['input']==input_name
            model_inputs_df.loc[input_name_inds,'value']=input_value
        model_inputs_df.to_csv(path+'/input_parameter_values.csv')

        for input_df in input_df_list:
            df = results_dict[input_df]
            df.to_csv(path+'/'+input_df+'.csv')

        #Get capacity results for each year and save as csv.
        build_years = int(harbor_retirement_year)-2020+1
        for year in range(build_years):

            capacities_mw = self.get_capacities_mw(year).items()
            if year == 0:
                resource_list = list(self.get_capacities_mw(year).keys())
                capacities_df = pd.DataFrame({'resource':resource_list})

            capacities = list(self.get_capacities_mw(year).values())

            capacity_year = year + 2020
            capacities_df[str(capacity_year)] = capacities

        capacities_df['Units']='MW'

        capacities_df.to_csv(path+'/capacity_results_mw.csv')
        
        #Get generation results for each year and save as separate csvs.
#         nonstorage_resources = [s for s in resource_list if 'storage' not in s]
#         storage_resources = [s for s in resource_list if 'storage' in s]

        for year in range(build_years):

            hourly_gen_df = pd.DataFrame()
            hourly_storage_df = pd.DataFrame()

            results_hour_start = 8760*year
            results_hour_end = 8760*year + 8760

            resource_gen_dict = {}
            storage_charge_dict = {}
            storage_discharge_dict = {}
            storage_state_of_charge_dict = {}

            for resource in self.disp.index:
                gen_list = []
                for i_gen in self.disp_gen[resource][results_hour_start:results_hour_end]:
                    gen = i_gen.solution_value()
                    gen_list.append(gen)
                if any(gen_list):
                    print(resource)
                    resource_gen_dict[resource]=gen_list

            for resource in self.nondisp.index:
                if resource in self.capacity_vars.keys():
                    profile_max = max(self.profiles[resource])
                    profile = self.profiles[resource] / profile_max

                    capacity_cumulative = self.capacity_vars[resource][0:year+1]
                    capacity_total = 0
                    for i, var in enumerate(capacity_cumulative):
                        capacity_total += var.solution_value()

                    gen_list = profile * capacity_total
                    if any(gen_list):
                        resource_gen_dict[resource]=gen_list

            for resource in self.storage.index:
                storage_hourly_charge = []
                for i,var in enumerate(self.storage_charge_vars[resource]):
                    charge = var.solution_value()
                    storage_hourly_charge.append(-charge)

                storage_hourly_charge = storage_hourly_charge[results_hour_start:results_hour_end]

                if any(storage_hourly_charge):
                    storage_charge_dict[resource] = storage_hourly_charge

            for resource in self.storage.index:
                storage_hourly_discharge = []
                for i,var in enumerate(self.storage_discharge_vars[resource]):
                    discharge = var.solution_value() 
                    storage_hourly_discharge.append(discharge)

                storage_hourly_discharge = storage_hourly_discharge[results_hour_start:results_hour_end]
                if any(storage_hourly_discharge):
                    storage_discharge_dict[resource]=storage_hourly_discharge
                    
            for resource in self.storage.index:
                storage_hourly_state_of_charge = []
                for i,var in enumerate(self.storage_state_of_charge_vars[resource]):
                    state_of_charge = var.solution_value()
                    storage_hourly_state_of_charge.append(state_of_charge)

                storage_hourly_state_of_charge = storage_hourly_state_of_charge[results_hour_start:results_hour_end]
                if any(storage_hourly_state_of_charge):
                    storage_state_of_charge_dict[resource]=storage_hourly_state_of_charge


            for resource in self.resources.index:
                if resource in resource_gen_dict.keys():
                    hourly_gen = resource_gen_dict[resource]
                    hourly_gen_df[str(resource)]= hourly_gen
                else:
                    hourly_gen_df[str(resource)]= 0
            
            hourly_gen_df = hourly_gen_df.fillna(0)
            hourly_gen_df.to_csv(path+'/hourly_generation_results_mwh_year{}.csv'.format(year))

            for resource in self.storage.index:
                if resource in storage_charge_dict.keys():
                    hourly_charge = storage_charge_dict[resource]
                    hourly_storage_df[str(resource)+'_CHARGE']= hourly_charge
                else:
                    hourly_storage_df[str(resource)+'_CHARGE'] = 0

            for resource in self.storage.index:
                if resource in storage_discharge_dict.keys():
                    hourly_discharge = storage_discharge_dict[resource]
                    hourly_storage_df[str(resource)+'_DISCHARGE']= hourly_discharge
                else:
                    hourly_storage_df[str(resource)+'_DISCHARGE']= 0
                
            for resource in self.storage.index:
                if resource in storage_state_of_charge_dict.keys():
                    hourly_state_of_charge = storage_state_of_charge_dict[resource]
                    hourly_storage_df[str(resource)+'_STATE_OF_CHARGE']= hourly_state_of_charge
                else:
                    hourly_storage_df[str(resource)+'_STATE_OF_CHARGE']= 0
                
            hourly_storage_df = hourly_storage_df.fillna(0)
            hourly_storage_df.to_csv(path+'/hourly_storage_results_mwh_year{}.csv'.format(year))

    
    def get_lcoe_per_mwh(self):
        
        #Write a csv of LCOE for each resource in each build year.
        lcoe_per_mwh_by_resource_df = pd.DataFrame()
        lcoe_per_mwh_w_emissions_by_resource_df = pd.DataFrame()
        
        lcoe_per_mwh_by_resource = {}
        
        for i,resource in enumerate(self.capacity_vars.keys()):
            
            lcoe_per_mwh_by_resource_df.loc[i,'resource']=resource
            lcoe_per_mwh_w_emissions_by_resource_df.loc[i,'resource']=resource
            
            lcoe_per_mwh_by_resource[resource]={}
            
            for build_year in range(self.build_years):
        
                resource_inds = self.resource_costs['resource']==resource
                cost_year = self.build_start_year + build_year

                capex = self.lcoe_dict[build_year][resource]['capex']
                fixed_extrapolated = self.lcoe_dict[build_year][resource]['fixed_extrapolated']

                #Get range of hours for last build_year to index correctly into the list of generation variables.
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
                        
                        variable_cost_w_cobenefits = variable_cost + resource_monetized_emissions_mwh
                    
                    variable_cost_extrapolated = variable_cost * self.extrapolate_then_discount[build_year]
                    print(resource, 'variable_cost_extrapolated=', variable_cost_extrapolated)
                    variable_cost_w_cobenefits_extrapolated = variable_cost_w_cobenefits * self.extrapolate_then_discount[build_year]
                    print(resource, 'variable_cost_w_cobenefits_extrapolated=', variable_cost_w_cobenefits_extrapolated)


                    self.lcoe_dict[build_year][resource]['variable_extrapolated'] = variable_cost_extrapolated

                    summed_gen = 0
                    #Sums generation in the given build year.
                    for i_gen in self.disp_gen[str(resource)][demand_start_hour:demand_end_hour]:
                        summed_gen += i_gen.solution_value()

                    capacity_cumulative = self.capacity_vars[resource][0:self.build_years+1]
                    capacity_total = 0
                    for i, var in enumerate(capacity_cumulative):
                        capacity_total += var.solution_value()

                    if summed_gen >0:
                        mwh_per_mw = summed_gen / capacity_total
               
                        self.lcoe_dict[build_year][resource]['annual_generation_per_mw'] = mwh_per_mw
                    else:
                        print(resource,': Annual generation is 0 for this resource. Cannot calculate LCOE.')

                elif resource in self.nondisp.index:
                    
                    resource_monetized_emissions_mwh = (self.nondisp.loc[resource, 'co2_short_tons_per_mwh']*self.social_cost_carbon_short_ton) + self.nondisp.loc[resource, 'nox_lbs_per_mwh']/2000*self.nox_cost_short_ton_la + self.nondisp.loc[resource, 'so2_lbs_per_mwh']/2000*self.so2_cost_short_ton_la + self.nondisp.loc[resource, 'pm25_lbs_per_mwh']/2000*self.pm25_cost_short_ton_la

                    profile_max = max(self.profiles[resource])
                    summed_gen = sum(self.profiles[resource] / profile_max)
                    
                    #Add variable costs extrapolated over portfolio timespan to lcoe dictionary.
                    self.lcoe_dict[build_year][resource]['variable_extrapolated']=self.resource_costs.loc[variable_cost_inds & resource_inds, str(cost_year)].item() * summed_gen * self.extrapolate_then_discount[build_year]

                    #Add annual mwh generated per mw capacity to lcoe dictionary.
                    self.lcoe_dict[build_year][resource]['annual_generation_per_mw']=summed_gen
                    
                    variable_cost_extrapolated = self.lcoe_dict[build_year][resource]['variable_extrapolated']
                    mwh_per_mw = self.lcoe_dict[build_year][resource]['annual_generation_per_mw']
                    print(resource, mwh_per_mw)

                if 'annual_generation_per_mw' in self.lcoe_dict[build_year][resource].keys():
                    
                    
                    #Calculate lcoe_per_mwh without co-benefits.
                    variable_costs = variable_cost_extrapolated * mwh_per_mw
                    lcoe_per_mw = capex + fixed_extrapolated + variable_costs
                    lcoe_per_mwh = lcoe_per_mw / (mwh_per_mw*self.portfolio_timespan)
                    
                    lcoe_per_mwh_by_resource[resource][build_year] = lcoe_per_mwh
                    
                    #Calculate lcoe_per_mwh with co-benefits.
                    variable_costs_w_cobenefits = variable_cost_w_cobenefits_extrapolated * mwh_per_mw
                    lcoe_per_mw_w_cobenefits = capex + fixed_extrapolated + variable_costs_w_cobenefits
                    lcoe_per_mwh_w_cobenefits = lcoe_per_mw_w_cobenefits / (mwh_per_mw*self.portfolio_timespan)
                    
                    lcoe_per_mwh_by_resource_df.loc[i, cost_year] = lcoe_per_mwh
                    lcoe_per_mwh_w_emissions_by_resource_df.loc[i, cost_year] = lcoe_per_mwh_w_cobenefits
                
            lcoe_per_mwh_by_resource_df['lcoe_type'] = 'no_cobenefits'
            lcoe_per_mwh_w_emissions_by_resource_df['lcoe_type'] = 'w_resource_emissions'
            
            lcoe_per_mwh_by_resource_df = pd.concat([lcoe_per_mwh_by_resource_df,lcoe_per_mwh_w_emissions_by_resource_df], ignore_index=True)
                
        return lcoe_per_mwh_by_resource_df
        
        
                
                    
    def get_capacities_mw(self, build_year):
        
        capacities = {}
        for resource in self.capacity_vars:
            capacity = self.capacity_vars[resource][build_year].solution_value()
            capacities[resource] = capacity
            
        for resource in self.storage_capacity_vars:
            capacity = self.storage_capacity_vars[resource][build_year].solution_value()
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
            if resource in self.capacity_vars.keys():
                profile_max = max(self.profiles[resource])
                summed_gen = sum(self.profiles[resource]) / profile_max
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
#         self.profiles = pd.read_csv('data/gen_self.profiles.csv')
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
#             profile_max = max(self.profiles[resource])
#             summed_gen = sum(self.profiles[resource]) / profile_max
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