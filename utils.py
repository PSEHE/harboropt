import pandas as pd

#Right now, the code sums generation from all units. Can change code to sum only OTC units. Make sure units are consistent with capacity variables (MW vs KW).
def get_harbor_data(filename): 
    harborgen = pd.read_csv(filename)
    harborgen.fillna(0, inplace=True) 
    harborgen['datetime'] = pd.to_datetime(harborgen['OP_DATE']) + pd.to_timedelta(harborgen['OP_HOUR'],unit='h')
    harborgen.drop(['OP_DATE','OP_HOUR'],inplace=True,axis=1)
    harborgen['mwh'] = harborgen['OP_TIME'] * harborgen['GLOAD..MW.']
    harborgen.set_index(['datetime'], inplace=True, drop=True)
    return harborgen
  
def get_solar_data():
    solargen = pd.read_csv('data/solar_gen_1kwSAM.csv')
    solargen['kwh'] = solargen['AC inverter power | (W)']/1000
    solargen['datetime'] = pd.to_datetime(solargen['Time stamp'], format='%b %d,  %I:%M %p')
    solargen.drop(['Time stamp', 'AC inverter power | (W)'],inplace=True,axis=1)
    return solargen
