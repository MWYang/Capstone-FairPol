from context import fp
import json
import pandas as pd

chicago = pd.read_csv('../data/ChicagoHomicides2012to2017.csv',
    parse_dates=['Date'])
train = chicago.loc[chicago.Year <= 2014]
test = chicago.loc[chicago.Year == 2015]

with open('../data/Illinois2015CensusTracts.json', 'r') as f:
    ill_shape = json.load(f)
ill_demo = pd.read_csv('../data/Illinois2015CensusTractsDemographics.csv',
                    dtype={'tract': str, 'county': str}, index_col=0)

# reload(fp)
pp = fp.PredPol(chicago, grid_size=0.01)
pp.get_demographics(ill_shape, 'TRACTCE',
    ill_demo, 'tract', 'black', 'white',
    region_preprocess=lambda x: ''.join(x.split()))
pp.grid_cells.describe()
