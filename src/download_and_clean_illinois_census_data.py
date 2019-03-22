import pandas as pd
import censusdata
from fairpol.helpers import download_file


def main(verbose=False, data_dir='../data/'):
    if verbose:
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.precision', 2)

        print("Available race variables:")
        print(censusdata.search('acs5', 2015, 'label', 'race'))
        print("Table to download:")
        censusdata.printtable(censusdata.censustable('acs5', 2015, 'B02001'))

    variables = list(censusdata.censustable('acs5', 2015, 'B02001').keys())
    # remove variables for margin of errors
    variables = list(filter(lambda x: x[-1] != 'M', variables))
    if verbose:
        print("Variables:")
        print(variables)

    illinois_demo = censusdata.download(
        'acs5', 2015, censusdata.censusgeo([('state', '17'), ('tract', '*')]),
        variables)

    illinois_demo.rename({
        'B02001_001E': 'total',
        'B02001_002E': 'white',
        'B02001_003E': 'black',
        'B02001_004E': 'native',
        'B02001_005E': 'asian',
        'B02001_006E': 'pacific',
        'B02001_007E': 'other',
        'B02001_008E': 'two_or_more',
        'B02001_009E': 'two_or_more_including_other',
        'B02001_010E': 'two_or_more_excluding_other'
    },
                         axis='columns',
                         inplace=True)

    illinois_demo.other = illinois_demo.other + \
        illinois_demo['two_or_more_including_other'] + \
        illinois_demo['two_or_more_excluding_other']

    illinois_demo = illinois_demo[[
        'total', 'white', 'black', 'native', 'asian', 'pacific', 'other'
    ]]
    total = illinois_demo.total
    illinois_demo.white /= total
    illinois_demo.black /= total
    illinois_demo.native /= total
    illinois_demo.asian /= total
    illinois_demo.pacific /= total
    illinois_demo.other /= total

    illinois_demo['censusgeo'] = illinois_demo.index
    illinois_demo.reset_index(level=0, drop=True, inplace=True)

    illinois_demo['tract'] = illinois_demo['censusgeo'].apply(lambda x: x.geo[
        2][1]).astype(str)
    illinois_demo['county'] = illinois_demo['censusgeo'].apply(lambda x: x.geo[
        1][1])
    illinois_demo['county_name'] = illinois_demo['censusgeo'].apply(
        lambda x: x.name.split(',')[1][1:-7])
    illinois_demo.drop('censusgeo', axis='columns', inplace=True)

    if verbose:
        print(illinois_demo.sample(frac=10 / len(illinois_demo)))
        print(illinois_demo.describe())

    illinois_demo = illinois_demo.loc[illinois_demo.county_name == 'Cook']
    illinois_demo.to_csv(data_dir + 'Illinois2015CensusTractsDemographics.csv')
    print("Successfully downloaded Illinois demographic data.")

    url = "https://github.com/uscensusbureau/citysdk/raw/master/v2/GeoJSON/500k/2015/17/tract.json"
    fname = 'Illinois2015CensusTracts.json'
    target = data_dir + fname
    download_file(url, target)
    print("Successfully downloaded Illinois census tract shapefile.")


if __name__ == '__main__':
    main()
