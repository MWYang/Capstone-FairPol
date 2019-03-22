import dask.dataframe as dd
import geopy.distance
import pandas as pd
import scipy as sp
from dask.diagnostics import ProgressBar
from fairpol.helpers import download_file


def main(data_dir='../data/'):
    url = "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD"
    fname = 'AllChicagoCrimes.csv'
    target = data_dir + fname
    print("Downloading raw Chicago crime data...")
    download_file(url, target)

    ProgressBar().register()  # Setup Dask progress bar
    chicago = dd.read_csv(target, assume_missing=True, parse_dates=['Date'])
    chicago = chicago[sp.logical_and(2012 <= chicago['Year'],
                                     chicago['Year'] <= 2017)]

    chicago = chicago.loc[chicago["Primary Type"] == "HOMICIDE"]
    chicago = chicago.dropna(subset=[
        'X Coordinate', 'Y Coordinate', 'Community Area', 'Latitude',
        'Longitude', 'Date'
    ])

    chicago['x'] = chicago['X Coordinate']
    chicago['y'] = chicago['Y Coordinate']
    chicago['t'] = pd.to_numeric(chicago.Date)

    # 0-1 normalize t to prevent overflow issues in PredPol
    chicago.t -= chicago.t.min()
    chicago.t /= chicago.t.max()

    print("Processing raw Chicago crime data...")
    chicago = chicago.compute()
    print("Processing complete.")

    # Compute kilometer distance from longitude and latitude,
    x_min = chicago.x.min()
    y_min = chicago.y.min()
    x_max = chicago.x.max()
    y_max = chicago.y.max()
    MIN_LON = chicago.loc[chicago.x == x_min, 'Longitude'].min()
    MAX_LON = chicago.loc[chicago.x == x_max, 'Longitude'].max()
    MIN_LAT = chicago.loc[chicago.y == y_min, 'Latitude'].min()
    MAX_LAT = chicago.loc[chicago.y == y_max, 'Latitude'].max()
    meters_height = geopy.distance.distance((MIN_LAT, MIN_LON),
                                            (MAX_LAT, MIN_LON)).m
    meters_width = geopy.distance.distance((MIN_LAT, MIN_LON),
                                           (MIN_LAT, MAX_LON)).m
    # print(meters_height, meters_width)

    # Normalize by these values to obtain roughly 150m by 150m grid cells
    chicago.x -= chicago.x.min()
    chicago.y -= chicago.y.min()
    chicago.x /= chicago.x.max()
    chicago.y /= chicago.y.max()
    chicago.x *= round(meters_width / 150 / 100, 2)
    chicago.y *= round(meters_height / 150 / 100, 2)

    chicago.to_csv(data_dir + 'ChicagoHomicides2012to2017.csv')
    print("Successfully wrote cleaned Chicago crime data to file.")

    chicago_small = chicago[sp.logical_and(2014 <= chicago['Year'],
                                           chicago['Year'] <= 2015)]
    chicago_small.to_csv(data_dir + 'ChicagoHomicides2014to2015.csv')


if __name__ == '__main__':
    main()
