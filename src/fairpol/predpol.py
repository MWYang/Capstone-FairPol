import math
import pandas as pd
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
from shapely.geometry import shape, Point
from tqdm.auto import tqdm


class PredPol:
    """A class for training PredPol on a dataset and making predictions.

    Attributes:
        train: a pointer to the pandas DataFrame indicating the training set

        T: the maximum time value encountered across all data (test and train)

        height: the maximum y-value in `train`

        width: the maximum x-value in `train`

        grid_size: the grid size specified in the constructor

        grid_cells: a pandas DataFrame where each row is a grid cell. contains
        `x` and `y` columns indicating the center point of the grid cell.

        interpolator: a scipy.interpolate.RegularGridInterpolator object, used
        to match coordinates to a grid cell. Call by using
        `self.interpolator(vals)`, where `vals` is a M x 2 array-like, where each
        row indicates a different coordinate. Returns a M x 2 array, where each
        row are the coordinates the center of the matched grid cell.

        eta: numeric, internal PredPol parameter for aftershock spatial decay
        omega: numeric, internal PredPol parameter for aftershock temporal decay
        sigma: numeric, internal PredPol parameter for background spatial decay
    """
    def __init__(self, train, T=10.0, grid_size=0.1):
        """Constructor for PredPol object.

        Args:
            train: a pandas DataFrame where each row is an observed crime, used
            to learn the model parameters. Must contain at least the following
            columns:
                t: numeric, indicating the time of the crime. All values must be
                less than the `T` supplied as an argument to this constructor.
                x: numeric, indicating the horizontal position of the crime.
                y: numeric, indicating the vertical position of the crime.
            For numerical safety (avoiding overflows), it's recommended that t,
            x, and y are all normalized to have relatively small values.

            T: float, indicating the maximum `t` value in the _combined_ train
            and test set of data.

            grid_size: float, indicating the level of discretization/total
            number of grid cells in the PredPol model. Smaller grid sizes have
            more grid cells.
        Returns: an initialized PredPol object before fitting parameter values.
        """
        self.train = train
        self.T = T
        self.height = train.y.max()
        self.width = train.x.max()
        self.grid_size = grid_size

        # Generate the center of each grid cell
        x_vals = sp.arange(0, self.width, self.grid_size)
        y_vals = sp.arange(0, self.height, self.grid_size)
        diff = round(self.grid_size / 2,
            -int(math.floor(math.log10(self.grid_size))) + 1)
        x_vals += diff
        y_vals += diff

        xs, ys = sp.meshgrid(x_vals, y_vals)
        # print(xs)
        # print(ys)
        self.grid_cells = sp.vstack(sp.dstack((xs, ys)))
        self.grid_cells = pd.DataFrame(self.grid_cells, columns=['x', 'y'])

        self.interpolator = RegularGridInterpolator(
            points=(x_vals, y_vals),
            values=sp.arange(1, len(self.grid_cells) + 1).reshape(
                (len(y_vals), len(x_vals))
            ).transpose(),
            method='nearest',
            bounds_error=False, fill_value=None
        )

        self.eta = None
        self.omega = None
        self.sigma = None

    def fit(self, steps=200, eps=1e-5, num_restarts=3):
        """Learn the parameters omega, sigma, eta from training data `train` via
        EM.

        Args:
            steps: integer, the maximum number of EM steps to take before
            terminating
            eps: integer, the tolerance threshold for terminating early from the
            optimization routine. Optimization stops when the change in each
            parameter is no greater than the value specified here.
            num_restarts: integers, the number of restarts to perform

        Returns: a pointer to `self`, where the parameter attributes omega,
        sigma, and eta have been learned.
        """
        train = self.train
        N = len(train)
        T = self.T

        # Allocate "p" matrices
        p_aftershock = sp.zeros((N, N))
        p_background = sp.zeros((N, N))

        # Massage data into proper format and compute reusable values for E-step
        # (only needs to be done once)

        # Provides iteration over the data (each pair of rows) without using a
        # `for` loop
        i, j = sp.ogrid[0:N, 0:N]
        t_i = train.iloc[i.reshape(N, )]['t'].values.reshape(N, 1)
        t_j = train.iloc[j.reshape(N, )]['t'].values.reshape(1, N)
        x_i = train.iloc[i.reshape(N, )]['x'].values.reshape(N, 1)
        x_j = train.iloc[j.reshape(N, )]['x'].values.reshape(1, N)
        y_i = train.iloc[i.reshape(N, )]['y'].values.reshape(N, 1)
        y_j = train.iloc[j.reshape(N, )]['y'].values.reshape(1, N)

        distance = (x_j - x_i)**2 + (y_j - y_i)**2
        time_check = t_i < t_j
        origin_check = sp.logical_and((x_i != x_j), (y_i != y_j))
        trigger_check = time_check * origin_check
        t_diff = t_j - t_i

        best_params = None
        best_likelihood = None
        for _ in range(num_restarts):
            # Initialize parameters to random values
            # Parameters for triggering kernel
            omega = sp.absolute(stats.norm.rvs(scale=0.10))  # time decay
            sigma = sp.absolute(sp.randn())  # spatial decay
            # Parameter for background rate
            eta = sp.absolute(sp.randn())
            # Loop until convergence
            for step in tqdm(range(steps)):
                old_parameters = sp.stack([omega, sigma, eta])

                # E-step: Calculate p matrices according to equations (9) and (10)
                # "[P and P^b] contain the probabilities that event i triggered
                # homicide j through either the triggering kernel g or the
                # background rate kernel"

                p_aftershock = trigger_check * omega * sp.exp(-omega * t_diff) \
                    * stats.norm.pdf(distance, scale=sigma)
                p_background = origin_check * stats.norm.pdf(distance, scale=eta)

                # Normalize as necessary
                Z = p_aftershock + p_background  # also used for likelihood
                nonzero_Z = Z > 0
                p_aftershock[nonzero_Z] /= Z[nonzero_Z]
                p_background[nonzero_Z] /= T * Z[nonzero_Z]

                # M-step: Update parameters

                aftershock_sum = sp.sum(p_aftershock)
                time_gaps = T - t_i
                omega_denom = sp.sum(p_aftershock * (t_j - t_i)) \
                            + sp.sum(time_gaps * sp.exp(-omega * time_gaps))
                omega = aftershock_sum / omega_denom

                sigma = sp.sum(p_aftershock * distance) / (2.0 * aftershock_sum)
                eta = sp.sum(p_background * distance) / (2.0 * sp.sum(p_background))

                diff = sp.absolute(old_parameters - sp.stack([omega, sigma, eta])).max()
                if diff < eps:
                    print("Convergence met after {} iterations: {}".format(
                        step, sp.sum(Z)
                    ))
                    break
            # Check the likelihood given the discovered parameter values by
            # summing over Z
            likelihood = sp.sum(Z)
            if best_likelihood is None or likelihood > best_likelihood:
                best_params = omega, sigma, eta
                best_likelihood = likelihood

        self.p_aftershock, self.p_background = p_aftershock, p_background
        self.omega, self.sigma, self.eta = best_params
        return self

    def get_demographics(self, shape_json, shape_json_key,
                         demographic_data,
                         region_column,
                         black_column,
                         white_column,
                         region_preprocess=lambda x: x):
        """Associate each grid cell with demographic information.

        Args:
            shape_json: JSON file object containing shapefile information
            shape_json_key: the name of the region identifier in `shape_json`
            demographic_data: a pandas DataFrame containing demographic
            information; must contain the columns specified by `[region_column,
            black_column, white_column]`
            region_column: a string identifying the column containing region
            names
            black_column: a string identifying the column identifying the
            proportion black
            white_column: a string identifying the column identifying the
            proportion white
            region_preprocess: optional lambda, in case the string identifiers
            in the shape_json need to be transformed in order to match the
            region identifiers in `demographic_data.region_column`

        Returns: a pointer to `self`, where the `grid_cells` attribute now
        contains columns for the corresponding region, percentage black, and
        percentage white
        """
        train = self.train
        x_min = train.x.min()
        y_min = train.y.min()
        x_max = train.x.max()
        y_max = train.y.max()
        MIN_LON = train.loc[train.x == x_min, 'Longitude'].min()
        MAX_LON = train.loc[train.x == x_max, 'Longitude'].max()
        MIN_LAT = train.loc[train.y == y_min, 'Latitude'].min()
        MAX_LAT = train.loc[train.y == y_max, 'Latitude'].max()

        def _associate(row):
            region = -1
            x = row.x
            y = row.y

            lon, lat = MIN_LON + (x / x_max) * (MAX_LON - MIN_LON),\
                MIN_LAT + (y / y_max) * (MAX_LAT - MIN_LAT)
            point = Point(lon, lat)
            for feature in shape_json['features']:
                polygon = shape(feature['geometry'])
                if polygon.contains(point):
                    region = region_preprocess(feature['properties'][shape_json_key])
                    break

            black = 0
            white = 0
            if region != -1:
                try:
                    row = demographic_data.loc[
                        demographic_data[region_column] == region,
                        [black_column, white_column]
                    ]
                    black = float(row[black_column])
                    white = float(row[white_column])

                    total = black + white
                    black = black / total
                    white = white / total
                except:
                    raise(RuntimeWarning(region, black, white))
                    pass

            return region, black, white

        tqdm.pandas()
        (
            self.grid_cells['region'],
            self.grid_cells['black'],
            self.grid_cells['white']
        ) = zip(*self.grid_cells.progress_apply(_associate, axis=1))
        return self

    def predict(self, data):
        """Given the observations in `data`, return the estimated conditional
        intensities at each grid cell for the next day.

        Typically, `data` will be the values in `train` concatenated with
        additional rows of observed crime (the internal PredPol parameters do
        not need to be re-computed with every new observation).

        Args:
            data: a pandas DataFrame matching the specification of the `train`
            argument to the PredPol() constructor
        Returns:
            an array with length `self.grid_cells.shape[0]`, where the ith value
            corresponds to the predicted intensity for the ith grid cell
        """

        N = len(data)
        sigma = self.sigma
        omega = self.omega
        eta = self.eta

        # Initialize the grid of values at which to calculate conditional
        # intensities
        grid_cells = self.grid_cells
        M = grid_cells.shape[0]
        rates = sp.zeros(M)

        # Provides iteration over the data (each pair of grid_cells and rows in
        # the data)
        i, j = sp.ogrid[0:N, 0:M]
        data_by_i = data.iloc[i.reshape(N, )]
        t_i = data_by_i['t'].values.reshape(N, 1)
        x_i = data_by_i['x'].values.reshape(N, 1)
        y_i = data_by_i['y'].values.reshape(N, 1)

        x_j = grid_cells.iloc[j.reshape(M, ), 0].values.reshape(1, M)
        y_j = grid_cells.iloc[j.reshape(M, ), 1].values.reshape(1, M)
        # Calculate reusable values
        distance = (x_j - x_i)**2 + (y_j - y_i)**2

        # print(rates.shape)

        # Calculate the intensity at each location
        # Estimate the background average by KDE
        rates += sp.sum(stats.norm.pdf(distance, scale=eta), axis=0)

        # Estimate the amount of excitation at each cell from past crimes
        rates += sp.sum(omega * sp.exp(-omega * t_i) * stats.norm.pdf(distance, scale=sigma), axis=0)

        # idx_sorted = sp.argsort(rates)[::-1]
        # return rates[idx_sorted], grid_cells[idx_sorted, :]
        return rates
