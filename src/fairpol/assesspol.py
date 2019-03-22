from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import scipy as sp


class AssessPol():
    """A class for assessing the performance of a PredPol object. Maintains
    previously computed results as part of its state.

    Attributes:
        pred_obj: a PredPol object
        results: a pandas DataFrame, extends `pred_obj.grid_cells` with more
            columns that indicate the predicted intensities and actual crime
            counts in each grid cell for each of the days in the test set
        lambda_columns: list of strings indicating the columns in `results` with
            predicted intensities
        actual_columns: list of strings indicating the columns in `results` with
            actual counts
        self.tqdm_leave: boolean, whether or not to persist `tqdm` progress bars,
            default: False
    """
    def __init__(self, pred_obj):
        """Constructor for AssessPol. Nothing complicated happening here.

        Args:
            pred_obj: a PredPol object to assess
        """
        self.pred_obj = pred_obj
        self.results = pred_obj.grid_cells.copy()
        self.lambda_columns = []
        self.actual_columns = []
        self.tqdm_leave = False

    def generate_predictions(self, data, date_range):
        """Compute and store predicted intensities for each day in `date_range`.
        Each prediction generates:
        - a predicted conditional intensity for each grid cell
        - a count of the actual number of crimes for each grid cell
        Each of these results is appended into a new column in self.results,
        and a list of these column identifiers is saved in `self.lambda_columns`
        and `self.actual_columns`.

        Args:
            data: a pandas DataFrame where each row is an observed crime, used
            to learn the model parameters. Typically will contain the rows in
            `self.pred_obj.train` and additional ones.Must contain at least the
            following columns:
                t: numeric, indicating the time of the crime. All values must be
                less than the `T` supplied as an argument to this constructor.
                x: numeric, indicating the horizontal position of the crime.
                y: numeric, indicating the vertical position of the crime.
            For numerical safety (avoiding overflows), it's recommended that t,
            x, and y are all normalized to have relatively small values.
            date_range: an array of numpy datetimes indicating the test days
        """
        results = self.results
        pp = self.pred_obj

        # Iterate over each day i
        for day_i in tqdm(date_range):
            day_i_train_idx = data.Date < day_i
            day_i_train = data.loc[day_i_train_idx]
            day_i_test_idx = data.Date.dt.date == day_i
            day_i_test = data.loc[day_i_test_idx]

            # Make predictions
            lambdas = pp.predict(day_i_train)
            # Record how much crime actually occurred on the ith day
            actual = count_seen(pp, day_i_test)['num_observed'].values

            self.lambda_columns.append(str(day_i) + '_lambda')
            self.actual_columns.append(str(day_i) + '_actual')
            results[self.lambda_columns[-1]] = lambdas
            results[self.actual_columns[-1]] = actual
        return self

    def get_predicted_intensities(self):
        """Convenience function for returning the predicted intensities
        """
        return self.results[self.lambda_columns]

    def get_actual_counts(self):
        """Convenience function for returning the actual crime counts
        """
        return self.results[self.actual_columns]

    def _iterator(self):
        """Helper function for returning an iterator through each day's predicted
        intensities and actual crime counts, with a progress bar

        Returns: a tqdm iterator object where each item is:
            (index, name of predictions column, name of actual results column)
        """
        return tqdm(enumerate(zip(self.lambda_columns, self.actual_columns)),
            total=len(self.lambda_columns), leave=self.tqdm_leave)

    def compute_accuracy(self):
        """Computes accuracy across the range in `self.date_range`.

        Returns: a pandas DataFrame with three columns corresponding to each
            kind of prediction method (PredPol, perfect prediction (god), and
            the baseline (naive_count)). The entries of each column are an array
            where the ith entry is the average accuracy over `self.date_range`
            when visiting i number of grid cells
        """
        accuracy = {
            method: sp.zeros((len(self.results), len(self.lambda_columns)))
            for method in ['predpol', 'god', 'naive_count']
        }
        naive_count = count_seen(self.pred_obj, self.pred_obj.train)['num_observed']

        for i, (lambda_col, actual_col) in self._iterator():
            actual_vals = self.results[actual_col].values
            accuracy['god'][:, i] = sp.sort(actual_vals)[::-1]

            sorted_idx = sp.argsort(self.results[lambda_col])[::-1]
            accuracy['predpol'][:, i] = actual_vals[sorted_idx]

            sorted_idx = sp.argsort(naive_count)[::-1]
            accuracy['naive_count'][:, i] = actual_vals[sorted_idx]

            naive_count += self.results[actual_col]

        # Compute CI and p-values here
        for k, v in accuracy.items():
            accuracy[k] = sp.sum(v, axis=1)
            accuracy[k] = sp.cumsum(accuracy[k] / sp.sum(accuracy[k]))
        return pd.DataFrame(accuracy)

    def compute_fairness(self):
        """Computes fairness across the range in `self.date_range`.

        Returns: a pandas DataFrame with three columns corresponding to each
            kind of prediction method (PredPol, perfect prediction (god), and
            the baseline (naive_count)). The entries of each column are an array
            where the ith entry is the average fairness over `self.date_range`
            when visiting i number of grid cells
        """
        fairness = {
            method: sp.zeros((len(self.results), len(self.lambda_columns)))
            for method in ['predpol', 'god', 'naive_count', 'random']
        }
        naive_count = count_seen(self.pred_obj, self.pred_obj.train)['num_observed']
        black = self.pred_obj.grid_cells.black.fillna(0)
        white = self.pred_obj.grid_cells.white.fillna(0)

        for i, (lambda_col, actual_col) in self._iterator():
            pct_black_caught = (self.results[actual_col] * black).values
            pct_black_caught /= sp.sum(pct_black_caught)
            pct_white_caught = (self.results[actual_col] * white).values
            pct_white_caught /= sp.sum(pct_white_caught)
            # On some days, no crime occurs. The following line treats those results
            # as zeros.
            fair_diff = sp.nan_to_num(pct_black_caught - pct_white_caught)

            sorted_idx = sp.argsort(self.results[actual_col])[::-1]
            fairness['god'][:, i] += fair_diff[sorted_idx]

            sorted_idx = sp.argsort(self.results[lambda_col])[::-1]
            fairness['predpol'][:, i] += fair_diff[sorted_idx]

            sorted_idx = sp.argsort(naive_count)[::-1]
            fairness['naive_count'][:, i] += fair_diff[sorted_idx]

            naive_count += self.results[actual_col]

            fairness['random'][:, i] += fair_diff[sorted_idx.sample(frac=1)]

        for k, v in fairness.items():
            fairness[k] = sp.sum(v, axis=1)
            fairness[k] = sp.cumsum(fairness[k]) / len(self.lambda_columns)
        return pd.DataFrame(fairness)

    def assess_calibration(self):
        """Assess if PredPol is calibrated by conditioning on predicted intensity
        and checking the correlation between number of crimes and demographics.
        """
        black = self.pred_obj.grid_cells.black
        not_nan = sp.logical_not(sp.isnan(black.values))

        bins = sp.histogram_bin_edges(self.get_predicted_intensities(), bins='auto')
        correlations = sp.empty((len(self.lambda_columns), len(bins)))
        correlations[:] = sp.nan
        for i, (lambda_col, actual_col) in self._iterator():
            idx_bins = sp.digitize(self.results[lambda_col], bins)
            for j in range(len(bins)):
                idx_selected = sp.logical_and(idx_bins == j, not_nan)
                if sp.sum(idx_selected) > 2:
                    actual = self.results.loc[idx_selected, actual_col]
                    demographics = black.loc[idx_selected]
                    correlations[i, j] = sp.stats.pearsonr(actual, demographics)[0]

        correlations /= (i + 1)  # take the average over the results
        return correlations


def count_seen(pred_obj, test_data):
    """Given the crime data in `test_data`, compute how much crime was actually
    observed in each grid cell in `pred_obj`.
    """
    result = pred_obj.grid_cells[['x', 'y']].copy()
    counts = pred_obj.interpolator(test_data[['x', 'y']]).astype(int)
    counts = pd.DataFrame(counts)[0].value_counts()
    counts = counts.reindex(index=result.index, axis='index', fill_value=0)
    result['num_observed'] = counts
    return result


def plot_accuracy(accuracy, plot_perfect=False):
    fig, ax = plt.subplots()
    fraction_flagged = sp.arange(0.0, 1.0, 1.0 / len(accuracy))
    ax.plot(fraction_flagged, accuracy['predpol'], label='PredPol', color='blue')
    ax.plot(fraction_flagged, fraction_flagged, label='Random', color='orange')
    if plot_perfect:
        ax.plot(fraction_flagged, accuracy['god'], label='PerfectPrediction', color='green')
    ax.plot(fraction_flagged, accuracy['naive_count'], label='NaiveCounting', color='red')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set(xlabel='Grid cells visited', ylabel='Crime caught')
    return fig, ax


def plot_fairness(fairness, plot_perfect=False):
    fig, ax = plt.subplots()
    fraction_flagged = sp.arange(0.0, 1.0, 1.0 / len(fairness))
    ax.plot(fraction_flagged, fairness['predpol'], label='PredPol', color='blue')
    ax.plot(fraction_flagged, fairness['random'], label='Random', color='orange')
    if plot_perfect:
        ax.plot(fraction_flagged, fairness['god'], label='PerfectPrediction', color='green')
    ax.plot(fraction_flagged, fairness['naive_count'], label='NaiveCounting', color='red')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    if mpl.rcParams['text.usetex']:
        ax.set(xlabel='Grid cells visited',
               ylabel='\% black crime caught - \% white crime caught')
    else:
        ax.set(xlabel='Grid cells visited',
               ylabel='% black crime caught - % white crime caught')
    return fig, ax
