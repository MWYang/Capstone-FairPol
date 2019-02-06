from tqdm.auto import tqdm
import pandas as pd
import scipy as sp


class AssessPol():
    def __init__(self, pred_obj):
        self.pred_obj = pred_obj
        self.results = pred_obj.grid_cells.copy()
        self.lambda_columns = []
        self.actual_columns = []

    def generate_predictions(self, data, date_range):
        '''Each prediction generates:
        - a predicted conditional intensity for each grid cell
        - a count of the actual number of crimes for each grid cell
        Append each of these results into a new column in self.results
        Save a list of the column identifiers for predicted intensities
        and actual counts
        '''
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
        return self.results[self.lambda_columns]

    def get_actual_counts(self):
        return self.results[self.actual_columns]

    def _iterator(self):
        # Wrap zip object in tqdm to get a progress bar
        return tqdm(enumerate(zip(self.lambda_columns, self.actual_columns)),
            total=len(self.lambda_columns))

    def compute_accuracy(self, methods='all'):
        """
        - methods (string):
            One of ['all', 'predpol', 'god', 'running_count'];
            Default: 'all'
        """
        accuracy = {
            method: sp.zeros((len(self.results), len(self.lambda_columns)))
            for method in ['predpol', 'god', 'running_count']
        }
        running_count = count_seen(self.pred_obj, self.pred_obj.train)['num_observed']

        for i, (lambda_col, actual_col) in self._iterator():
            actual_vals = self.results[actual_col].values
            accuracy['god'][:, i] = sp.sort(actual_vals)[::-1]

            sorted_idx = sp.argsort(self.results[lambda_col])[::-1]
            accuracy['predpol'][:, i] = actual_vals[sorted_idx]

            sorted_idx = sp.argsort(running_count)[::-1]
            accuracy['running_count'][:, i] = actual_vals[sorted_idx]

            running_count += self.results[actual_col]

        # Compute CI and p-values here
        for k, v in accuracy.items():
            accuracy[k] = sp.sum(v, axis=1)
            accuracy[k] = sp.cumsum(accuracy[k] / sp.sum(accuracy[k]))
        return pd.DataFrame(accuracy)

    def compute_fairness(self, methods='all'):
        """
        - methods (string):
            One of ['all', 'predpol', 'god', 'running_count'];
            Default: 'all'
        """
        fairness = {
            method: sp.zeros((len(self.results), len(self.lambda_columns)))
            for method in ['predpol', 'god', 'running_count']
        }
        running_count = count_seen(self.pred_obj, self.pred_obj.train)['num_observed']
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

            sorted_idx = sp.argsort(running_count)[::-1]
            fairness['running_count'][:, i] += fair_diff[sorted_idx]

            running_count += self.results[actual_col]

        for k, v in fairness.items():
            fairness[k] = sp.sum(v, axis=1)
            fairness[k] = sp.cumsum(fairness[k]) / len(self.lambda_columns)
        return pd.DataFrame(fairness)


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

