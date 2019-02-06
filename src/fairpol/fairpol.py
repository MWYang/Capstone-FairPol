# from fairpol.assesspol import AssessPol
from fairpol.helpers import solve_multi_knapsack
import scipy as sp


# class FairPol(AssessPol):
#     def __init__(self, assess_obj=None, pred_obj=None):
#         if assess_obj is None and pred_obj is None:
#             raise(RuntimeError("Please initialize FairPol with either a PredPol or AssessPol object."))
#         if assess_obj:
#             self.results = assess_obj.results.copy()
#             self.lambda_columns = assess_obj.lambda_columns.copy()
#             self.actual_columns = assess_obj.actual_columns.copy()
#         else:
#             super().__init__(pred_obj)
#             print("Don't forget to run generate_predictions().")
#         self.knapsack_results = None
#         self.sorting_results = None


def compute_fp_knapsack(assess_obj, num_cells,
                        total_cells_factor=5, max_gap=500, precision=4,
                        verbose=False):
    '''Computes both a fairer selection of grid cells for each day tested in
    `assess_obj` and the accuracy/fairness curves for those results.
    '''
    knapsack_items = {}
    knapsack_accuracy = sp.zeros(len(num_cells))
    knapsack_fairness = sp.zeros(len(num_cells))

    black = assess_obj.pred_obj.grid_cells.black.fillna(0)
    white = assess_obj.pred_obj.grid_cells.white.fillna(0)

    for i, (lambda_col, actual_col) in assess_obj._iterator():
        chosen_items = {}
        # lambda_col[:-7] is the date-string
        knapsack_items[lambda_col[:-7]] = chosen_items

        # Compute values needed for knapsack
        profits = assess_obj.results[lambda_col].values.astype(int)
        idx_profits_sorted = sp.argsort(profits)[::-1]

        pct_black_pred = (sp.log(assess_obj.results[lambda_col]) * black).values
        pct_black_pred /= sp.sum(pct_black_pred)
        pct_white_pred = (sp.log(assess_obj.results[lambda_col]) * white).values
        pct_white_pred /= sp.sum(pct_white_pred)
        pct_policed_gap = pct_black_pred - pct_white_pred
        pct_policed_gap = (pct_policed_gap * (10 ** precision)).astype(int)
        # print(pct_policed_gap.describe())

        # First part of equality constraint
        pct_overpoliced_black = pct_policed_gap.copy()
        pct_overpoliced_black[pct_policed_gap < 0] = 0

        # Second part of equality constraint
        pct_overpoliced_white = -pct_policed_gap.copy()
        pct_overpoliced_white[pct_policed_gap > 0] = 0

        # Compute values needed for assessment
        num_actual = assess_obj.results[actual_col].values
        pct_black_caught = (assess_obj.results[actual_col] * black).values
        pct_black_caught /= sp.sum(pct_black_caught)
        pct_white_caught = (assess_obj.results[actual_col] * white).values
        pct_white_caught /= sp.sum(pct_white_caught)
        fair_diff = sp.nan_to_num(pct_black_caught - pct_white_caught)

        for j, N in enumerate(num_cells):
            # Take only the top N * total_cells_factor cells in terms of
            # predicted intensities for knapsack to improve prediction speed
            idx_taken = idx_profits_sorted[:N * total_cells_factor]

            idx_chosen = solve_multi_knapsack(
                profits[idx_taken].tolist(), [
                    [1] * len(idx_taken),
                    pct_overpoliced_black[idx_taken].tolist(),
                    pct_overpoliced_white[idx_taken].tolist()
                ], [N, max_gap, max_gap], verbose
            )
            # idx_taken[idx_chosen] are the indices of the original grid cells
            # selected by the knapsack procedure
            chosen_items[N] = idx_taken[idx_chosen]
            knapsack_accuracy[j] += sp.sum(num_actual[chosen_items[N]])
            knapsack_fairness[j] += sp.sum(fair_diff[chosen_items[N]])

    # Compute accuracy as a percentage of total crime
    knapsack_accuracy /= assess_obj.get_actual_counts().values.sum()
    knapsack_fairness /= (i + 1)

    return knapsack_items, knapsack_accuracy, knapsack_fairness


def compute_fp_sorting(assess_obj, alphas):
    pass
