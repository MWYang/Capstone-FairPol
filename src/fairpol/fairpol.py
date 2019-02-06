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


def compute_fp_knapsack(assess_obj, num_cells=sp.arange(100, 700, 100),
                        max_gap=20, precision=4, verbose=True):
    '''Need to define this function's behavior and return type.
    Q: Should I compute accuracy/fairness here?
    Q: How should I return the results of the chosen cells?
    '''

    black = assess_obj.pred_obj.grid_cells.black
    white = assess_obj.pred_obj.grid_cells.white
    results = sp.zeros((len(num_cells), len(assess_obj.results)))

    for i, (lambda_col, actual_col) in assess_obj._iterator():
        profits = assess_obj.results[actual_col].values.astype(int)

        pct_black_caught = (assess_obj.results[actual_col] * black).values
        pct_black_caught /= sp.sum(pct_black_caught)
        pct_white_caught = (assess_obj.results[actual_col] * white).values
        pct_white_caught /= sp.sum(pct_white_caught)

        pct_policed_gap = pct_black_caught - pct_white_caught
        pct_policed_gap = (pct_policed_gap * (10 ** precision)).astype(int)
        # print(pct_policed_gap.describe())

        pct_overpoliced_black = pct_policed_gap.copy()
        pct_overpoliced_black[pct_policed_gap < 0] = 0

        pct_overpoliced_white = -pct_policed_gap.copy()
        pct_overpoliced_white[pct_policed_gap > 0] = 0

        idx_chosen_items = solve_multi_knapsack(profits.tolist(), [
            [1] * len(profits),
            pct_overpoliced_black.tolist(), pct_overpoliced_white.tolist()
        ], [max_num_cells, max_gap, max_gap], verbose)

    return


def compute_fp_sorting(assess_obj):
    pass
