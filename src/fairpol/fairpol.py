# from fairpol.assesspol import AssessPol
from fairpol.helpers import solve_multi_knapsack
from tqdm.auto import tqdm
import scipy as sp


def compute_fp_knapsack(assess_obj, num_cells,
                        total_cells_factor=5, max_gap=0.05, precision=4,
                        verbose=False):
    """Computes both a fairer selection of grid cells for each day tested in
    `assess_obj` and the accuracy/fairness curves for those results.

    Args:
        assess_obj: an AssessPol object
        num_cells: an array indicating the number of grid cells to compute the
            fairness modification on (calculating on every number in
            range(len(grid_cells)) is prohibitively expensive)
        total_cells_factor: an integer indicating the total number of cells to
            consider at every iteration. Rather than considering all grid cells
            at each step, we speed up computation by only comparing the top
            (num_cell * total_cells_factor) grid cells at each iteration.
        max_gap: an integer indicating the maximum tolerable fairness gap (the
            parameter D in chapter 4 of the final report).
        precision: an integer indicating the number of decimal points to
            preserve when passing weights to the MKP solver (which expects
            integer weights while the original fairness costs are in the unit
            range)
    Returns:
        a tuple consisting of:
            - a dictionary containing the information of which grid cells were
              chosen for each day and each number of grid cells in `num_cells`
              specifically: maps from datestrings to another dictionary, where
              that dictionary maps from numbers N in `num_cells` to a list of
              indexes
            - array, average accuracy results for each number of grid cells considered
            - array, average fairness results for each number of grid cells considered
    """
    knapsack_items = {}
    knapsack_accuracy = sp.zeros(len(num_cells))
    knapsack_fairness = sp.zeros(len(num_cells))

    black = assess_obj.pred_obj.grid_cells.black.fillna(0)
    white = assess_obj.pred_obj.grid_cells.white.fillna(0)
    # scale max_gap by the same factor as we scale the weights
    max_gap *= 10 ** precision

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
    # Compute fairness as an average over all days
    knapsack_fairness /= (i + 1)

    return knapsack_items, knapsack_accuracy, knapsack_fairness


def compute_fp_sorting(assess_obj, alphas):
    """Deprecated, do not use.
    """
    accuracy = sp.zeros((len(assess_obj.results), len(alphas)))
    fairness = accuracy.copy()
    black = assess_obj.pred_obj.grid_cells.black.fillna(0)
    white = assess_obj.pred_obj.grid_cells.white.fillna(0)
    # Save the original `results` dataframe in the assess_obj
    # Make a copy that this function will mutate
    orig_results = assess_obj.results
    assess_obj.results = assess_obj.results.copy()
    lambdas = assess_obj.get_predicted_intensities().apply(sp.log)
    tqdm_leave = assess_obj.tqdm_leave
    assess_obj.tqdm_leave = False
    for i, alpha in enumerate(tqdm(alphas)):
        # Calculate new lambda_cols based on alpha
        pct_black_predicted = lambdas.multiply(black, axis='index')
        pct_black_predicted = pct_black_predicted.divide(
            pct_black_predicted.sum(axis=0), axis=1)
        pct_white_predicted = lambdas.multiply(white, axis='index')
        pct_white_predicted = pct_white_predicted.divide(
            pct_white_predicted.sum(axis=0), axis=1)
        fair_diff = 1.0 - sp.absolute(pct_black_predicted - pct_white_predicted)
        avg_crime_captured = (pct_black_predicted + pct_white_predicted) / 2.0
        alpha_lambdas = (
            ((1.0 - alpha) * avg_crime_captured) + (alpha * fair_diff)
        )
        # Put new lambda values in `results` dataframe
        assess_obj.results[assess_obj.lambda_columns] = alpha_lambdas
        # Compute accuracy and fairness results
        accuracy[:, i] = assess_obj.compute_accuracy()['predpol']
        fairness[:, i] = assess_obj.compute_fairness()['predpol']
        pass
    # Restore the original `results` object
    assess_obj.results = orig_results
    assess_obj.tqdm_leave = tqdm_leave
    return accuracy, fairness
