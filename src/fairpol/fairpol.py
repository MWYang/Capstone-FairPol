from fairpol.helpers import solve_multi_knapsack


def fairness_knapsack(df, max_num_cells, max_gap=20, precision=4, verbose=True):
    profits = df.rate.astype(int)

    pct_policed_gap = df['pct_black_predicted'] - df['pct_white_predicted']
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
    return idx_chosen_items
