from ortools.algorithms import pywrapknapsack_solver
import math


def round_down(x, a):
    """Helper function to round `x` down to the nearest multiple of `a`.
    """
    return round(math.floor(x / a) * a, -int(math.floor(math.log10(a))))


def solve_multi_knapsack(profits, weights, capacities, verbose=True):
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        # KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'MKP_Solve')
        KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER, 'MKP_Solve')
    solver.Init(profits, weights, capacities)
    computed_profit = solver.Solve()
    if verbose:
        print(('optimal profit = ' + str(computed_profit)))

    idx_chosen_items = []
    for i in range(len(profits)):
        if solver.BestSolutionContains(i):
            idx_chosen_items.append(i)
    return idx_chosen_items
