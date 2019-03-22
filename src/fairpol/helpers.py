from ortools.algorithms import pywrapknapsack_solver
import math


def round_down(x, a):
    """Helper function to round `x` down to the nearest multiple of `a`.

    Args:
        x: float
        a: float
    Returns:
        The closest multiple of `a` to `x` without exceeding `x`.
    """
    return round(math.floor(x / a) * a, -int(math.floor(math.log10(a))))


def solve_multi_knapsack(profits, weights, capacities, verbose=True):
    """Wrapper function for solving a multi-dimensional knapsack problem. Uses
    Google's ORtools library.

    Args:
        profits: list of N integers, indicating the profits associated with each
            of N items.
        weights: multi-dimensional list (M * N) of integers, where M is the
            number of weight dimensions to consider and N is the number of items
        capacities: list of M integers, indicating the max capacity along each of
            M dimensions
    Returns:
        list of integer indices of the chosen items
    """
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
