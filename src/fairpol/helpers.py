from ortools.algorithms import pywrapknapsack_solver
import math
import os.path
import requests
from tqdm import tqdm


def download_file(url, target):
    """Helper function to download file with progress bar.

    Args:
        url: string, indicating location of file to be downloaded
        target: string, indicating final location and name of downloaded file
    Returns: None
    """
    if not os.path.isfile(target):
        # Download with progress bar code modified from:
        # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0

        with open(target, 'wb') as f:
            tqdm_obj = tqdm(
                r.iter_content(block_size),
                total=math.ceil(total_size / block_size),
                unit='KB',
                unit_scale=True)
            for data in tqdm_obj:
                wrote = wrote + len(data)
                f.write(data)
        print("Download complete.")
    else:
        print("File already downloaded.")


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
        KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER,
        'MKP_Solve')
    solver.Init(profits, weights, capacities)
    computed_profit = solver.Solve()
    if verbose:
        print(('optimal profit = ' + str(computed_profit)))

    idx_chosen_items = []
    for i in range(len(profits)):
        if solver.BestSolutionContains(i):
            idx_chosen_items.append(i)
    return idx_chosen_items
