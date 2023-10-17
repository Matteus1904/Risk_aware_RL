from abc import ABC, abstractmethod
import time
import scipy as sp
from scipy.optimize import minimize



    
class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    """

    @property
    @abstractmethod
    def engine(self):
        """Name of the optimization engine being used"""
        return "engine_name"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @staticmethod
    def verbose(opt_func):
        """
        A static method decorator that makes the decorated function verbose.

        This method will print the optimization time of the decorated function
        if the `verbose` attribute of the instance is set to True.

        Parameters:
        opt_func (function): The function to be decorated.

        Returns:
        function: The decorated function.
        """

        def wrapper(self, *args, **kwargs):
            tic = time.time()
            result = opt_func(self, *args, **kwargs)
            toc = time.time()
            if self.verbose:
                print(f"result optimization time:{toc-tic} \n")

            return result

        return wrapper


class SciPyOptimizer(Optimizer):
    """
    Optimizer class using the SciPy optimization library.

    Attributes:
        engine (str): Name of the optimization engine.
        opt_method (str): Optimization method to use.
        opt_options (dict): Options for the optimization method.
        verbose (bool): Whether to print optimization progress and timing.
    """

    engine = "SciPy"

    def __init__(self, opt_method, opt_options, verbose=False):
        """
        Initialize a SciPyOptimizer instance.

        :param opt_method: str, the name of the optimization method to use.
        :param opt_options: dict, options for the optimization method.
        :param verbose: bool, whether to print the optimization time and the objective function value before and after optimization.
        """
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose

    @Optimizer.verbose
    def optimize(self, objective, initial_guess, bounds, constraints=(), verbose=False):
        """
        Optimize the objective function using the specified method and options.

        :param objective: function, the objective function to optimize.
        :param initial_guess: array-like, the initial guess for the optimization.
        :param bounds: tuple, the lower and upper bounds for the optimization.
        :param constraints: tuple, the equality and inequality constraints for the optimization.
        :param verbose: bool, whether to print the objective function value before and after optimization.
        :return: array-like, the optimal solution.
        """

        weight_bounds = sp.optimize.Bounds(bounds[0], bounds[1], keep_feasible=True)

        before_opt = objective(initial_guess)
        opt_result = minimize(
            objective,
            x0=initial_guess,
            method=self.opt_method,
            bounds=weight_bounds,
            options=self.opt_options,
            constraints=constraints,
            tol=1e-7,
        )
        if verbose:
            print(f"before:{before_opt},\nafter:{opt_result.fun}")

        return opt_result.x