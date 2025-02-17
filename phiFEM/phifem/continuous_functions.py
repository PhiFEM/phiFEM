from   basix.ufl import _ElementBase
from   collections.abc import Callable
from   dolfinx.fem import Function, FunctionSpace
import functools
import numpy as np
import numpy.typing as npt
from   phiFEM.phifem.derivatives import negative_laplacian
from   phiFEM.phifem.utils import immutable

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class ContinuousFunction:
    """ Class to represent a continuous (in the sense of non-discrete) function."""

    def __init__(self, expression: NDArrayFunction) -> None:
        """ Intialize a continuous function.

        Args:
            expression: a method giving the expression of the continuous function.
        """
        self.expression: NDArrayFunction = expression
        self.interpolations: dict[_ElementBase, Function] = {}

        # Updates the signature of __call__ to match expression's signature
        functools.update_wrapper(self, expression)

    @immutable("x")
    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Call the continuous function for computation.

        Args:
            x: coordinates of the points in which the function is computed.
        
        Returns:
            The array of values of the function.
        """

        ndim = x.ndim
        if ndim > 2:
            raise ValueError("The input (ndarray) dimension must be 2 at most.")
        
        shape = x.shape
        if shape[0] > 3:
            raise ValueError("The input (coordinates) dimension must be 3 at most.")

        return self.expression(x)

    def interpolate(self, FE_space: FunctionSpace) -> Function:
        """ Interpolate the function onto a finite element space.
        A dict is created in order to remember previous interpolations and save computational time.

        Args:
            FE_space: a finite element space in which the function will be interpolated.

        Returns:
            The dict of interpolations, with a new entry if needed.
        """
        element = FE_space.element
        if element not in self.interpolations.keys():
            interpolation = Function(FE_space)
            interpolation.interpolate(self.__call__)
            self.interpolations[element] = interpolation
        return self.interpolations[element]
            
class Levelset(ContinuousFunction):
    """ Class to represent a levelset function as a continuous function."""

    def __init__(self, expression: NDArrayFunction) -> None:
        super().__init__(expression)
        self.detection_expression: NDArrayFunction | None = None
    
    def set_detection_expression(self, expression: NDArrayFunction) -> None:
        self.detection_expression = expression

    def get_detection_expression(self):
        detection_expression = self.expression
        if self.detection_expression is not None:
            detection_expression = self.detection_expression
        return detection_expression

    def exterior(self, t: float, padding: float =0.) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]:
        """ Compute a lambda function determining if the point x is outside the domain defined by the isoline of level t.
        
        Args:
            t: level of the isoline.
            padding: padding parameter.
        
        Return:
            lambda function taking a tuple of coordinates and returning a boolean 
        """
        return lambda x: self(x) > t + padding
    
    def interior(self, t: float, padding: float =0.) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]:
        """ Compute a lambda function determining if the point x is inside the domain defined by the isoline of level t.
        
        Args:
            t: level of the isoline.
            padding: padding parameter.
        
        Return:
            lambda function taking a tuple of coordinates and returning a boolean 
        """
        return lambda x: self(x) < t - padding
    
class ExactSolution(ContinuousFunction):
    """ Class to represent the exact solution of the PDE as a continuous function."""

    def __init__(self, expression: NDArrayFunction) -> None:
        """ Intialize a continuous function.

        Args:
            expression: a method giving the expression of the continuous function.
        """
        super().__init__(expression)
        self.nlap: ContinuousFunction | None = None 
    
    def compute_negative_laplacian(self) -> None:
        """ Compute the negative laplacian of the function."""
        comp_nlap = negative_laplacian(self)
        self.nlap = ContinuousFunction(comp_nlap)
    
    def get_negative_laplacian(self) -> ContinuousFunction:
        if self.nlap is None:
            raise ValueError("CONTINUOUS_FUNCTION_NAME.nlap is None, did you forget to compute the negative laplacian ? (CONTINUOUS_FUNCTION_NAME.compute_negative_laplacian)")
        return self.nlap