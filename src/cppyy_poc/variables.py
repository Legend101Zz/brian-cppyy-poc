"""
Variable tracking system.

In Brian2, every quantity (voltage, current, time, etc.) is represented
as a Variable object. These objects know their type, size, and how to
access their data.

This simplified version shows the core concepts without all of Brian2's
complexity.
"""

from typing import Any

import numpy as np


class Variable:
    """
    Base class for all variables.
    
    A Variable represents a quantity that exists in the simulation.
    It might be an array of values (like neuron voltages), a single
    constant (like the timestep), or a dynamically-sized array.
    """
    
    def __init__(self, name: str, dtype: type):
        """
        Initialize a variable.
        
        Args:
            name: The variable's identifier (e.g., 'v' for voltage)
            dtype: The Python type (float, int, etc.)
        """
        self.name = name
        self.dtype = dtype
    
    def get_cpp_type(self) -> str:
        """
        Convert Python type to C++ type.
        
        This is a simple mapping - real Brian2 has a more sophisticated
        system that handles units, complex types, etc.
        """
        type_map = {
            float: 'double',
            int: 'int',
            np.float64: 'double',
            np.int32: 'int',
        }
        return type_map.get(self.dtype, 'double')
    
    def get_value(self) -> Any:
        """Get the current value of this variable."""
        raise NotImplementedError("Subclasses must implement get_value")


class Constant(Variable):
    """
    A constant value that doesn't change during simulation.
    
    Examples: timestep (dt), time constant (tau), number of neurons (N)
    
    In the generated C++ code, constants become function parameters.
    """
    
    def __init__(self, name: str, value: float):
        """
        Create a constant.
        
        Args:
            name: Variable name (e.g., 'dt')
            value: The constant value (e.g., 0.0001)
        """
        super().__init__(name, type(value))
        self.value = value
    
    def get_value(self) -> float:
        """Return the constant value."""
        return self.value
    
    def __repr__(self):
        return f"Constant({self.name}={self.value})"


class ArrayVariable(Variable):
    """
    An array of values, one per neuron.
    
    Examples: voltage array (v), current array (I)
    
    In the generated C++ code, arrays become pointer parameters plus
    a size parameter. cppyy automatically extracts the pointer from
    NumPy arrays, achieving zero-copy data sharing.
    """
    
    def __init__(self, name: str, data: np.ndarray):
        """
        Create an array variable.
        
        Args:
            name: Variable name (e.g., 'v')
            data: NumPy array containing the values
        """
        super().__init__(name, data.dtype.type)
        self.data = data
    
    def get_value(self) -> np.ndarray:
        """Return the NumPy array."""
        return self.data
    
    def __repr__(self):
        return f"ArrayVariable({self.name}, shape={self.data.shape})"


class VariableResolver:
    """
    Manages the collection of variables for a code object.
    
    This is like Brian2's variable resolution system, but simplified.
    In the real system, variables can come from multiple sources
    (the group, the namespace, mathematical operations, etc.).
    
    Here we just track what variables exist and provide them to
    the code generator.
    """
    
    def __init__(self):
        """Initialize an empty variable collection."""
        self.variables = {}
    
    def add(self, variable: Variable):
        """
        Register a variable.
        
        Args:
            variable: The Variable object to add
        """
        self.variables[variable.name] = variable
    
    def get(self, name: str) -> Variable:
        """
        Get a variable by name.
        
        Args:
            name: Variable name
            
        Returns:
            The Variable object
            
        Raises:
            KeyError if variable doesn't exist
        """
        return self.variables[name]
    
    def get_all(self) -> dict:
        """Get all variables as a dictionary."""
        return self.variables.copy()
    
    def __repr__(self):
        return f"VariableResolver({list(self.variables.keys())})"       