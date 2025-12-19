"""
Variable tracking system.

In Brian2, every quantity (voltage, current, time, etc.) is represented
as a Variable object. These objects know their type, size, and how to
access their data.

This simplified version shows the core concepts without all of Brian2's
complexity.
"""

from typing import Any, Dict, List, Optional, Tuple

import cppyy
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
        
        Returns:
            The corresponding C++ type as a string.
            
        Example:
            >>> var = Variable('x', np.float64)
            >>> var.get_cpp_type()
            'double'
        """
        type_map = {
            float: 'double',
            int: 'int',
            np.float64: 'double',
            np.float32: 'float',
            np.int32: 'int32_t',
            np.int64: 'int64_t',
            bool: 'bool',
            np.bool_: 'bool',
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
    
    MEMORY MODEL:
    - Python/NumPy owns the memory
    - C++ receives a raw pointer to that memory
    - Changes in C++ are immediately visible in Python
    - No data copying occurs!
    
    Attributes:
        data: The underlying NumPy array
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
    
    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return self.data.size
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array."""
        return self.data.shape
    
    def __repr__(self):
        return f"ArrayVariable({self.name}, shape={self.data.shape})"

class DynamicArrayVariable(Variable):
    """
    A dynamically-sized array that can grow during simulation.
    
    This is essential for monitors that record data over time.
    Unlike fixed ArrayVariable, DynamicArrayVariable can:
    - Grow as new data is recorded
    - Provide efficient appending via C++ std::vector
    - Support both 1D (times) and 2D (recorded values) arrays
    
    IMPLEMENTATION:
    We use cppyy to define and instantiate C++ std::vector objects
    that can be efficiently resized. The data can then be viewed
    as a NumPy array without copying.
    
    MEMORY MODEL:
    - C++ owns the memory (via std::vector)
    - Python gets a view via cppyy's automatic conversion
    - Growing the vector may invalidate previous pointers
      (that's why we re-extract the data pointer after each resize)
    
    Attributes:
        cpp_vector: The underlying cppyy std::vector object
        dimensions: Number of dimensions (1 or 2)
    """
    
    # Track if we've defined the C++ helper classes yet
    _cpp_helpers_defined = False
    
    @classmethod
    def _ensure_cpp_helpers(cls):
        """
        Define C++ helper classes for dynamic arrays if not already done.
        
        This is called lazily on first use to avoid cppyy overhead
        if dynamic arrays aren't needed.
        """
        if cls._cpp_helpers_defined:
            return
            
        # Define a simple DynamicArray2D class in C++
        # This mirrors Brian2's DynamicArray2D but simplified
        cppyy.cppdef("""
        #include <vector>
        #include <cstring>
        #include <algorithm>
        
        namespace brian_poc {
            
            /**
             * A 2D dynamic array stored in row-major order.
             * 
             * This class provides:
             * - Efficient resizing with over-allocation
             * - Row-major storage for NumPy compatibility
             * - Direct data pointer access for zero-copy views
             */
            template<typename T>
            class DynamicArray2D {
            private:
                std::vector<T> buffer;
                size_t num_rows;
                size_t num_cols;
                
            public:
                DynamicArray2D(size_t rows = 0, size_t cols = 0)
                    : num_rows(rows), num_cols(cols)
                {
                    buffer.resize(rows * cols, T(0));
                }
                
                // Resize to new dimensions, preserving existing data where possible
                void resize(size_t new_rows, size_t new_cols) {
                    if (new_rows == num_rows && new_cols == num_cols) return;
                    
                    std::vector<T> new_buffer(new_rows * new_cols, T(0));
                    
                    // Copy existing data
                    size_t copy_rows = std::min(num_rows, new_rows);
                    size_t copy_cols = std::min(num_cols, new_cols);
                    
                    for (size_t i = 0; i < copy_rows; ++i) {
                        for (size_t j = 0; j < copy_cols; ++j) {
                            new_buffer[i * new_cols + j] = buffer[i * num_cols + j];
                        }
                    }
                    
                    buffer.swap(new_buffer);
                    num_rows = new_rows;
                    num_cols = new_cols;
                }
                
                // Access element at (row, col)
                T& operator()(size_t row, size_t col) {
                    return buffer[row * num_cols + col];
                }
                
                const T& operator()(size_t row, size_t col) const {
                    return buffer[row * num_cols + col];
                }
                
                // Get raw pointer for NumPy interop
                T* data() { return buffer.data(); }
                const T* data() const { return buffer.data(); }
                
                size_t rows() const { return num_rows; }
                size_t cols() const { return num_cols; }
                size_t size() const { return buffer.size(); }
            };
            
            // Explicit instantiations for common types
            template class DynamicArray2D<double>;
            template class DynamicArray2D<float>;
            template class DynamicArray2D<int>;
            
        }  // namespace brian_poc
        """)
        
        cls._cpp_helpers_defined = True
        
    def __init__(
        self,
        name: str,
        dtype: type = np.float64,
        dimensions: int = 1,
        initial_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Create a dynamic array variable.
        
        Args:
            name: Variable name (e.g., 't' for recorded times)
            dtype: NumPy data type for elements
            dimensions: 1 for vectors (like time), 2 for matrices (like recorded values)
            initial_shape: Initial size. For 1D: (n,), for 2D: (rows, cols)
        """
        super().__init__(name, dtype)
        self._ensure_cpp_helpers()
        
        self.dimensions = dimensions
        
        # Map NumPy dtype to C++ type for template instantiation
        cpp_type_map = {
            np.float64: 'double',
            np.float32: 'float',
            np.int32: 'int',
            np.int64: 'long',
        }
        cpp_type = cpp_type_map.get(dtype, 'double')
        
        if dimensions == 1:
            # Use std::vector directly for 1D
            self.cpp_vector = cppyy.gbl.std.vector[cpp_type]()
            if initial_shape:
                self.cpp_vector.resize(initial_shape[0])
        else:
            # Use our custom DynamicArray2D for 2D
            rows = initial_shape[0] if initial_shape else 0
            cols = initial_shape[1] if initial_shape and len(initial_shape) > 1 else 0
            self.cpp_array = cppyy.gbl.brian_poc.DynamicArray2D[cpp_type](rows, cols)
            
    def resize(self, new_shape: Tuple[int, ...]):
        """
        Resize the dynamic array.
        
        For 1D: new_shape = (new_length,)
        For 2D: new_shape = (new_rows, new_cols)
        
        IMPORTANT: This may invalidate any existing NumPy views!
        Always call get_value() again after resizing.
        """
        if self.dimensions == 1:
            self.cpp_vector.resize(new_shape[0])
        else:
            self.cpp_array.resize(new_shape[0], new_shape[1])
            
    def push_back(self, value: float):
        """
        Append a value to a 1D dynamic array.
        
        This is more efficient than resize + set for single appends.
        Only available for 1D arrays.
        
        Args:
            value: The value to append
        """
        if self.dimensions != 1:
            raise ValueError("push_back only available for 1D arrays")
        self.cpp_vector.push_back(value)
    
    def get_value(self) -> np.ndarray:
        """
        Get a NumPy array view of the data.
        
        IMPORTANT: This creates a VIEW, not a copy!
        - Changes to the NumPy array affect the C++ data
        - If the C++ array is resized, this view becomes invalid
        
        Returns:
            NumPy array view of the underlying data
        """
        if self.dimensions == 1:
            # Convert std::vector to NumPy array (creates a view)
            size = len(self.cpp_vector)
            if size == 0:
                return np.array([], dtype=self.dtype)
            
            # Get pointer and create view
            ptr = self.cpp_vector.data()
            return np.frombuffer(
                cppyy.ll.cast['double*'](ptr),
                dtype=self.dtype,
                count=size
            ).copy()  # Copy because vector might resize
        else:
            # For 2D, extract data from our DynamicArray2D
            rows = self.cpp_array.rows()
            cols = self.cpp_array.cols()
            if rows == 0 or cols == 0:
                return np.array([]).reshape(rows, cols)
            
            ptr = self.cpp_array.data()
            # Create view with proper shape
            arr = np.frombuffer(
                cppyy.ll.cast['double*'](ptr),
                dtype=self.dtype,
                count=rows * cols
            ).reshape(rows, cols).copy()
            return arr
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Current shape of the array."""
        if self.dimensions == 1:
            return (len(self.cpp_vector),)
        else:
            return (self.cpp_array.rows(), self.cpp_array.cols())
    
    def __repr__(self):
        return f"DynamicArrayVariable({self.name}, shape={self.shape}, dtype={self.dtype.__name__})"


class VariableResolver:
    """
    Manages the collection of variables for a code object.
    
    This is like Brian2's variable resolution system, but simplified.
    In the real system, variables can come from multiple sources
    (the group, the namespace, mathematical operations, etc.).
    
    The resolver:
    1. Tracks what variables exist
    2. Provides them to the code generator
    3. Handles variable lookup by name
    4. Supports iteration for namespace building
    
    Attributes:
        variables: Dictionary mapping names to Variable objects
    """
    
    def __init__(self):
        """Initialize an empty variable collection."""
        self.variables: Dict[str, Variable] = {}
    
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
            KeyError: If variable doesn't exist
        """
        return self.variables[name]
    
    def get_all(self) -> Dict[str, Variable]:
        """Get all variables as a dictionary."""
        return self.variables.copy()
    
    def get_arrays(self) -> Dict[str, ArrayVariable]:
        """Get only ArrayVariable instances."""
        return {
            name: var for name, var in self.variables.items()
            if isinstance(var, ArrayVariable)
        }
    
    def get_constants(self) -> Dict[str, Constant]:
        """Get only Constant instances."""
        return {
            name: var for name, var in self.variables.items()
            if isinstance(var, Constant)
        }
    
    def __contains__(self, name: str) -> bool:
        """Check if a variable exists."""
        return name in self.variables
    
    def __iter__(self):
        """Iterate over variable names."""
        return iter(self.variables)
    
    def __repr__(self):
        return f"VariableResolver({list(self.variables.keys())})"