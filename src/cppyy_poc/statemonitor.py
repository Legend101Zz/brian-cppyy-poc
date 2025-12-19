"""
StateMonitor: Records state variables over time.

The StateMonitor is essential for analyzing simulation results. It:
1. Records specified variables at each timestep
2. Stores data in dynamically-growing arrays
3. Provides easy access to recorded data for plotting

KEY CONCEPTS:
- Uses DynamicArray2D for efficient storage (rows=timesteps, cols=neurons)
- Only records specified neuron indices (not all neurons)
- Shares memory with C++ for zero-copy recording

COMPARISON TO BRIAN2:
Brian2's StateMonitor uses Cython with DynamicArray2D backing.
Our version uses cppyy with similar dynamic arrays but:
- Everything is JIT compiled
- No .pyx files needed
- Simpler template system
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cppyy
import numpy as np

from .clock import Clock
from .variables import (ArrayVariable, Constant, DynamicArrayVariable,
                        VariableResolver)

# Counter for unique monitor IDs
_statemonitor_counter = 0


class StateMonitor:
    """
    Records state variables from a NeuronGroup during simulation.
    
    The monitor observes a source group and records specified variables
    at each timestep. Data is stored in 2D arrays where:
    - Rows = timesteps
    - Columns = recorded neurons
    
    MEMORY MANAGEMENT:
    Recording uses C++ std::vector internally for efficient growth.
    The recorded data can be accessed as NumPy arrays for analysis.
    
    Attributes:
        source: The NeuronGroup being monitored
        variables: List of variable names being recorded
        record: Indices of neurons to record (True = all)
        t: Array of recorded times
        
    Example:
        >>> neurons = NeuronGroup(100, "dv/dt = (-v + I) / tau")
        >>> monitor = StateMonitor(neurons, 'v', record=[0, 10, 50])
        >>> 
        >>> for _ in range(1000):
        ...     neurons.update()
        ...     monitor.record_timestep()
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(monitor.t, monitor.v.T)
        >>> plt.xlabel('Time (s)')
        >>> plt.ylabel('Voltage')
    """
    
    # Track if we've defined the C++ recording infrastructure
    _cpp_infrastructure_defined = False
    
    @classmethod
    def _ensure_cpp_infrastructure(cls):
        """
        Define C++ classes and functions needed for recording.
        
        This is called lazily on first StateMonitor creation to avoid
        cppyy overhead if monitors aren't used.
        """
        if cls._cpp_infrastructure_defined:
            return
        
        # Define DynamicArray2D if not already defined
        # (might have been defined by DynamicArrayVariable)
        try:
            _ = cppyy.gbl.brian_poc.DynamicArray2D['double']
        except:
            cppyy.cppdef("""
            #include <vector>
            #include <algorithm>
            
            namespace brian_poc {
                
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
                    
                    void resize(size_t new_rows, size_t new_cols) {
                        if (new_rows == num_rows && new_cols == num_cols) return;
                        
                        std::vector<T> new_buffer(new_rows * new_cols, T(0));
                        
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
                    
                    T& operator()(size_t row, size_t col) {
                        return buffer[row * num_cols + col];
                    }
                    
                    T* data() { return buffer.data(); }
                    size_t rows() const { return num_rows; }
                    size_t cols() const { return num_cols; }
                    size_t size() const { return buffer.size(); }
                };
                
                template class DynamicArray2D<double>;
                template class DynamicArray2D<float>;
                template class DynamicArray2D<int>;
            }
            """)
        
        cls._cpp_infrastructure_defined = True
    
    def __init__(
        self,
        source,  # NeuronGroup
        variables: Union[str, List[str]],
        record: Union[bool, List[int], np.ndarray] = True,
        clock: Optional[Clock] = None,
        name: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Create a StateMonitor.
        
        Args:
            source: The NeuronGroup to monitor
            variables: Variable name(s) to record (e.g., 'v' or ['v', 'I'])
            record: Which neuron indices to record:
                    - True: record all neurons
                    - List/array of ints: record specific indices
            clock: Clock for timing. Defaults to source's clock.
            name: Optional name for this monitor
            verbose: If True, print progress messages
        """
        global _statemonitor_counter
        self.id = _statemonitor_counter
        _statemonitor_counter += 1
        
        self.name = name or f"statemonitor_{self.id}"
        self.source = source
        self.verbose = verbose
        
        # Ensure C++ infrastructure is ready
        self._ensure_cpp_infrastructure()
        
        # Normalize variables to a list
        if isinstance(variables, str):
            variables = [variables]
        self.record_variables = variables
        
        # Use source's clock if not specified
        self.clock = clock or source.clock
        
        # Set up recording indices
        if record is True:
            self.record_indices = np.arange(source.n_neurons, dtype=np.int32)
        else:
            self.record_indices = np.asarray(record, dtype=np.int32)
        
        self.n_indices = len(self.record_indices)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Creating StateMonitor: {self.name}")
            print(f"{'='*70}")
            print(f"  Source:    {source.name}")
            print(f"  Variables: {variables}")
            print(f"  Recording: {self.n_indices} neurons")
            if self.n_indices <= 10:
                print(f"  Indices:   {list(self.record_indices)}")
        
        # Initialize storage
        self._init_storage()
        
        # Generate recording code
        self._create_recording_code()
    
    def _init_storage(self):
        """
        Initialize the data storage arrays.
        
        We use:
        - std::vector<double> for time points (1D, grows each timestep)
        - DynamicArray2D for each recorded variable (rows=times, cols=neurons)
        """
        if self.verbose:
            print(f"\n  Initializing storage...")
        
        # Time storage (1D vector)
        self._t_vector = cppyy.gbl.std.vector['double']()
        
        # Storage for each recorded variable
        self._recorded_data: Dict[str, Any] = {}
        
        for varname in self.record_variables:
            # Create DynamicArray2D with 0 rows (timesteps) and n_indices columns
            arr = cppyy.gbl.brian_poc.DynamicArray2D['double'](0, self.n_indices)
            self._recorded_data[varname] = arr
            
            if self.verbose:
                print(f"    {varname}: DynamicArray2D (grows × {self.n_indices})")
        
        # Track number of recorded timesteps
        self._N = 0
    
    def _create_recording_code(self):
        """
        Generate and compile C++ code for recording.
        
        The recording function:
        1. Gets current time from clock
        2. Reads source variable values at recorded indices
        3. Appends to dynamic arrays
        
        We generate a custom function for each StateMonitor to handle
        the specific variables and indices.
        """
        if self.verbose:
            print(f"\n  Generating recording code...")
        
        # Build unique function name
        func_name = f'record_statemonitor_{self.id}'
        
        # Generate C++ code
        # This is a simpler approach than Brian2's template system
        # since we can leverage cppyy's flexibility
        
        var_decls = []
        var_records = []
        
        for varname in self.record_variables:
            var_decls.append(f"double* source_{varname}")
            var_records.append(f"""
        // Record {varname}
        for (int i = 0; i < n_indices; ++i) {{
            int src_idx = indices[i];
            recorded_{varname}(new_row, i) = source_{varname}[src_idx];
        }}""")
        
        code = f"""
        #include <vector>
        
        namespace brian_poc {{
            
            void {func_name}(
                std::vector<double>& t_vec,
                double current_time,
                int* indices,
                int n_indices,
                {', '.join(var_decls)},
                {', '.join(f'DynamicArray2D<double>& recorded_{v}' for v in self.record_variables)}
            ) {{
                // Append current time
                t_vec.push_back(current_time);
                
                // Get new row index
                size_t new_row = t_vec.size() - 1;
                
                // Resize recorded arrays to accommodate new row
                {chr(10).join(f'recorded_{v}.resize(new_row + 1, n_indices);' for v in self.record_variables)}
                
                // Record values at monitored indices
                {''.join(var_records)}
            }}
        }}
        """
        
        # Compile the recording function
        try:
            cppyy.cppdef(code)
            self._record_func = getattr(cppyy.gbl.brian_poc, func_name)
            
            if self.verbose:
                print(f"    ✓ Compiled {func_name}")
                
        except Exception as e:
            print(f"    ✗ Compilation failed: {e}")
            print(f"\nGenerated code:\n{code}")
            raise
    
    def record_timestep(self):
        """
        Record the current state of monitored variables.
        
        Call this once per timestep after updating the source group.
        The current time and variable values will be appended to storage.
        
        PERFORMANCE NOTE:
        The recording is done in C++ for efficiency. The only Python
        overhead is the function call itself.
        """
        # Get current time
        current_time = self.clock.t
        
        # Build argument list for recording function
        args = [
            self._t_vector,
            current_time,
            self.record_indices,
            self.n_indices,
        ]
        
        # Add source arrays for each recorded variable
        for varname in self.record_variables:
            source_array = getattr(self.source, varname)
            args.append(source_array)
        
        # Add recorded data arrays
        for varname in self.record_variables:
            args.append(self._recorded_data[varname])
        
        # Call the C++ recording function
        self._record_func(*args)
        
        self._N += 1
    
    @property
    def t(self) -> np.ndarray:
        """
        Array of recorded times.
        
        Returns:
            1D NumPy array of time values in seconds
        """
        # Convert std::vector to NumPy array
        n = len(self._t_vector)
        if n == 0:
            return np.array([], dtype=np.float64)
        
        # Create array from vector data
        arr = np.zeros(n, dtype=np.float64)
        for i in range(n):
            arr[i] = self._t_vector[i]
        return arr
    
    @property
    def N(self) -> int:
        """Number of recorded timesteps."""
        return self._N
    
    def __getattr__(self, name: str) -> np.ndarray:
        """
        Access recorded variable data by name.
        
        This allows natural access like: monitor.v, monitor.I
        
        Returns:
            2D NumPy array with shape (n_timesteps, n_recorded_neurons)
        """
        if name.startswith('_') or name in ('source', 'record_variables', 
                                             'record_indices', 'clock', 'name',
                                             'id', 'n_indices', 'verbose'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        
        if name in self.record_variables:
            return self._get_recorded_array(name)
        
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Recorded variables: {self.record_variables}"
        )
    
    def _get_recorded_array(self, varname: str) -> np.ndarray:
        """
        Extract recorded data as a NumPy array.
        
        Args:
            varname: Name of the recorded variable
            
        Returns:
            2D array with shape (n_timesteps, n_recorded_neurons)
        """
        cpp_array = self._recorded_data[varname]
        rows = cpp_array.rows()
        cols = cpp_array.cols()
        
        if rows == 0:
            return np.array([]).reshape(0, cols)
        
        # Copy data from C++ to NumPy
        # (we copy because the C++ array might resize and invalidate pointers)
        result = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = cpp_array(i, j)
        
        return result
    
    def __repr__(self) -> str:
        return (
            f"StateMonitor({self.name}, recording {self.record_variables} "
            f"from {self.source.name}, {self._N} timesteps)"
        )