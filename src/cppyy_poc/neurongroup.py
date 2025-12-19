"""
NeuronGroup: A collection of neurons with a differential equation.

This is a simplified version of Brian2's NeuronGroup. It shows how
a high-level user interface connects to the low-level code generation
and compilation system.

KEY CONCEPTS:
1. State variables (v, I) are NumPy arrays
2. Equations define how state evolves
3. Code generation creates C++ for the update
4. cppyy compiles and executes the C++

COMPARISON TO BRIAN2:
- Brian2 parses equations into an abstract representation
- We hardcode the equation parsing for simplicity
- Brian2 supports multiple numerical methods
- We use simple Euler integration
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .clock import Clock
from .codegen import CodeGenerator
from .codeobject import CppyyCodeObject
from .variables import ArrayVariable, Constant, VariableResolver

# Global counter to ensure unique function names across all NeuronGroups
_neurongroup_counter = 0


class NeuronGroup:
    """
    A group of neurons with state variables and a differential equation.
    
    This is the user-facing interface. Users create a NeuronGroup,
    specify an equation, and the system automatically:
    1. Creates state arrays
    2. Generates C++ code
    3. Compiles with cppyy
    4. Provides update() method to evolve the system
    
    The generated C++ code operates directly on the NumPy arrays,
    achieving near-native performance without any data copying.
    
    Attributes:
        n_neurons: Number of neurons in this group
        equation: The differential equation string
        dt: Integration timestep
        tau: Time constant (for the default equation)
        v: Voltage array (state variable)
        I: Input current array
        clock: The Clock governing this group's updates
        
    Example:
        >>> neurons = NeuronGroup(
        ...     n_neurons=100,
        ...     equation="dv/dt = (-v + I) / tau",
        ...     dt=0.0001,
        ...     tau=0.01
        ... )
        >>> neurons.v[:] = np.random.randn(100) * 0.1
        >>> for _ in range(1000):
        ...     neurons.update()
        >>> print(f"Mean voltage: {neurons.v.mean():.4f}")
    """
    
    def __init__(
        self,
        n_neurons: int,
        equation: str,
        dt: float = 0.0001,
        tau: float = 0.01,
        clock: Optional[Clock] = None,
        namespace: Optional[str] = None,
        name: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Create a neuron group.
        
        Args:
            n_neurons: Number of neurons to simulate
            equation: Differential equation (e.g., "dv/dt = (-v + I) / tau")
            dt: Integration timestep in seconds (default: 0.1ms)
            tau: Time constant in seconds (default: 10ms)
            clock: Clock for scheduling. If None, creates a new one.
            namespace: Optional C++ namespace for isolation
            name: Optional name for this group
            verbose: If True, print progress during setup
        """
        # Assign unique ID
        global _neurongroup_counter
        self.id = _neurongroup_counter
        _neurongroup_counter += 1
        
        self.name = name or f"neurongroup_{self.id}"
        self.n_neurons = n_neurons
        self.equation = equation
        self.dt = dt
        self.tau = tau
        self.namespace = namespace
        self.verbose = verbose
        
        # Create or use provided clock
        self.clock = clock or Clock(dt=dt)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Creating NeuronGroup: {self.name}")
            print(f"{'='*70}")
            print(f"  Neurons:  {n_neurons}")
            print(f"  Equation: {equation}")
            print(f"  dt:       {dt*1000:.3f} ms")
            print(f"  tau:      {tau*1000:.1f} ms")
        
        # Create state arrays
        self._init_state_arrays()
        
        # Set up variable tracking
        self._init_variables()
        
        # Generate and compile update code
        self._create_update_code()
    
    def _init_state_arrays(self):
        """
        Initialize the state variable arrays.
        
        In Brian2, array initialization can be complex, handling
        units, user-specified values, etc. Here we simplify:
        - v (voltage): initialized to 0
        - I (current): random noise for interesting dynamics
        """
        if self.verbose:
            print(f"\n  Initializing state arrays...")
        
        # Voltage - main state variable
        self.v = np.zeros(self.n_neurons, dtype=np.float64)
        
        # Input current - randomized for variety
        self.I = np.random.randn(self.n_neurons).astype(np.float64) * 0.1
        
        if self.verbose:
            print(f"    v: shape={self.v.shape}, dtype={self.v.dtype}")
            print(f"    I: shape={self.I.shape}, dtype={self.I.dtype}")
    
    def _init_variables(self):
        """
        Create Variable objects to track our state.
        
        The VariableResolver collects all variables that the
        code generator needs to know about.
        """
        if self.verbose:
            print(f"\n  Creating Variable objects...")
        
        self.variables = VariableResolver()
        self.variables.add(ArrayVariable('v', self.v))
        self.variables.add(ArrayVariable('I', self.I))
        self.variables.add(Constant('dt', self.dt))
        self.variables.add(Constant('tau', self.tau))
        
        if self.verbose:
            print(f"    Registered: {list(self.variables.get_all().keys())}")
    
    def _create_update_code(self):
        """
        Generate and compile the C++ code for updating neurons.
        
        This is where we go from user equation to compiled machine code:
        1. Parse equation (simplified here)
        2. Generate C++ code from template
        3. Create CodeObject
        4. CodeObject compiles with cppyy
        """
        if self.verbose:
            print(f"\n  CODE GENERATION PHASE")
        
        # In real Brian2, there's a sophisticated equation parser
        # Here we just hardcode the right-hand side
        # For "dv/dt = (-v + I) / tau", the RHS is "(-v + I) / tau"
        equation_rhs = "(-v_current + I_current) / tau"
        
        # Get template directory
        template_dir = Path(__file__).parent / 'templates'
        
        # Create code generator
        codegen = CodeGenerator(template_dir)
        
        # Generate unique function name
        function_name = f'update_neurons_{self.id}'
        if self.namespace:
            # If using namespace, we'll access via namespace::function
            full_function_name = f'{self.namespace}_{function_name}'
        else:
            full_function_name = function_name
        
        if self.verbose:
            print(f"    Function name: {full_function_name}")
            print(f"    Generating C++ code from template...")
        
        # Generate C++ code
        cpp_code = codegen.generate_neuron_update(
            function_name=full_function_name,
            variables=self.variables,
            equation=self.equation,
            equation_rhs=equation_rhs,
            namespace_name=self.namespace
        )
        
        if self.verbose:
            print(f"    Generated {len(cpp_code)} characters of C++ code")
        
        # Create CodeObject (this compiles the code)
        self.code_object = CppyyCodeObject(
            name=f'{self.name}_stateupdater',
            code=cpp_code,
            function_name=full_function_name,
            variables=self.variables.get_all(),
            verbose=self.verbose
        )
        
        # Prepare namespace
        self.code_object.prepare_namespace()
    
    def update(self):
        """
        Update neuron states for one timestep.
        
        This calls the compiled C++ function, which operates
        directly on our NumPy arrays at native speed.
        
        After this call:
        - self.v contains the new voltages
        - self.clock has advanced by dt
        """
        self.code_object()
        self.clock.tick()
    
    def run(self, duration: float):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Simulation time in seconds
            
        Returns:
            Number of timesteps executed
        """
        n_steps = int(duration / self.dt)
        
        if self.verbose:
            print(f"\n  Running for {duration*1000:.1f}ms ({n_steps} steps)...")
        
        for _ in range(n_steps):
            self.update()
        
        if self.verbose:
            print(f"  Simulation complete. Final t={self.clock.t*1000:.1f}ms")
        
        return n_steps
    
    def show_code(self):
        """Display the generated C++ code."""
        self.code_object.show_generated_code()
    
    @property
    def t(self) -> float:
        """Current simulation time in seconds."""
        return self.clock.t
    
    def __repr__(self) -> str:
        return (
            f"NeuronGroup({self.name}, n={self.n_neurons}, "
            f"t={self.t*1000:.1f}ms)"
        )