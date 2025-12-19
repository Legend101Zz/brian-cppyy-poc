"""
NeuronGroup: A collection of neurons with a differential equation.

This is a simplified version of Brian2's NeuronGroup. It shows how
a high-level user interface connects to the low-level code generation
and compilation system.
"""

from pathlib import Path

import numpy as np
from cppyy_poc.codegen import CodeGenerator
from cppyy_poc.codeobject import CppyyCodeObject
from cppyy_poc.variables import ArrayVariable, Constant, VariableResolver

# Global counter to ensure unique function names
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
    """
    
    def __init__(self, n_neurons: int, equation: str, dt: float = 0.0001, tau: float = 0.01):
        """
        Create a neuron group.
        
        Args:
            n_neurons: Number of neurons to simulate
            equation: Differential equation (e.g., "dv/dt = (-v + I) / tau")
            dt: Integration timestep in seconds
            tau: Time constant in seconds
        """
        # Assign a unique ID to this neuron group
        # This ensures that if we create multiple neuron groups in the same
        # Python session, each one gets its own uniquely-named C++ function
        # This is critical because cppyy maintains a single global C++ namespace
        global _neurongroup_counter
        self.id = _neurongroup_counter
        _neurongroup_counter += 1
        
        self.n_neurons = n_neurons
        self.equation = equation
        self.dt = dt
        self.tau = tau
        
        print(f"\n{'='*70}")
        print(f"Creating NeuronGroup (ID: {self.id})")
        print(f"{'='*70}")
        print(f"Number of neurons: {n_neurons}")
        print(f"Equation: {equation}")
        print(f"Timestep dt: {dt}")
        print(f"Time constant tau: {tau}")
        
        # Create state arrays
        # In real Brian2, array initialization depends on the units
        # and user-specified initial values
        print(f"\nInitializing state arrays...")
        self.v = np.zeros(n_neurons, dtype=np.float64)  # Voltage
        self.I = np.random.randn(n_neurons) * 0.1      # Input current
        print(f"   v: voltage array, shape={self.v.shape}")
        print(f"   I: current array, shape={self.I.shape}")
        
        # Set up variables
        # These Variable objects track what data exists and how
        # to access it
        print(f"\nCreating Variable objects...")
        self.variables = VariableResolver()
        self.variables.add(ArrayVariable('v', self.v))
        self.variables.add(ArrayVariable('I', self.I))
        self.variables.add(Constant('dt', dt))
        self.variables.add(Constant('tau', tau))
        print(f"   Variables: {self.variables}")
        
        # Generate and compile update code
        self._create_update_code()
    
    def _create_update_code(self):
        """
        Generate and compile the C++ code for updating neurons.
        
        This is where we go from user equation to compiled machine code:
        1. Parse equation (simplified here)
        2. Generate C++ code from template
        3. Create CodeObject
        4. CodeObject compiles with cppyy
        """
        print(f"\nCODE GENERATION PHASE")
        print(f"{'='*70}")
        
        # In real Brian2, there's a sophisticated equation parser
        # Here we just hardcode the right-hand side
        # For "dv/dt = (-v + I) / tau", the RHS is "(-v + I) / tau"
        equation_rhs = "(-v_current + I_current) / tau"
        
        # Get template directory
        template_dir = Path(__file__).parent / 'templates'
        
        # Create code generator
        codegen = CodeGenerator(template_dir)
        print(f"   Code generator: {codegen}")
        
        # Generate a unique function name for this neuron group
        # This is crucial: each neuron group must have its own function name
        # to avoid conflicts in cppyy's global C++ namespace
        function_name = f'update_neurons_{self.id}'
        print(f"   Function name: {function_name}")
        
        # Generate C++ code
        print(f"\n   Generating C++ code from template...")
        cpp_code = codegen.generate_neuron_update(
            function_name=function_name,
            variables=self.variables,
            equation=self.equation,
            equation_rhs=equation_rhs
        )
        print(f"   Generated {len(cpp_code)} characters of C++ code")
        
        # Create CodeObject (this compiles the code)
        self.code_object = CppyyCodeObject(
            name=f'neurongroup_{self.id}_stateupdater',
            code=cpp_code,
            function_name=function_name,
            variables=self.variables.get_all()
        )
        
        # Prepare namespace
        self.code_object.prepare_namespace()
    
    def update(self):
        """
        Update neuron states for one timestep.
        
        This calls the compiled C++ function, which operates
        directly on our NumPy arrays at native speed.
        """
        self.code_object()
    
    def show_code(self):
        """Display the generated C++ code."""
        self.code_object.show_generated_code()