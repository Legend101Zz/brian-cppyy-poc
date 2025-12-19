"""
Code generator that produces C++ from templates.

This is the heart of the system. It takes:
1. User equations (like "dv/dt = (-v + I) / tau")
2. Variable information
3. A Jinja2 template

And produces:
- Clean C++ code that cppyy can compile

The generated code is just a string - it doesn't touch the filesystem.
Everything happens in memory.

We use Jinja2, the same template engine Brian2 uses. Templates are
.cpp.j2 files that contain C++ code with Jinja2 placeholders.
"""

from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from .variables import (ArrayVariable, Constant, DynamicArrayVariable,
                        VariableResolver)


class CodeGenerator:
    """
    Generates C++ code from templates and equations.
    
    This mimics Brian2's code generation system but simplified.
    The real system has multiple generators for different targets
    (Cython, C++, etc.) and handles complex equation parsing.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the code generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / 'templates'        
        # Set up Jinja2 environment
        # This is the template rendering engine
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,           # Remove newlines after blocks
            lstrip_blocks=True,         # Remove leading whitespace
        )
        
        # Register custom filters for type conversion
        self.env.filters['cpp_type'] = self._get_cpp_type
    
    def _get_cpp_type(self, dtype) -> str:
        """
        Jinja2 filter to convert Python/NumPy types to C++.
        
        Usage in template: {{ var.dtype | cpp_type }}
        """
        import numpy as np
        type_map = {
            float: 'double',
            int: 'int',
            np.float64: 'double',
            np.float32: 'float',
            np.int32: 'int32_t',
            np.int64: 'int64_t',
            bool: 'bool',
        }
        return type_map.get(dtype, 'double')

    def generate_neuron_update(
        self,
        function_name: str,
        variables: VariableResolver,
        equation: str,
        equation_rhs: str,
        namespace_name: Optional[str] = None
    ) -> str:
        """
        Generate C++ code for updating neuron states.
    
        1. We load the template
        2. We extract information from variables
        3. We render the template with that information
        4. We get back a C++ code string
        
        Args:
            function_name: Name for the C++ function
            variables: All variables (arrays, constants)
            equation: The differential equation (for documentation)
            equation_rhs: Right-hand side of equation (the computation)
            namespace_name: Optional C++ namespace to wrap the code in
        
        Returns:
            Complete C++ code as a string
        """
        # Load the template
        template = self.env.get_template('neuron_update.cpp.j2')
        
        # Build parameter list for the C++ function
        # Order: arrays first, then constants, then n_neurons
        parameters = []
        
        # Add array parameters (as pointers)
        for name, var in variables.get_all().items():
            if isinstance(var, ArrayVariable):
                cpp_type = var.get_cpp_type()
                parameters.append(f"{cpp_type}* {name}")
        
        # Add constant parameters
        for name, var in variables.get_all().items():
            if isinstance(var, Constant):
                cpp_type = var.get_cpp_type()
                parameters.append(f"{cpp_type} {name}")
        
        # Add neuron count
        parameters.append("int n_neurons")
        
        # Render the template
        code = template.render(
            function_name=function_name,
            parameters=parameters,
            equation=equation,
            equation_rhs=equation_rhs,
            loop_variable='_idx',
            namespace=namespace_name,
        )
        
        return code

    def generate_statemonitor_record(
        self,
        function_name: str,
        recorded_variables: List[str],
        source_variables: Dict[str, str],  # name -> cpp_type
        namespace_name: Optional[str] = None
    ) -> str:
        """
        Generate C++ code for StateMonitor recording.
        
        The StateMonitor needs to:
        1. Record the current time
        2. For each recorded variable, copy values for monitored indices
        3. Grow the storage arrays as needed
        
        This is more complex than neuron update because it uses
        dynamic arrays that grow over time.
        
        Args:
            function_name: Name for the generated C++ function
            recorded_variables: List of variable names to record
            source_variables: Map of variable names to their C++ types
            namespace_name: Optional C++ namespace
            
        Returns:
            Complete C++ code as a string
        """
        template = self.env.get_template('statemonitor.cpp.j2')
        
        code = template.render(
            function_name=function_name,
            recorded_variables=recorded_variables,
            source_variables=source_variables,
            namespace=namespace_name,
        )
        
        return code
    
    def generate_dynamic_array_helpers(self, namespace_name: str = "brian_poc") -> str:
        """
        Generate C++ helper classes for dynamic arrays.
        
        These are needed for StateMonitor and other components that
        need to store data that grows during simulation.
        
        Args:
            namespace_name: C++ namespace to put helpers in
            
        Returns:
            C++ code defining DynamicArray1D and DynamicArray2D
        """
        template = self.env.get_template('dynamic_array.h.j2')
        return template.render(namespace=namespace_name)    
    
    def __repr__(self):
        return f"CodeGenerator(templates={self.env.loader.searchpath})"    