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
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .variables import ArrayVariable, Constant, VariableResolver


class CodeGenerator:
    """
    Generates C++ code from templates and equations.
    
    This mimics Brian2's code generation system but simplified.
    The real system has multiple generators for different targets
    (Cython, C++, etc.) and handles complex equation parsing.
    """
    
    def __init__(self, template_dir: Path):
        """
        Initialize the code generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        # Set up Jinja2 environment
        # This is the template rendering engine
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,           # Remove newlines after blocks
            lstrip_blocks=True,         # Remove leading whitespace
        )
    
    def generate_neuron_update(
        self,
        function_name: str,
        variables: VariableResolver,
        equation: str,
        equation_rhs: str
    ) -> str:
        """
        Generate C++ code for updating neuron states.
        
        This is where the magic happens:
        1. We load the template
        2. We extract information from variables
        3. We render the template with that information
        4. We get back a C++ code string
        
        Args:
            function_name: Name for the C++ function
            variables: All variables (arrays, constants)
            equation: The differential equation (for documentation)
            equation_rhs: Right-hand side of equation (the computation)
        
        Returns:
            Complete C++ code as a string
        """
        # Load the template
        template = self.env.get_template('neuron_update.cpp.j2')
        
        # Build the parameter list for the C++ function
        # This is critical: we need to pass arrays as pointers
        # and constants as values
        parameters = []
        
        for var_name, var in variables.get_all().items():
            if isinstance(var, ArrayVariable):
                # Arrays become: "double* v, int v_size"
                cpp_type = var.get_cpp_type()
                parameters.append(f"{cpp_type}* {var_name}")
                # We don't explicitly pass size separately in this simple version
                # In real Brian2, array sizes are parameters too
                
            elif isinstance(var, Constant):
                # Constants become: "double dt"
                cpp_type = var.get_cpp_type()
                parameters.append(f"{cpp_type} {var_name}")
        
        # Add the neuron count parameter
        parameters.append("int n_neurons")
        
        # Render the template
        # Jinja2 will replace all {{ variable }} placeholders
        # with the values we provide
        cpp_code = template.render(
            function_name=function_name,
            parameters=parameters,
            equation=equation,
            equation_rhs=equation_rhs,
            loop_variable='i',  # Loop index variable name
        )
        
        return cpp_code
    
    def __repr__(self):
        return f"CodeGenerator(templates={self.env.loader.searchpath})"    