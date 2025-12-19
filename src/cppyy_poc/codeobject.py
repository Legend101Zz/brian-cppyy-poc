"""
CodeObject: The bridge between Python and JIT-compiled C++.

This is the core of the cppyy approach. A CodeObject:
1. Takes generated C++ code as a string
2. Compiles it using cppyy (JIT, in memory, no files!)
3. Provides a callable interface to execute the compiled code
4. Manages the namespace (arguments to pass to C++)

Compare this to Brian2's Cython approach:
- Cython: Write .pyx → compile to .cpp → compile to .so → load → execute
- cppyy: String → JIT compile → execute (all in memory!)
"""

from typing import Any, Dict, List

import cppyy
import numpy as np

from .variables import ArrayVariable, Constant, Variable


class CppyyCodeObject:
    """
    A CodeObject represents a compiled piece of code ready to execute.
    
    Lifecycle:
    1. __init__: Receive C++ code string and variable info
    2. compile(): Feed code to cppyy (parsing, AST building)
    3. prepare_namespace(): Get NumPy arrays and constant values
    4. __call__: Execute the compiled function (JIT compile on first call)
    
    Memory state evolution:
    - After __init__: C++ code exists as Python string
    - After compile(): AST exists in Cling, symbol registered
    - After first __call__: Machine code exists in executable memory
    - Subsequent __call__s: Direct machine code execution
    """
    
    def __init__(
        self,
        name: str,
        code: str,
        function_name: str,
        variables: Dict[str, Variable]
    ):
        """
        Create a CodeObject.
        
        Args:
            name: Human-readable name (for debugging)
            code: The complete C++ code as a string
            function_name: Name of the C++ function to call
            variables: Dictionary of Variable objects
        """
        self.name = name
        self.code = code
        self.function_name = function_name
        self.variables = variables
        
        # This will hold the cppyy function proxy after compilation
        self.compiled_function = None
        
        # This will hold the arguments to pass to the function
        self.namespace = {}
        
        # Compile immediately
        print(f"\n{'='*70}")
        print(f"Creating CodeObject: {name}")
        print(f"{'='*70}")
        self.compile()
    
    def compile(self):
        """
        Compile the C++ code using cppyy.
        
        INTERNAL FLOW:
        1. Python string → CPyCppyy bridge
        2. CPyCppyy → Cling interpreter
        3. Cling → Clang parser
        4. Clang builds Abstract Syntax Tree (AST)
        5. Function registered in symbol table
        6. NO machine code yet (lazy compilation!)
        
        MEMORY STATE AFTER THIS:
        - Python heap: Still has the code string
        - Cling memory: Has AST representation
        - Symbol table: Has function metadata
        - Executable memory: Empty (compilation happens on first call)
        """
        print(f"\nCOMPILATION PHASE")
        print(f"   Feeding C++ code to cppyy...")
        print(f"   Code length: {len(self.code)} characters")
        
        try:
            # This is THE critical call!
            # Internally, cppyy does:
            # 1. Converts Python string to C++ std::string
            # 2. Passes to Cling's parser
            # 3. Clang lexes and parses the code
            # 4. AST nodes allocated in Cling's memory
            # 5. Semantic analysis validates types
            # 6. Symbol registered (but not compiled to machine code)
            
            cppyy.cppdef(self.code)
            
            print(f"Parsing complete")
            print(f"AST built in Cling's memory")
            print(f"Function '{self.function_name}' registered")
            print(f"Machine code NOT generated yet (lazy!)")
            
        except Exception as e:
            print(f"\nCOMPILATION FAILED!")
            print(f"   Error: {e}")
            print(f"\n   Generated code:")
            print(f"   {'-'*70}")
            for i, line in enumerate(self.code.split('\n'), 1):
                print(f"   {i:3d} | {line}")
            print(f"   {'-'*70}")
            raise
        
        # Get a reference to the function
        # This creates a Python proxy object
        print(f"\n Getting function reference...")
        self.compiled_function = getattr(cppyy.gbl, self.function_name)
        
        print(f" Proxy object created: {type(self.compiled_function)}")
        print(f" Function signature: {self.compiled_function.__doc__}")
        
        # At this point in memory:
        # - Python has a CPPOverload proxy object
        # - The proxy has a NULL function pointer
        # - Cling has the AST ready to compile when needed
    
    def prepare_namespace(self):
        """
        Build the namespace (arguments for the C++ function).
        
        Unlike Cython which passes a dictionary, cppyy functions
        take explicit parameters. So "namespace" here means
        "the values we'll pass as arguments".
        
        For arrays: Get the NumPy array (cppyy extracts pointer automatically)
        For constants: Get the numeric value
        """
        print(f"\nNAMESPACE PREPARATION")
        print(f"   Building argument list...")
        
        self.namespace = {}
        
        for name, var in self.variables.items():
            if isinstance(var, ArrayVariable):
                # Get the NumPy array
                # cppyy will automatically extract the data pointer
                array = var.get_value()
                self.namespace[name] = array
                print(f"   • {name}: array shape={array.shape}, "
                      f"ptr={array.ctypes.data:#x}")
                
            elif isinstance(var, Constant):
                # Get the constant value
                value = var.get_value()
                self.namespace[name] = value
                print(f"   • {name}: constant value={value}")
        
        # Add neuron count (derived from array size)
        if 'v' in self.namespace:
            n_neurons = len(self.namespace['v'])
            self.namespace['n_neurons'] = n_neurons
            print(f"   • n_neurons: {n_neurons}")
    
    def __call__(self):
        """
        Execute the compiled C++ function.
        
        ON FIRST CALL:
        1. Argument conversion (Python → C++)
        2. cppyy checks: is function compiled? NO!
        3. Lazy compilation triggers:
           - AST → LLVM IR generation
           - LLVM optimization passes
           - Machine code generation
           - Store function pointer
        4. Execute machine code
        5. Return to Python
        
        ON SUBSEQUENT CALLS:
        1. Argument conversion (Python → C++)
        2. cppyy checks: is function compiled? YES!
        3. Skip to step 4: Execute machine code
        4. Return to Python
        
        PERFORMANCE:
        - First call: ~0.05-0.5 seconds (includes compilation)
        - Later calls: ~0.0001-0.001 seconds (pure execution)
        """
        print(f"\n EXECUTION PHASE")
        
        # Build argument list in correct order
        # Must match the C++ function signature!
        args = []
        
        # The order is: arrays, then constants, then n_neurons
        for name, var in self.variables.items():
            if isinstance(var, ArrayVariable):
                args.append(self.namespace[name])
        
        for name, var in self.variables.items():
            if isinstance(var, Constant):
                args.append(self.namespace[name])
        
        args.append(self.namespace['n_neurons'])
        
        print(f"   Calling C++ function with {len(args)} arguments...")
        
        # THIS IS THE MAGIC MOMENT!
        # On first call: triggers JIT compilation
        # On later calls: directly executes machine code
        
        import time
        start = time.time()
        
        result = self.compiled_function(*args)
        
        elapsed = time.time() - start
        
        print(f"Execution complete in {elapsed*1000:.4f} ms")
        
        if not hasattr(self, '_first_call_done'):
            print(f"   (First call: included JIT compilation time)")
            self._first_call_done = True
        else:
            print(f"   (Subsequent call: pure execution time)")
        
        return result
    
    def show_generated_code(self):
        """Display the generated C++ code with line numbers."""
        print(f"\n{'='*70}")
        print(f"GENERATED C++ CODE")
        print(f"{'='*70}")
        for i, line in enumerate(self.code.split('\n'), 1):
            print(f"{i:3d} | {line}")
        print(f"{'='*70}")            
        print(f"{i:3d} | {line}")
        print(f"{'='*70}")