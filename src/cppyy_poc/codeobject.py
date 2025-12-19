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

from typing import Any, Callable, Dict, List, Optional

import cppyy
import numpy as np

from .variables import ArrayVariable, Constant, DynamicArrayVariable, Variable


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
    
    Attributes:
    name: Human-readable name for debugging
    code: The C++ source code
    function_name: Name of the C++ function
    variables: Dictionary of Variable objects
    compiled_function: The cppyy function proxy after compilation
    namespace: Arguments to pass to the function
    """
    def __init__(
        self,
        name: str,
        code: str,
        function_name: str,
        variables: Dict[str, Variable],
        verbose: bool = True
    ):
        """
        Create a CodeObject.
        
        Args:
            name: Human-readable name (for debugging)
            code: The complete C++ code as a string
            function_name: Name of the C++ function to call
            variables: Dictionary of Variable objects
            verbose: If True, print compilation progress
        """
        self.name = name
        self.code = code
        self.function_name = function_name
        self.variables = variables
        self.verbose = verbose
        
        # These will be set during compilation
        self.compiled_function: Optional[Callable] = None
        self.namespace: Dict[str, Any] = {}
        
        # Track execution count for debugging
        self._execution_count = 0
        self._first_call_done = False
        
        # Compile immediately
        if self.verbose:
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
        if self.verbose:
            print(f"\n   COMPILATION PHASE")
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
            
            if self.verbose:
                print(f"Parsing complete")
                print(f"AST built in Cling's memory")
                print(f"Function '{self.function_name}' registered")
                print(f"Machine code NOT generated yet (lazy!)")
            
        except Exception as e:
            print(f"\n COMPILATION FAILED!")
            print(f"   Error: {e}")
            print(f"\n   Generated code:")
            print(f"   {'-'*70}")
            for i, line in enumerate(self.code.split('\n'), 1):
                print(f"   {i:3d} | {line}")
            print(f"   {'-'*70}")
            raise
        
        # Get a reference to the function
        # This creates a Python proxy object
        if self.verbose:
            print(f"\n   Getting function reference...")
        
        self.compiled_function = getattr(cppyy.gbl, self.function_name)
        
        if self.verbose:
            print(f"Proxy object created: {type(self.compiled_function).__name__}")
            if self.compiled_function.__doc__:
                sig = self.compiled_function.__doc__.split('\n')[0]
                print(f"Signature: {sig}")
    
    def prepare_namespace(self):
        """
        Build the namespace (arguments for the C++ function).
        
        Unlike Cython which passes a dictionary, cppyy functions
        take explicit parameters. So "namespace" here means
        "the values we'll pass as arguments".
        
        For arrays: Get the NumPy array (cppyy extracts pointer automatically)
        For constants: Get the numeric value
        """
        if self.verbose:
            print(f"\n   NAMESPACE PREPARATION")
            print(f"   Building argument list...")
        
        self.namespace = {}
        
        for name, var in self.variables.items():
            if isinstance(var, ArrayVariable):
                # Get the NumPy array
                # cppyy will automatically extract the data pointer
                array = var.get_value()
                self.namespace[name] = array
                if self.verbose:
                    print(f"   • {name}: array shape={array.shape}, "
                          f"ptr={array.ctypes.data:#x}")
                
            elif isinstance(var, Constant):
                # Get the constant value
                value = var.get_value()
                self.namespace[name] = value
                if self.verbose:
                    print(f"   • {name}: constant value={value}")
        
        # Add neuron count (derived from array size)
        if 'v' in self.namespace:
            n_neurons = len(self.namespace['v'])
            self.namespace['n_neurons'] = n_neurons
            if self.verbose:
                print(f"   • n_neurons: {n_neurons}")
    
    def _build_args(self) -> List[Any]:
        """
        Build the argument list in the correct order for the C++ function.
        
        The order must match the function signature exactly:
        1. Array pointers (in variable order)
        2. Constants (in variable order)
        3. n_neurons
        
        Returns:
            List of arguments ready to pass to the C++ function
        """
        args = []
        
        # Arrays first
        for name, var in self.variables.items():
            if isinstance(var, ArrayVariable):
                args.append(self.namespace[name])
        
        # Then constants
        for name, var in self.variables.items():
            if isinstance(var, Constant):
                args.append(self.namespace[name])
        
        # Finally n_neurons
        if 'n_neurons' in self.namespace:
            args.append(self.namespace['n_neurons'])
        
        return args
    
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
        - First call: ~10-100ms (includes compilation)
        - Later calls: ~0.01-0.1ms (pure execution)
        """
        self._execution_count += 1
        
        args = self._build_args()
        
        if self.verbose and not self._first_call_done:
            print(f"\n   EXECUTION #{self._execution_count} (FIRST - will JIT compile)")
            print(f"   Calling {self.function_name} with {len(args)} arguments...")
        
        # Execute the function
        # On first call, this triggers JIT compilation
        result = self.compiled_function(*args)
        
        if not self._first_call_done:
            self._first_call_done = True
            if self.verbose:
                print(f"   JIT compilation complete")
                print(f"   Machine code now cached")
        
        return result
    
    def show_generated_code(self):
        """Display the generated C++ code with line numbers."""
        print(f"\n{'='*70}")
        print(f"Generated C++ code for: {self.name}")
        print(f"{'='*70}")
        for i, line in enumerate(self.code.split('\n'), 1):
            print(f"{i:3d} | {line}")
        print(f"{'='*70}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this CodeObject."""
        return {
            'name': self.name,
            'function_name': self.function_name,
            'code_length': len(self.code),
            'execution_count': self._execution_count,
            'is_compiled': self._first_call_done,
            'num_variables': len(self.variables),
        }