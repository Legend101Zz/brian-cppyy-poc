"""
Brian2 cppyy Proof of Concept.

A minimal demonstration of how cppyy could replace Cython in Brian2.
"""

from .clock import Clock
from .codegen import CodeGenerator
from .codeobject import CppyyCodeObject
from .inspector import CppyyInspector
from .network import Network
from .neurongroup import NeuronGroup
from .statemonitor import StateMonitor
from .variables import (ArrayVariable, Constant, DynamicArrayVariable,
                        Variable, VariableResolver)

__version__ = "0.2.0"

__all__ = [
    # Core simulation objects
    'NeuronGroup',
    'StateMonitor',
    'Network',
    'Clock',
    # Variable system
    'Variable',
    'Constant',
    'ArrayVariable',
    'DynamicArrayVariable',
    'VariableResolver',
    # Code generation
    'CodeGenerator',
    'CppyyCodeObject',
    # Debugging/Introspection
    'CppyyInspector',
]