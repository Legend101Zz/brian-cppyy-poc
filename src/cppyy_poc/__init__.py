"""
Brian2 cppyy Proof of Concept.

A minimal demonstration of how cppyy could replace Cython in Brian2.
"""

from .codegen import CodeGenerator
from .codeobject import CppyyCodeObject
from .neurongroup import NeuronGroup
from .variables import ArrayVariable, Constant, Variable, VariableResolver

__version__ = "0.1.0"

__all__ = [
    'NeuronGroup',
    'Variable',
    'Constant',
    'ArrayVariable',
    'VariableResolver',
    'CodeGenerator',
    'CppyyCodeObject',
]