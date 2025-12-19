"""
CppyyInspector: Deep introspection of cppyy objects and memory.

This module provides tools to inspect:
1. How cppyy represents C++ objects in Python
2. Memory addresses and layouts
3. Compilation states (AST vs machine code)
4. Namespace contents and conflicts
5. Function overload resolution

This is invaluable for:
- Debugging compilation issues
- Understanding cppyy's internal representation
- Verifying zero-copy data sharing
- Detecting namespace collisions
"""

import gc
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import cppyy
import numpy as np

# Optional: rich for beautiful console output
try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Optional: psutil for memory stats
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class CppyyObjectInfo:
    """
    Detailed information about a cppyy-bound object.
    
    This captures everything we can learn about how cppyy
    represents a C++ object in Python.
    """
    name: str
    cpp_type: str
    python_type: str
    address: Optional[int] = None
    size_bytes: Optional[int] = None
    is_compiled: bool = False
    is_proxy: bool = False
    overloads: List[str] = field(default_factory=list)
    members: Dict[str, str] = field(default_factory=dict)
    docstring: Optional[str] = None
    
    def summary(self) -> str:
        """One-line summary of the object."""
        status = "compiled" if self.is_compiled else "AST-only"
        addr = f"@{self.address:#x}" if self.address else ""
        return f"{self.name}: {self.cpp_type} [{status}] {addr}"


@dataclass  
class NamespaceInfo:
    """
    Information about a cppyy namespace.
    
    Tracks what's defined in a namespace and helps detect conflicts.
    """
    name: str
    functions: Set[str] = field(default_factory=set)
    classes: Set[str] = field(default_factory=set)
    variables: Set[str] = field(default_factory=set)
    namespaces: Set[str] = field(default_factory=set)


@dataclass
class MemorySnapshot:
    """
    A snapshot of memory usage at a point in time.
    
    Useful for tracking memory growth during simulation.
    """
    timestamp: datetime
    label: str
    python_rss_mb: float
    python_heap_mb: float
    numpy_arrays_mb: float
    cppyy_objects: int
    gc_objects: int


class CppyyInspector:
    """
    Inspector for cppyy objects, memory, and namespaces.
    
    This class provides comprehensive introspection capabilities:
    
    1. OBJECT INSPECTION:
       - Examine how C++ objects are represented in Python
       - Check compilation state (AST vs machine code)
       - View function signatures and overloads
    
    2. MEMORY TRACKING:
       - Take snapshots of memory usage
       - Compare snapshots to detect leaks
       - Track NumPy array memory separately
    
    3. NAMESPACE ANALYSIS:
       - List all symbols in cppyy.gbl
       - Detect potential conflicts
       - Visualize namespace hierarchy
    
    4. ARRAY INSPECTION:
       - Verify zero-copy sharing
       - Check memory layout (C-contiguous, etc.)
       - Compare Python and C++ views of same data
    
    Example:
        >>> inspector = CppyyInspector()
        >>> 
        >>> # Inspect a compiled function
        >>> cppyy.cppdef("void foo(int x) {}")
        >>> info = inspector.inspect_object(cppyy.gbl.foo)
        >>> print(info.summary())
        foo: CPPOverload [compiled] 
        >>>
        >>> # Take memory snapshots
        >>> snapshot1 = inspector.take_memory_snapshot("before")
        >>> # ... do stuff ...
        >>> snapshot2 = inspector.take_memory_snapshot("after")
        >>> inspector.compare_snapshots(snapshot1, snapshot2)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Create an inspector.
        
        Args:
            verbose: If True, print detailed information during inspection
        """
        self.verbose = verbose
        self.snapshots: List[MemorySnapshot] = []
        self._known_namespaces: Dict[str, NamespaceInfo] = {}
        
        # Set up console for rich output
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def inspect_object(self, obj: Any, name: Optional[str] = None) -> CppyyObjectInfo:
        """
        Deeply inspect a cppyy-bound object.
        
        This examines how cppyy represents a C++ object in Python,
        including its type, address, compilation state, and more.
        
        Args:
            obj: The cppyy object to inspect
            name: Optional name (will try to determine automatically)
            
        Returns:
            CppyyObjectInfo with all discovered information
        """
        info = CppyyObjectInfo(
            name=name or getattr(obj, '__name__', str(type(obj).__name__)),
            cpp_type=getattr(obj, '__cpp_name__', str(type(obj))),
            python_type=str(type(obj)),
        )
        
        # Check if it's a cppyy proxy
        type_name = str(type(obj))
        info.is_proxy = 'cppyy' in type_name.lower()
        
        # Try to get memory address
        if hasattr(obj, '__address__'):
            info.address = obj.__address__
        elif hasattr(obj, 'data'):
            # For arrays, get the data pointer
            try:
                if hasattr(obj.data, '__call__'):
                    ptr = obj.data()
                    if hasattr(ptr, '__int__'):
                        info.address = int(ptr)
            except:
                pass
        
        # Get docstring (contains signature for functions)
        info.docstring = getattr(obj, '__doc__', None)
        
        # For functions, check overloads
        if 'CPPOverload' in type_name or 'function' in type_name.lower():
            info.is_compiled = True  # Functions are compiled on definition
            if info.docstring:
                # Parse overloads from docstring
                info.overloads = [
                    line.strip() for line in info.docstring.split('\n')
                    if line.strip() and '=>' not in line
                ]
        
        # For classes, get members
        if 'Meta' in type_name or hasattr(obj, '__dict__'):
            try:
                for attr_name in dir(obj):
                    if not attr_name.startswith('_'):
                        try:
                            attr = getattr(obj, attr_name)
                            info.members[attr_name] = str(type(attr).__name__)
                        except:
                            info.members[attr_name] = "<error>"
            except:
                pass
        
        if self.verbose:
            self._print_object_info(info)
        
        return info
    
    def _print_object_info(self, info: CppyyObjectInfo):
        """Print object info using rich if available."""
        if RICH_AVAILABLE and self.console:
            table = Table(title=f"cppyy Object: {info.name}", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("C++ Type", info.cpp_type)
            table.add_row("Python Type", info.python_type)
            table.add_row("Is Proxy", str(info.is_proxy))
            table.add_row("Is Compiled", str(info.is_compiled))
            if info.address:
                table.add_row("Address", f"{info.address:#x}")
            if info.overloads:
                table.add_row("Overloads", "\n".join(info.overloads[:5]))
            if info.members:
                members_str = ", ".join(list(info.members.keys())[:10])
                table.add_row("Members", members_str)
            
            self.console.print(table)
        else:
            print(f"\n{'='*60}")
            print(f"cppyy Object: {info.name}")
            print(f"{'='*60}")
            print(f"  C++ Type:    {info.cpp_type}")
            print(f"  Python Type: {info.python_type}")
            print(f"  Is Proxy:    {info.is_proxy}")
            print(f"  Is Compiled: {info.is_compiled}")
            if info.address:
                print(f"  Address:     {info.address:#x}")
    
    def inspect_array(self, numpy_array: np.ndarray, name: str = "array") -> Dict[str, Any]:
        """
        Inspect a NumPy array's memory layout.
        
        This is crucial for verifying zero-copy sharing with C++.
        For efficient C++ interop, arrays should be:
        - C-contiguous (row-major)
        - Properly aligned
        - Owned by Python (not a view that might be invalidated)
        
        Args:
            numpy_array: The array to inspect
            name: Name for display purposes
            
        Returns:
            Dictionary with detailed memory information
        """
        info = {
            'name': name,
            'shape': numpy_array.shape,
            'dtype': str(numpy_array.dtype),
            'size_elements': numpy_array.size,
            'size_bytes': numpy_array.nbytes,
            'data_pointer': numpy_array.ctypes.data,
            'c_contiguous': numpy_array.flags['C_CONTIGUOUS'],
            'f_contiguous': numpy_array.flags['F_CONTIGUOUS'],
            'writeable': numpy_array.flags['WRITEABLE'],
            'owns_data': numpy_array.flags['OWNDATA'],
            'aligned': numpy_array.flags['ALIGNED'],
            'strides': numpy_array.strides,
        }
        
        if self.verbose:
            self._print_array_info(info)
        
        return info
    
    def _print_array_info(self, info: Dict[str, Any]):
        """Print array info using rich if available."""
        if RICH_AVAILABLE and self.console:
            table = Table(title=f"NumPy Array: {info['name']}", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Shape", str(info['shape']))
            table.add_row("Dtype", info['dtype'])
            table.add_row("Size (bytes)", f"{info['size_bytes']:,}")
            table.add_row("Data Pointer", f"{info['data_pointer']:#x}")
            table.add_row("C-Contiguous", "✓" if info['c_contiguous'] else "✗")
            table.add_row("Owns Data", "✓" if info['owns_data'] else "✗")
            table.add_row("Writeable", "✓" if info['writeable'] else "✗")
            
            self.console.print(table)
        else:
            print(f"\n--- Array: {info['name']} ---")
            print(f"  Shape:        {info['shape']}")
            print(f"  Dtype:        {info['dtype']}")
            print(f"  Size:         {info['size_bytes']:,} bytes")
            print(f"  Data Pointer: {info['data_pointer']:#x}")
            print(f"  C-Contiguous: {info['c_contiguous']}")
            print(f"  Owns Data:    {info['owns_data']}")
    
    def take_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """
        Take a snapshot of current memory usage.
        
        This captures:
        - Python process RSS (total memory)
        - Python heap usage
        - NumPy array memory
        - Number of cppyy proxy objects
        - Total garbage collector tracked objects
        
        Args:
            label: Descriptive label for this snapshot
            
        Returns:
            MemorySnapshot with current memory state
        """
        gc.collect()  # Clean up first for accurate measurement
        
        # Get process memory
        python_rss = 0.0
        python_heap = 0.0
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_info = process.memory_info()
            python_rss = mem_info.rss / (1024 * 1024)  # MB
            python_heap = mem_info.vms / (1024 * 1024)  # MB
        
        # Count NumPy memory
        numpy_mb = 0.0
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                numpy_mb += obj.nbytes / (1024 * 1024)
        
        # Count cppyy objects
        cppyy_count = sum(
            1 for obj in gc.get_objects()
            if 'cppyy' in str(type(obj)).lower()
        )
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            label=label,
            python_rss_mb=python_rss,
            python_heap_mb=python_heap,
            numpy_arrays_mb=numpy_mb,
            cppyy_objects=cppyy_count,
            gc_objects=len(gc.get_objects())
        )
        
        self.snapshots.append(snapshot)
        
        if self.verbose:
            self._print_snapshot(snapshot)
        
        return snapshot
    
    def _print_snapshot(self, snapshot: MemorySnapshot):
        """Print memory snapshot."""
        if RICH_AVAILABLE and self.console:
            panel = Panel(
                f"RSS: {snapshot.python_rss_mb:.1f} MB | "
                f"NumPy: {snapshot.numpy_arrays_mb:.1f} MB | "
                f"cppyy objects: {snapshot.cppyy_objects} | "
                f"GC objects: {snapshot.gc_objects:,}",
                title=f"Memory Snapshot: {snapshot.label}",
                border_style="blue"
            )
            self.console.print(panel)
        else:
            print(f"\n[Memory Snapshot: {snapshot.label}]")
            print(f"  RSS:          {snapshot.python_rss_mb:.1f} MB")
            print(f"  NumPy:        {snapshot.numpy_arrays_mb:.1f} MB")
            print(f"  cppyy objects: {snapshot.cppyy_objects}")
            print(f"  GC objects:   {snapshot.gc_objects:,}")
    
    def compare_snapshots(
        self,
        before: MemorySnapshot,
        after: MemorySnapshot
    ) -> Dict[str, float]:
        """
        Compare two memory snapshots.
        
        Useful for detecting memory leaks or understanding
        memory growth during simulation.
        
        Args:
            before: Earlier snapshot
            after: Later snapshot
            
        Returns:
            Dictionary with differences
        """
        diff = {
            'rss_delta_mb': after.python_rss_mb - before.python_rss_mb,
            'numpy_delta_mb': after.numpy_arrays_mb - before.numpy_arrays_mb,
            'cppyy_delta': after.cppyy_objects - before.cppyy_objects,
            'gc_delta': after.gc_objects - before.gc_objects,
            'time_delta_s': (after.timestamp - before.timestamp).total_seconds(),
        }
        
        if self.verbose:
            if RICH_AVAILABLE and self.console:
                table = Table(
                    title=f"Memory Change: {before.label} → {after.label}",
                    box=box.ROUNDED
                )
                table.add_column("Metric", style="cyan")
                table.add_column("Change", style="yellow")
                
                table.add_row("RSS", f"{diff['rss_delta_mb']:+.2f} MB")
                table.add_row("NumPy", f"{diff['numpy_delta_mb']:+.2f} MB")
                table.add_row("cppyy objects", f"{diff['cppyy_delta']:+d}")
                table.add_row("Time elapsed", f"{diff['time_delta_s']:.3f}s")
                
                self.console.print(table)
            else:
                print(f"\n[Memory Change: {before.label} → {after.label}]")
                print(f"  RSS:    {diff['rss_delta_mb']:+.2f} MB")
                print(f"  NumPy:  {diff['numpy_delta_mb']:+.2f} MB")
                print(f"  cppyy:  {diff['cppyy_delta']:+d} objects")
        
        return diff
    
    def scan_global_namespace(self) -> NamespaceInfo:
        """
        Scan cppyy.gbl for all defined symbols.
        
        This is useful for:
        - Understanding what's been defined
        - Detecting potential naming conflicts
        - Debugging "symbol not found" errors
        
        Returns:
            NamespaceInfo with all discovered symbols
        """
        info = NamespaceInfo(name="cppyy.gbl")
        
        for name in dir(cppyy.gbl):
            if name.startswith('_'):
                continue
            
            try:
                obj = getattr(cppyy.gbl, name)
                type_str = str(type(obj))
                
                if 'CPPOverload' in type_str or 'function' in type_str.lower():
                    info.functions.add(name)
                elif 'Meta' in type_str or 'class' in type_str.lower():
                    info.classes.add(name)
                elif 'namespace' in type_str.lower():
                    info.namespaces.add(name)
                else:
                    info.variables.add(name)
            except:
                pass
        
        self._known_namespaces['gbl'] = info
        
        if self.verbose:
            self._print_namespace_info(info)
        
        return info
    
    def _print_namespace_info(self, info: NamespaceInfo):
        """Print namespace info using rich if available."""
        if RICH_AVAILABLE and self.console:
            tree = Tree(f"[bold blue]{info.name}[/bold blue]")
            
            if info.namespaces:
                ns_branch = tree.add("[cyan]Namespaces[/cyan]")
                for ns in sorted(info.namespaces):
                    ns_branch.add(ns)
            
            if info.classes:
                cls_branch = tree.add("[green]Classes[/green]")
                for cls in sorted(info.classes):
                    cls_branch.add(cls)
            
            if info.functions:
                fn_branch = tree.add("[yellow]Functions[/yellow]")
                for fn in sorted(info.functions):
                    fn_branch.add(fn)
            
            if info.variables:
                var_branch = tree.add("[magenta]Variables[/magenta]")
                for var in sorted(info.variables):
                    var_branch.add(var)
            
            self.console.print(tree)
        else:
            print(f"\n{'='*60}")
            print(f"Namespace: {info.name}")
            print(f"{'='*60}")
            print(f"  Namespaces: {sorted(info.namespaces)}")
            print(f"  Classes:    {sorted(info.classes)}")
            print(f"  Functions:  {sorted(info.functions)}")
            print(f"  Variables:  {sorted(info.variables)}")
    
    def check_for_conflicts(self, names: List[str]) -> List[str]:
        """
        Check if any given names would conflict with existing symbols.
        
        Call this before defining new C++ code to avoid silent conflicts.
        
        Args:
            names: List of names you want to define
            
        Returns:
            List of names that already exist in cppyy.gbl
        """
        conflicts = []
        for name in names:
            if hasattr(cppyy.gbl, name):
                conflicts.append(name)
        
        if conflicts and self.verbose:
            if RICH_AVAILABLE and self.console:
                self.console.print(
                    f"[bold red]WARNING:[/bold red] These names already exist: {conflicts}"
                )
            else:
                print(f"WARNING: These names already exist: {conflicts}")
        
        return conflicts
    
    def verify_zero_copy(
        self,
        numpy_array: np.ndarray,
        cpp_ptr_getter: callable
    ) -> bool:
        """
        Verify that a NumPy array and C++ pointer share the same memory.
        
        This is crucial for performance - if they don't share memory,
        you're copying data on every call!
        
        Args:
            numpy_array: The NumPy array
            cpp_ptr_getter: A callable that returns the C++ pointer
            
        Returns:
            True if they share the same memory address
            
        Example:
            >>> arr = np.zeros(100)
            >>> # Assuming cpp_func stores the pointer
            >>> inspector.verify_zero_copy(arr, lambda: stored_ptr)
        """
        numpy_ptr = numpy_array.ctypes.data
        cpp_ptr = cpp_ptr_getter()
        
        # Convert cppyy pointer to int if needed
        if hasattr(cpp_ptr, '__int__'):
            cpp_ptr = int(cpp_ptr)
        
        same_memory = numpy_ptr == cpp_ptr
        
        if self.verbose:
            status = "✓ ZERO-COPY" if same_memory else "✗ COPYING DATA!"
            color = "green" if same_memory else "red"
            
            if RICH_AVAILABLE and self.console:
                self.console.print(
                    f"[{color}]{status}[/{color}]  "
                    f"NumPy: {numpy_ptr:#x}  C++: {cpp_ptr:#x}"
                )
            else:
                print(f"{status}  NumPy: {numpy_ptr:#x}  C++: {cpp_ptr:#x}")
        
        return same_memory