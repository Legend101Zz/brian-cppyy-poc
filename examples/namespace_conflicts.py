#!/usr/bin/env python3
"""
Demonstration of Multiple Networks and Namespace Conflict Handling.

This example shows how cppyy handles (and how to avoid) namespace conflicts
when running multiple simulations or creating multiple objects.

KEY CONCEPTS:
1. cppyy maintains a single global C++ namespace (cppyy.gbl)
2. All compiled code lives in this shared space
3. Without careful naming, conflicts can occur
4. Solutions: unique IDs, C++ namespaces, or namespace isolation

This script demonstrates:
- Creating multiple NeuronGroups
- How unique IDs prevent conflicts
- Using C++ namespaces for complete isolation
- Inspecting the cppyy namespace

Run:
    python examples/namespace_conflicts.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cppyy
import numpy as np
from cppyy_poc import CppyyInspector, Network, NeuronGroup, StateMonitor


def demo_unique_ids():
    """
    Demonstrate how unique IDs prevent basic naming conflicts.
    
    Each NeuronGroup gets a unique ID that's embedded in its function name.
    This prevents conflicts when creating multiple groups.
    """
    print("\n" + "="*70)
    print("DEMO 1: Unique IDs Prevent Basic Conflicts")
    print("="*70)
    
    # Create first neuron group
    print("\n[Creating first NeuronGroup...]")
    neurons1 = NeuronGroup(
        n_neurons=10,
        equation="dv/dt = (-v + I) / tau",
        verbose=False  # Quiet mode
    )
    func1 = neurons1.code_object.function_name
    print(f"  Group 1 function: {func1}")
    
    # Create second neuron group with same parameters
    print("\n[Creating second NeuronGroup with same parameters...]")
    neurons2 = NeuronGroup(
        n_neurons=10,
        equation="dv/dt = (-v + I) / tau",
        verbose=False
    )
    func2 = neurons2.code_object.function_name
    print(f"  Group 2 function: {func2}")
    
    # They have different function names!
    print(f"\n  ✓ Different function names: {func1} != {func2}")
    print("  ✓ Both can coexist in cppyy.gbl without conflict")
    
    # Verify both work
    print("\n[Running both simulations...]")
    for _ in range(10):
        neurons1.update()
        neurons2.update()
    
    print(f"  Group 1 final mean v: {neurons1.v.mean():.4f}")
    print(f"  Group 2 final mean v: {neurons2.v.mean():.4f}")
    print("  ✓ Both simulations ran independently!")
    
    return neurons1, neurons2


def demo_explicit_namespaces():
    """
    Demonstrate using explicit C++ namespaces for stronger isolation.
    
    For complex scenarios with multiple projects or when you want
    complete isolation, you can wrap code in C++ namespaces.
    """
    print("\n" + "="*70)
    print("DEMO 2: Explicit C++ Namespaces")
    print("="*70)
    
    # Create groups in different namespaces
    print("\n[Creating NeuronGroups in different C++ namespaces...]")
    
    # Note: In a full implementation, we'd pass namespace to NeuronGroup
    # For this demo, we'll show how the concept works
    
    # Simulate namespace usage by defining code directly
    code_namespace_a = """
    namespace simulation_A {
        void update_demo_A(double* v, int n) {
            for (int i = 0; i < n; i++) {
                v[i] = v[i] * 0.99;  // Decay
            }
        }
    }
    """
    
    code_namespace_b = """
    namespace simulation_B {
        void update_demo_B(double* v, int n) {
            for (int i = 0; i < n; i++) {
                v[i] = v[i] * 0.98;  // Different decay
            }
        }
    }
    """
    
    print("\n  Compiling code in namespace 'simulation_A'...")
    cppyy.cppdef(code_namespace_a)
    
    print("  Compiling code in namespace 'simulation_B'...")
    cppyy.cppdef(code_namespace_b)
    
    # Both functions exist but in different namespaces
    print("\n[Verifying namespace isolation...]")
    print(f"  simulation_A::update_demo_A exists: {hasattr(cppyy.gbl.simulation_A, 'update_demo_A')}")
    print(f"  simulation_B::update_demo_B exists: {hasattr(cppyy.gbl.simulation_B, 'update_demo_B')}")
    
    # Show they're in different namespaces
    print(f"\n  Full paths:")
    print(f"    {cppyy.gbl.simulation_A.update_demo_A}")
    print(f"    {cppyy.gbl.simulation_B.update_demo_B}")
    
    # Use the functions
    print("\n[Using namespaced functions...]")
    arr_a = np.ones(5)
    arr_b = np.ones(5)
    
    cppyy.gbl.simulation_A.update_demo_A(arr_a, 5)
    cppyy.gbl.simulation_B.update_demo_B(arr_b, 5)
    
    print(f"  After A's decay (0.99): {arr_a}")
    print(f"  After B's decay (0.98): {arr_b}")
    print("  ✓ Namespaced functions work independently!")


def demo_detecting_conflicts():
    """
    Demonstrate how to detect and avoid naming conflicts.
    """
    print("\n" + "="*70)
    print("DEMO 3: Detecting and Avoiding Conflicts")
    print("="*70)
    
    inspector = CppyyInspector(verbose=False)
    
    # Check what's already defined
    print("\n[Scanning cppyy.gbl for existing symbols...]")
    
    existing_symbols = []
    for name in dir(cppyy.gbl):
        if not name.startswith('_') and not name.startswith('std'):
            existing_symbols.append(name)
    
    print(f"  Found {len(existing_symbols)} user-defined symbols")
    
    # Show some examples
    update_funcs = [s for s in existing_symbols if 'update' in s.lower()]
    print(f"\n  Update functions defined: {update_funcs}")
    
    # Check for potential conflict
    print("\n[Checking for potential conflicts before defining new code...]")
    
    test_names = ['my_new_function', 'update_neurons_0', 'nonexistent_function']
    
    for name in test_names:
        exists = hasattr(cppyy.gbl, name)
        status = "⚠️  EXISTS" if exists else "✓ Available"
        print(f"  {name}: {status}")
    
    # Demonstrate the inspector's conflict checker
    print("\n[Using inspector.check_for_conflicts()...]")
    inspector.verbose = True
    conflicts = inspector.check_for_conflicts(test_names)
    
    if conflicts:
        print(f"\n  Conflicts detected: {conflicts}")
        print("  Solution: Use unique names or namespaces!")
    else:
        print("\n  No conflicts detected - safe to proceed.")


def demo_network_isolation():
    """
    Demonstrate using the Network class for managed isolation.
    """
    print("\n" + "="*70)
    print("DEMO 4: Network-Level Isolation")
    print("="*70)
    
    print("\n[Creating two independent Networks...]")
    
    # Create first network
    net_excitatory = Network("excitatory_network", use_namespace=True)
    neurons_exc = NeuronGroup(
        n_neurons=20,
        equation="dv/dt = (-v + I) / tau",
        verbose=False
    )
    neurons_exc.I[:] = 0.5  # Excitatory input
    monitor_exc = StateMonitor(neurons_exc, 'v', record=[0, 10, 19], verbose=False)
    net_excitatory.add(neurons_exc, monitor_exc)
    
    # Create second network with different dynamics
    net_inhibitory = Network("inhibitory_network", use_namespace=True)
    neurons_inh = NeuronGroup(
        n_neurons=20,
        equation="dv/dt = (-v + I) / tau",
        verbose=False
    )
    neurons_inh.I[:] = -0.3  # Inhibitory input
    monitor_inh = StateMonitor(neurons_inh, 'v', record=[0, 10, 19], verbose=False)
    net_inhibitory.add(neurons_inh, monitor_inh)
    
    # Run both networks
    print("\n[Running both networks independently...]")
    
    duration = 0.01  # 10ms
    net_excitatory.run(duration)
    net_inhibitory.run(duration)
    
    # Compare results
    print("\n[Comparing results...]")
    print(f"\n  Excitatory network:")
    print(f"    Stats: {net_excitatory.get_stats()}")
    print(f"    Mean final v: {neurons_exc.v.mean():.4f}")
    
    print(f"\n  Inhibitory network:")
    print(f"    Stats: {net_inhibitory.get_stats()}")
    print(f"    Mean final v: {neurons_inh.v.mean():.4f}")
    
    # Check for global conflicts
    print("\n[Checking for cross-network conflicts...]")
    conflicts = Network.check_global_conflicts()
    
    if conflicts:
        print(f"  ⚠️  Conflicts detected: {conflicts}")
    else:
        print("  ✓ No conflicts between networks!")


def demo_inspecting_cppyy_memory():
    """
    Deep dive into how cppyy represents objects in memory.
    """
    print("\n" + "="*70)
    print("DEMO 5: cppyy Memory and Object Representation")
    print("="*70)
    
    inspector = CppyyInspector(verbose=True)
    
    # Create a simple C++ class
    print("\n[Defining a C++ class via cppyy...]")
    
    cppyy.cppdef("""
    struct DemoStruct {
        double value;
        int count;
        
        DemoStruct(double v = 0.0, int c = 0) : value(v), count(c) {}
        
        void increment() { count++; }
        double get_value() const { return value; }
    };
    """)
    
    # Create instance
    print("\n[Creating C++ object instance...]")
    obj = cppyy.gbl.DemoStruct(42.0, 10)
    
    # Inspect it
    print("\n[Inspecting the C++ object...]")
    info = inspector.inspect_object(obj, "demo_struct_instance")
    
    # Show memory address
    print(f"\n  Object exists at memory address: {id(obj):#x} (Python)")
    
    # Modify and verify
    print("\n[Modifying object...]")
    print(f"  Before: value={obj.value}, count={obj.count}")
    obj.increment()
    obj.value = 100.0
    print(f"  After:  value={obj.value}, count={obj.count}")
    
    # Demonstrate that changes persist
    print("\n  ✓ Changes to C++ object are persistent (not copied)")


def main():
    """Run all demonstrations."""
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  CPPYY NAMESPACE AND MEMORY DEMONSTRATION".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print("""
This script demonstrates how cppyy handles multiple objects and namespaces.

KEY TAKEAWAYS:
1. cppyy uses a single global namespace (cppyy.gbl)
2. Unique IDs prevent basic naming conflicts
3. C++ namespaces provide stronger isolation
4. The CppyyInspector helps debug memory and objects
""")
    
    input("Press Enter to start the demos...\n")
    
    # Run demos
    demo_unique_ids()
    input("\nPress Enter for next demo...")
    
    demo_explicit_namespaces()
    input("\nPress Enter for next demo...")
    
    demo_detecting_conflicts()
    input("\nPress Enter for next demo...")
    
    demo_network_isolation()
    input("\nPress Enter for next demo...")
    
    demo_inspecting_cppyy_memory()
    
    print("\n" + "#"*70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("#"*70)


if __name__ == "__main__":
    main()