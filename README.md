---
# Brian2 cppyy Proof of Concept

This repository contains a proof of concept (POC) demonstrating how **cppyy** can be used as an alternative to **Cython** for Brian2’s runtime code generation and just-in-time (JIT) compilation system.

The goal is to explore whether cppyy can simplify the toolchain while retaining performance and flexibility required for neural simulations.
---

## Purpose

This POC evaluates the feasibility of replacing Cython with cppyy in Brian2’s runtime backend.

Key motivations include:

- Simplified toolchain without `.pyx` files or ahead-of-time compilation
- Fully in-memory C++ compilation at runtime
- Easier debugging through direct access to generated C++ code
- Faster development iteration without recompilation steps
- Potential for dynamic runtime optimization using LLVM

---

## Installation

### Using `uv` (recommended)

```bash
uv sync
```

### Using `pip`

```bash
pip install -e .
```

### With development dependencies

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
from cppyy_poc import NeuronGroup, StateMonitor, CppyyInspector
import numpy as np

neurons = NeuronGroup(
    n_neurons=100,
    equation="dv/dt = (-v + I) / tau",
    dt=0.0001,
    tau=0.010
)

monitor = StateMonitor(neurons, 'v', record=[0, 50, 99])

for _ in range(1000):
    neurons.update()
    monitor.record_timestep()

print(f"Time points: {monitor.t.shape}")
print(f"Voltage data: {monitor.v.shape}")

inspector = CppyyInspector()
inspector.inspect_object(neurons.code_object.compiled_function)
inspector.scan_global_namespace()
```

---

## Project Structure

```
brian2-cppyy-poc/
├── README.md
├── pyproject.toml
├── src/cppyy_poc/
│   ├── __init__.py
│   ├── variables.py
│   ├── codegen.py
│   ├── codeobject.py
│   ├── neurongroup.py
│   ├── statemonitor.py
│   ├── network.py
│   ├── clock.py
│   ├── inspector.py
│   └── templates/
│       ├── neuron_update.cpp.j2
│       ├── statemonitor.cpp.j2
│       └── dynamic_array.h.j2
└── examples/
    ├── run_simulation.py
    ├── run_with_monitor.py
    └── namespace_conflicts.py
```

---

## Features Demonstrated

### 1. NeuronGroup

Simulates neurons defined by differential equations using Euler integration.

```python
neurons = NeuronGroup(
    n_neurons=100,
    equation="dv/dt = (-v + I) / tau"
)

neurons.v[:] = np.random.randn(100) * 0.1
neurons.update()
```

State variables are stored as NumPy arrays and updated directly by JIT-compiled C++ code.

---

### 2. StateMonitor

Records state variables over time using dynamic 2D arrays backed by C++.

```python
monitor = StateMonitor(neurons, 'v', record=[0, 10, 50])

for _ in range(1000):
    neurons.update()
    monitor.record_timestep()

t = monitor.t
v = monitor.v
```

---

### 3. CppyyInspector

Provides tools for inspecting cppyy internals, memory layout, and namespaces.

```python
inspector = CppyyInspector()

inspector.inspect_object(cppyy.gbl.my_function)
inspector.inspect_array(numpy_array)

snap1 = inspector.take_memory_snapshot("before")
snap2 = inspector.take_memory_snapshot("after")
inspector.compare_snapshots(snap1, snap2)

inspector.scan_global_namespace()
```

---

### 4. Network and Namespace Isolation

Supports multiple independent simulations without symbol conflicts.

```python
net1 = Network("simulation_A")
net1.add(NeuronGroup(100, "dv/dt=-v/tau"))

net2 = Network("simulation_B")
net2.add(NeuronGroup(100, "dv/dt=-v/tau"))

net1.run(0.1)
net2.run(0.1)
```

---

## Examples

### Basic Simulation

```bash
python examples/run_simulation.py
```

### Simulation with Recording

```bash
python examples/run_with_monitor.py
```

This generates voltage traces and saves a plot to `monitor_results.png`.

### Namespace Conflict Demonstration

```bash
python examples/namespace_conflicts.py
```

---

## How It Works

### Compilation Pipeline

```
User Equation
     ↓
Jinja2 Template
     ↓
Generated C++ Code (string)
     ↓
cppyy.cppdef()
     ↓
Cling Parsing and Symbol Registration
     ↓
LLVM JIT Compilation (first call)
     ↓
Native Machine Code Execution
```

---

### Memory Model

```
Python Process
├── NumPy Arrays (state variables)
├── cppyy Proxies
└── Cling / LLVM JIT

NumPy arrays are shared by pointer.
C++ code reads and writes directly to Python memory.
No data copying or marshalling occurs.
```

---

## Performance Notes

### First Call

- JIT compilation cost: approximately 10–100 ms
- Happens once per generated function

### Subsequent Calls

- Execution time: approximately 0.01–0.1 ms
- Direct native execution

### Zero-Copy Arrays

- NumPy arrays passed as raw pointers
- C++ operates directly on Python-managed memory

---

## Comparison with Cython

| Aspect          | Cython                | cppyy                   |
| --------------- | --------------------- | ----------------------- |
| Compilation     | Ahead-of-time         | Just-in-time            |
| Generated files | `.pyx`, `.cpp`, `.so` | None                    |
| Build time      | Seconds to minutes    | Milliseconds            |
| Debugging       | Complex               | Direct (C++ string)     |
| Dependencies    | External C compiler   | LLVM bundled            |
| Optimization    | Static                | Dynamic (runtime-aware) |

---

## Debugging Tips

### View Generated C++ Code

```python
neurons.code_object.show_generated_code()
```

### Inspect cppyy Objects

```python
inspector.inspect_object(cppyy.gbl.some_function)
```

### Check for Symbol Conflicts

```python
inspector.check_for_conflicts(['func_name_1', 'func_name_2'])
```

### Memory Debugging

```python
snap1 = inspector.take_memory_snapshot("before")
snap2 = inspector.take_memory_snapshot("after")
inspector.compare_snapshots(snap1, snap2)
```

---

## TODO / Future Work

- Spike generation and threshold detection
- Synaptic connections
- Additional numerical integration methods (RK4, exponential Euler)
- Unit system integration
- Performance benchmarks against Cython
- Full equation parsing support

---

## References

- Brian2 Documentation
- cppyy Documentation
- Brian2 Code Generation Internals
- Cling Interpreter

---

## License

MIT License. See the `LICENSE` file for details.

---

## Summary

This proof of concept explores cppyy as a potential replacement for Cython in Brian2’s runtime code generation system. It demonstrates neuron simulation, monitoring, namespace isolation, memory inspection, and zero-copy execution, while significantly simplifying the compilation and debugging workflow.
