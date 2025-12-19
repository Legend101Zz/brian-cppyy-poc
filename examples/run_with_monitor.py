"""
Demonstration of NeuronGroup with StateMonitor.

This example shows:
1. Creating a NeuronGroup with differential equations
2. Recording state variables with StateMonitor
3. Using CppyyInspector to examine memory and cppyy objects
4. Visualizing recorded data

Run:
    python examples/run_with_monitor.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Skipping plots.")

from cppyy_poc import CppyyInspector, NeuronGroup, StateMonitor


def main():
    """Run a simulation with recording and visualization."""
    
    print("\n" + "="*70)
    print("BRIAN2 CPPYY POC: NeuronGroup + StateMonitor")
    print("="*70)
    
    # Create inspector for debugging
    inspector = CppyyInspector(verbose=False)
    
    # Take initial memory snapshot
    print("\n[1] Taking initial memory snapshot...")
    snapshot_initial = inspector.take_memory_snapshot("initial")
    
    # =========================================================================
    # Create NeuronGroup
    # =========================================================================
    print("\n[2] Creating NeuronGroup...")
    
    neurons = NeuronGroup(
        n_neurons=50,
        equation="dv/dt = (-v + I) / tau",
        dt=0.0001,  # 0.1 ms timestep
        tau=0.010,  # 10 ms time constant
        verbose=True
    )
    
    # Set initial conditions
    # Random initial voltages
    neurons.v[:] = np.random.randn(neurons.n_neurons) * 0.2
    # Varied input currents so different neurons evolve differently
    neurons.I[:] = np.linspace(-0.5, 1.5, neurons.n_neurons)
    
    print(f"\n  Initial v: mean={neurons.v.mean():.3f}, std={neurons.v.std():.3f}")
    print(f"  Input I range: [{neurons.I.min():.2f}, {neurons.I.max():.2f}]")
    
    # =========================================================================
    # Create StateMonitor
    # =========================================================================
    print("\n[3] Creating StateMonitor...")
    
    # Record v from specific neurons: first, middle, last
    record_indices = [0, neurons.n_neurons // 2, neurons.n_neurons - 1]
    
    monitor = StateMonitor(
        source=neurons,
        variables=['v'],  # Variables to record
        record=record_indices,  # Which neurons to record
        verbose=True
    )
    
    print(f"\n  Recording neurons: {record_indices}")
    
    # =========================================================================
    # Inspect compiled code
    # =========================================================================
    print("\n[4] Inspecting compiled C++ code...")
    
    # Show the generated update function
    neurons.show_code()
    
    # Inspect the compiled function
    inspector.verbose = True
    inspector.inspect_object(neurons.code_object.compiled_function, "update_neurons")
    
    # Check NumPy array layout
    inspector.inspect_array(neurons.v, "neurons.v")
    
    # =========================================================================
    # Run simulation
    # =========================================================================
    print("\n[5] Running simulation...")
    inspector.verbose = False
    
    # Simulate for 50ms (500 steps at 0.1ms each)
    duration = 0.050  # 50 ms
    n_steps = int(duration / neurons.dt)
    
    print(f"\n  Simulating {duration*1000:.0f}ms ({n_steps} steps)...")
    
    for step in range(n_steps):
        # Update neurons
        neurons.update()
        
        # Record state
        monitor.record_timestep()
        
        # Progress indicator
        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{n_steps}, t={neurons.t*1000:.1f}ms")
    
    print(f"\n  ✓ Simulation complete!")
    print(f"  Final time: {neurons.t*1000:.1f}ms")
    print(f"  Recorded {monitor.N} timesteps")
    
    # =========================================================================
    # Analyze results
    # =========================================================================
    print("\n[6] Analyzing recorded data...")
    
    # Get recorded data
    t = monitor.t * 1000  # Convert to ms
    v = monitor.v         # Shape: (n_timesteps, n_recorded_neurons)
    
    print(f"\n  Recorded times shape: {t.shape}")
    print(f"  Recorded v shape:     {v.shape}")
    print(f"  Time range: [{t[0]:.2f}, {t[-1]:.2f}] ms")
    
    # Statistics for each recorded neuron
    for i, idx in enumerate(record_indices):
        v_i = v[:, i]
        print(f"\n  Neuron {idx}:")
        print(f"    Initial v: {v_i[0]:.4f}")
        print(f"    Final v:   {v_i[-1]:.4f}")
        print(f"    Mean v:    {v_i.mean():.4f}")
        print(f"    Input I:   {neurons.I[idx]:.4f}")
    
    # =========================================================================
    # Memory analysis
    # =========================================================================
    print("\n[7] Memory analysis...")
    inspector.verbose = True
    snapshot_final = inspector.take_memory_snapshot("after_simulation")
    inspector.compare_snapshots(snapshot_initial, snapshot_final)
    
    # =========================================================================
    # Scan cppyy namespace
    # =========================================================================
    print("\n[8] Scanning cppyy global namespace...")
    inspector.scan_global_namespace()
    
    # =========================================================================
    # Plot results
    # =========================================================================
    if HAS_MATPLOTLIB:
        print("\n[9] Creating plots...")
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot voltage traces
        ax1 = axes[0]
        for i, idx in enumerate(record_indices):
            ax1.plot(t, v[:, i], label=f'Neuron {idx} (I={neurons.I[idx]:.2f})')
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (v)')
        ax1.set_title('Recorded Voltage Traces')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot final voltage distribution
        ax2 = axes[1]
        ax2.bar(range(neurons.n_neurons), neurons.v, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Final Voltage')
        ax2.set_title(f'Final Voltage Distribution (t={neurons.t*1000:.0f}ms)')
        ax2.grid(True, alpha=0.3)
        
        # Highlight recorded neurons
        for idx in record_indices:
            ax2.bar(idx, neurons.v[idx], color='red', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('monitor_results.png', dpi=150)
        print("\n  ✓ Saved plot to monitor_results.png")
        plt.show()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    return neurons, monitor


if __name__ == "__main__":
    neurons, monitor = main()