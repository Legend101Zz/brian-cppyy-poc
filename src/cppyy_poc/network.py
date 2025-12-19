"""
Network: Manages multiple simulation objects and provides namespace isolation.

In Brian2, a Network collects NeuronGroups, Synapses, Monitors, etc. and
runs them together. It handles:
1. Scheduling (which objects run when)
2. Namespace management
3. Clock synchronization

Our simplified Network demonstrates:
- How to isolate multiple simulations via C++ namespaces
- Managing multiple NeuronGroups and Monitors together
- Detecting and avoiding namespace conflicts in cppyy

NAMESPACE ISOLATION:
cppyy maintains a single global C++ namespace (cppyy.gbl). When creating
multiple simulations, function name conflicts can occur. We solve this by:
1. Using unique function names (e.g., update_neurons_0, update_neurons_1)
2. Optionally wrapping code in C++ namespaces (namespace net_0 { ... })
3. Tracking what's been defined to detect conflicts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .clock import Clock


@dataclass
class NetworkStats:
    """Statistics about a network and its simulation."""
    name: str
    n_groups: int
    n_monitors: int
    total_neurons: int
    timesteps_run: int
    current_time: float
    cpp_functions: List[str] = field(default_factory=list)


class Network:
    """
    A container for simulation objects that handles scheduling and namespaces.
    
    The Network provides:
    1. OBJECT MANAGEMENT: Add/remove NeuronGroups, Monitors, etc.
    2. SCHEDULING: Run all objects in correct order each timestep
    3. NAMESPACE ISOLATION: Prevent C++ name conflicts between networks
    4. CLOCK SYNCHRONIZATION: Ensure all objects use compatible clocks
    
    NAMESPACE CONFLICT HANDLING:
    Each Network can optionally use a C++ namespace to isolate its
    compiled code. This is important when:
    - Running multiple independent simulations
    - Comparing different model configurations
    - Testing code without side effects
    
    Without namespaces, if you create two NeuronGroups with the same
    structure, they might conflict. With namespaces:
    - Network "net_a" puts code in namespace net_a { ... }
    - Network "net_b" puts code in namespace net_b { ... }
    - No conflicts!
    
    Attributes:
        name: Unique name for this network
        clock: The master clock for this network
        objects: Dictionary of all contained objects
        namespace: C++ namespace name for isolation
        
    Example:
        >>> # Create two isolated simulations
        >>> net1 = Network("simulation_A")
        >>> neurons1 = NeuronGroup(100, "dv/dt=-v/tau", namespace="sim_a")
        >>> net1.add(neurons1)
        >>> 
        >>> net2 = Network("simulation_B") 
        >>> neurons2 = NeuronGroup(100, "dv/dt=-v/tau", namespace="sim_b")
        >>> net2.add(neurons2)
        >>> 
        >>> # Run independently - no conflicts!
        >>> net1.run(duration=0.1)
        >>> net2.run(duration=0.1)
    """
    
    # Track all created networks for conflict detection
    _all_networks: Dict[str, 'Network'] = {}
    _defined_symbols: Set[str] = set()
    
    def __init__(
        self,
        name: str,
        dt: float = 0.0001,
        use_namespace: bool = True
        ):
        """
        Create a Network.
        Args:
            name: Unique identifier for this network
            dt: Default timestep for the network's clock
            use_namespace: If True, wrap compiled code in a C++ namespace
        """
        self.name = name
        self.clock = Clock(dt=dt, name=f"{name}_clock")
        self.use_namespace = use_namespace
        self.namespace = name if use_namespace else None
        
        # Storage for simulation objects
        self._neurongroups: List = []
        self._monitors: List = []
        self._all_objects: List = []
        
        # Track C++ symbols defined by this network
        self._my_symbols: Set[str] = set()
        
        # Register this network
        if name in Network._all_networks:
            raise ValueError(f"Network '{name}' already exists!")
        Network._all_networks[name] = self
        
        print(f"\n{'='*70}")
        print(f"Creating Network: {name}")
        print(f"{'='*70}")
        print(f"  Namespace isolation: {use_namespace}")
        if use_namespace:
            print(f"  C++ namespace: {self.namespace}")

    def add(self, *objects):
        """
        Add simulation objects to the network.
        
        Args:
            *objects: NeuronGroups, Monitors, etc. to add
        """
        for obj in objects:
            obj_type = type(obj).__name__
            
            if obj_type == 'NeuronGroup':
                self._neurongroups.append(obj)
                print(f"  Added NeuronGroup: {obj.name}")
            elif obj_type == 'StateMonitor':
                self._monitors.append(obj)
                print(f"  Added StateMonitor: {obj.name}")
            
            self._all_objects.append(obj)

    def run(self, duration: float, report: bool = True):
        """
        Run the simulation for a specified duration.
        
        This executes all objects in the correct order:
        1. Update all NeuronGroups (state equations)
        2. Record all Monitors
        3. Advance clocks
        
        Args:
            duration: Simulation time in seconds
            report: If True, print progress
        """
        n_steps = int(duration / self.clock.dt)
        
        if report:
            print(f"\n  Running {self.name} for {duration*1000:.1f}ms ({n_steps} steps)")
        
        for step in range(n_steps):
            # Update neuron groups
            for group in self._neurongroups:
                group.code_object()
            
            # Record monitors
            for monitor in self._monitors:
                monitor.record_timestep()
            
            # Advance clock
            self.clock.tick()
        
        # Sync object clocks
        for obj in self._all_objects:
            if hasattr(obj, 'clock'):
                obj.clock._t = self.clock.t
                obj.clock._timestep = self.clock.timestep
        
        if report:
            print(f"Complete. t={self.clock.t*1000:.1f}ms")

    def get_stats(self) -> NetworkStats:
        """Get statistics about this network."""
        total_neurons = sum(g.n_neurons for g in self._neurongroups)
        
        cpp_funcs = []
        for g in self._neurongroups:
            cpp_funcs.append(g.code_object.function_name)
        
        return NetworkStats(
            name=self.name,
            n_groups=len(self._neurongroups),
            n_monitors=len(self._monitors),
            total_neurons=total_neurons,
            timesteps_run=self.clock.timestep,
            current_time=self.clock.t,
            cpp_functions=cpp_funcs
        )

    @classmethod
    def check_global_conflicts(cls) -> Dict[str, List[str]]:
        """
        Check for symbol conflicts across all networks.
        
        Returns:
            Dictionary mapping conflicting symbols to their networks
        """
        all_symbols: Dict[str, List[str]] = {}
        
        for net_name, network in cls._all_networks.items():
            for symbol in network._my_symbols:
                if symbol not in all_symbols:
                    all_symbols[symbol] = []
                all_symbols[symbol].append(net_name)
        
        # Find conflicts (symbols in multiple networks)
        conflicts = {s: nets for s, nets in all_symbols.items() if len(nets) > 1}
        
        return conflicts