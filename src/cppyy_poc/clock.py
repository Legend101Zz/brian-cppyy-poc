"""
Clock: Manages simulation time for Brian2-style scheduling.

In Brian2, Clocks determine when different operations run. Each CodeRunner
(like NeuronGroup's state updater or StateMonitor) is associated with a Clock
that determines its update frequency.

This simplified implementation provides:
- Current time tracking
- Timestep (dt) management
- Step counting for iteration tracking
"""

from typing import Optional


class Clock:
    """
    A simulation clock that tracks time progression.
    
    The Clock is responsible for:
    1. Maintaining the current simulation time
    2. Providing the timestep (dt) for integration
    3. Counting timesteps for scheduling
    
    In the full Brian2 system, multiple clocks can exist with different
    dt values, allowing for multi-rate simulations. Here we simplify
    to a single clock per simulation.
    
    Attributes:
        dt: The timestep duration in seconds
        t: Current simulation time in seconds
        timestep: Number of timesteps completed
        
    Example:
        >>> clock = Clock(dt=0.0001)  # 0.1 ms timestep
        >>> clock.t
        0.0
        >>> clock.tick()
        >>> clock.t
        0.0001
        >>> clock.timestep
        1
    """
    
    # Counter to generate unique clock IDs
    _clock_counter = 0
    
    def __init__(self, dt:float = 0.0001, name: Optional[str] = None ):
        """
        Create a new Clock.
        
        Args:
            dt: Timestep duration in seconds. Default is 0.1ms (0.0001s),
                which is typical for neural simulations.
            name: Optional name for the clock. If not provided, a unique
                  name will be generated.
        """
        # Generate unique ID
        Clock._clock_counter += 1
        self.id = Clock._clock_counter
        self.name = name or f"clock_{self.id}"
        
        # Time parameters
        self._dt = dt
        self._t = 0.0
        self._timestep = 0
        
    @property
    def dt(self) -> float:
        """The timestep duration in seconds."""
        return self._dt
    
    @property
    def t(self) -> float:
        """Current simulation time in seconds."""
        return self._t
    
    @property
    def timestep(self) -> int:
        """Number of completed timesteps."""
        return self._timestep
    
    def tick(self):
        """
        Advance the clock by one timestep.
        
        This is called by the Network during simulation to progress time.
        Each tick:
        1. Increments the timestep counter
        2. Updates the current time (t = timestep * dt)
        """
        self._timestep += 1
        self._t = self._timestep * self._dt

    def reset(self):
        """
        Reset the clock to t=0.
        
        Useful for running multiple simulations with the same objects.
        """
        self._t = 0.0
        self._timestep = 0
        
    def __repr__(self) -> str:
        return f"Clock({self.name}, dt={self.dt*1000:.3f}ms, t={self.t*1000:.3f}ms)"
        