from abc import ABC, abstractmethod
import numpy as np

class BaseMarkovAggregator(ABC):
    """
    Abstract base class for all Markov Chain aggregation algorithms.
    Enforces a strict interface for empirical benchmarking.
    """
    
    def __init__(self, k_macro_states: int):
        """
        Initializes the aggregator with the target number of macro-states.
        
        Args:
            k_macro_states: Target dimensionality for the aggregated chain.
        """
        self.k = k_macro_states

    @abstractmethod
    def _partition_states(self, P: np.ndarray) -> np.ndarray:
        """
        Abstract method to be implemented by concrete subclasses.
        Contains the core mathematical logic to partition the state space.
        
        Args:
            P: (n, n) row-stochastic transition matrix.
            
        Returns:
            labels: (n,) array of integer assignments from 0 to k-1.
        """
        pass

    def aggregate(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        The standardized interface function to aggregate the input Markov Chain.
        
        Args:
            P: (n, n) row-stochastic transition matrix.
            
        Returns:
            P_hat: (k, k) aggregated transition matrix.
            V: (n, k) indicator matrix mapping micro-states to macro-states.
        """
        n = P.shape[0]
        
        # 1. Execute the specific algorithm's partitioning logic
        labels = self._partition_states(P)
        
        # 2. Construct the indicator matrix V
        V = np.zeros((n, self.k))
        for i, label in enumerate(labels):
            V[i, label] = 1.0
            
        # 3. Compute the aggregated matrix P_hat using uniform block averaging
        P_hat = np.zeros((self.k, self.k))
        for i in range(self.k):
            states_in_i = np.where(labels == i)[0]
            if len(states_in_i) == 0:
                continue
            for j in range(self.k):
                states_in_j = np.where(labels == j)[0]
                block_sum = np.sum(P[np.ix_(states_in_i, states_in_j)])
                P_hat[i, j] = block_sum / len(states_in_i)
                
        return P_hat, V