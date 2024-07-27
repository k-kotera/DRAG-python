import numpy as np
from numba import njit
from typing import Tuple

def DRAG(TS: np.ndarray, window_size: int, r: float, z_normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    DRAG algorithm for time series discords discovery.
    
    Parameters
    TS : 1D time series data of length n
    window_size : length of the subsequences 
    r : discords threshold
    Returns
    candidate_flags : flags whether S[i] is a candidate or not. (n - window_size + 1)
    NN_distances : NN distance if C[i] == True, else inf (n - window_size + 1)
    """
    subsequences: np.ndarray = generate_subsequences(TS, window_size, z_normalize)
    candidate_flags: np.ndarray = candidates_selection_phase(subsequences, r)
    candidate_flags, NN_distances = discord_refinement_phase(subsequences, candidate_flags, r)
    return candidate_flags, NN_distances

@njit
def generate_subsequences(TS: np.ndarray, window_size: int, z_normalize: bool = True) -> np.ndarray:
    """
    Function to create a set of subsequences from time series data.

    Parameters
    TS : 1D time series data of length n
    window_size : length of the subsequences 
    Returns
    S : a set of subsequences with shape (n - window_size + 1, window_size)
    """
    n: int = TS.shape[0]
    output_shape: Tuple[int, int] = (n - window_size + 1, window_size)
    S: np.ndarray = np.empty(output_shape, dtype=TS.dtype)
    for i in range(output_shape[0]):
        subsequence: np.ndarray = TS[i : i + window_size]
        if z_normalize:
            mean: float = np.mean(subsequence)
            std: float = np.std(subsequence)
            S[i] = (subsequence - mean) / std
        else:
            S[i] = subsequence
    return S

@njit
def candidates_selection_phase(S: np.ndarray, r: float) -> np.ndarray:
    """
    Phase 1 of DRAG.
    
    Parameters
    S : a set of subsequences with shape (n - window_size + 1, window_size)
    r : discords threshold
    Returns
    C : flags whether S[i] is a candidate or not. (n - window_size + 1)
    """
    discords_candidate_size: int
    window_size: int
    discords_candidate_size, window_size = S.shape
    C: np.ndarray = np.zeros(discords_candidate_size, dtype=np.bool_)
    for i in range(discords_candidate_size):
        iscandidate: bool = True
        j_indexes: np.ndarray = np.where(C == True)[0]
        for j in j_indexes:
            if abs(j - i) >= window_size: # Is (i,j) trivial match?
                d: float = np.linalg.norm(S[i] - S[j])
                if d < r:
                    C[j] = False
                    iscandidate = False
        if iscandidate:
            C[i] = True
    return C

@njit
def discord_refinement_phase(S: np.ndarray, C: np.ndarray, r: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase 2 of DRAG.
    
    Parameters
    S : a set of subsequences with shape (n - window_size + 1, window_size)
    C : flags whether S[i] is a candidate or not. (n - window_size + 1)
    r : discords threshold
    Returns
    C : flags whether S[i] is a candidate or not. (n - window_size + 1)
    C_dist : NN distance if C[i] == True, else inf (n - window_size + 1)
    """
    discords_candidate_size: int
    window_size: int
    discords_candidate_size, window_size = S.shape
    C_dist: np.ndarray = np.ones(discords_candidate_size) * np.inf
    for i in range(discords_candidate_size):
        j_indexes: np.ndarray = np.where(C == True)[0]
        for j in j_indexes:
            if abs(j - i) >= window_size: # Is (i,j) trivial match?
                d: float = np.linalg.norm(S[i] - S[j])
                if d < r:
                    C[j] = False
                else:
                    C_dist[j] = min(d, C_dist[j])
    return C, C_dist
