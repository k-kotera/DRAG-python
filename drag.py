import numpy as np
from numba import njit
from typing import Tuple

@njit
def generate_normalized_subsequences(TS: np.ndarray, window_size: int) -> np.ndarray:
    n: int = TS.shape[0]
    output_shape: Tuple[int, int] = (n - window_size + 1, window_size)
    S: np.ndarray = np.empty(output_shape, dtype=TS.dtype)
    for i in range(output_shape[0]):
        subsequence: np.ndarray = TS[i : i + window_size]
        mean: float = np.mean(subsequence)
        std: float = np.std(subsequence)
        S[i] = (subsequence - mean) / std
    return S

@njit
def candidates_selection_phase(S: np.ndarray, r: float) -> np.ndarray:
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

def drag(TS: np.ndarray, window_size: int, r: float) -> Tuple[np.ndarray, np.ndarray]:
    subsequences: np.ndarray = generate_normalized_subsequences(TS, window_size)
    candidate_flags: np.ndarray = candidates_selection_phase(subsequences, r)
    candidate_flags, NN_distances = discord_refinement_phase(subsequences, candidate_flags, r)
    return candidate_flags, NN_distances
