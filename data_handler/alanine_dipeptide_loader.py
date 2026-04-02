import numpy as np
import mdshare
from sklearn.cluster import KMeans

def load_alanine_dipeptide_markov_chain(
    n_states: int = 30,
    lag: int = 1,
    seed: int = 0,
    smoothing: float = 1e-8,
    working_directory: str = "data",
):
    """
    Downloads alanine dipeptide backbone dihedral trajectories, discretizes the
    trajectories into microstates, and builds a row-stochastic transition matrix.

    Requirements:
        pip install mdshare scikit-learn
    """
    

    # Download precomputed backbone dihedral trajectories
    dihedral_file = mdshare.fetch(
        "alanine-dipeptide-3x250ns-backbone-dihedrals.npz",
        working_directory=working_directory,
    )

    # Load all trajectory segments
    with np.load(dihedral_file) as fh:
        dihedral = [fh[f"arr_{i}"] for i in range(len(fh.files))]

    # Concatenate frames from all trajectories and cluster them into discrete states
    all_frames = np.concatenate(dihedral, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=20)
    kmeans.fit(all_frames)

    # Convert each continuous trajectory into a discrete trajectory
    dtrajs = [kmeans.predict(traj) for traj in dihedral]

    # Count transitions at the chosen lag time
    C = np.zeros((n_states, n_states), dtype=float)
    for traj in dtrajs:
        for t in range(len(traj) - lag):
            i = traj[t]
            j = traj[t + lag]
            C[i, j] += 1.0

    # Add tiny smoothing to avoid exact zero rows and normalize row-wise
    C = C + smoothing
    P = C / C.sum(axis=1, keepdims=True)

    return P