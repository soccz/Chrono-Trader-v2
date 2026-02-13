import os
import sys
import numpy as np
import pandas as pd
import warnings

# --- GPU/CPU Library Imports ---
try:
    from gpudtw import dtw_pairwise
    from cuml.cluster import DBSCAN as cuDBSCAN
    import cupy
    GPU_ENABLED = True
except ImportError:
    warnings.warn("cuML, cupy, or gpudtw not found. Falling back to CPU-based tslearn. This will be much slower.")
    from tslearn.clustering import TimeSeriesKMeans
    GPU_ENABLED = False

from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Add project root for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import get_market_index, create_sequences_from_index
from utils.logger import logger
from utils.config import config

def find_cluster_medoids(labels: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """
    Finds the medoid for each cluster given the labels and a pairwise distance matrix.
    The medoid is the point in the cluster with the minimum average distance to all other points in the same cluster.
    """
    medoid_indices = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # Skip noise points from DBSCAN
            continue
        
        # Get indices of all points belonging to the current cluster
        cluster_indices = np.where(labels == label)[0]
        
        # If cluster is too small, just take the first element
        if len(cluster_indices) == 0:
            continue
        if len(cluster_indices) == 1:
            medoid_indices.append(cluster_indices[0])
            continue

        # Create a sub-matrix of distances for the current cluster
        cluster_dist_matrix = distance_matrix[cluster_indices][:, cluster_indices]
        
        # Calculate the sum of distances for each point within the cluster
        intra_cluster_distances = cluster_dist_matrix.sum(axis=1)
        
        # Find the index of the point with the minimum sum of distances (the medoid)
        medoid_local_index = np.argmin(intra_cluster_distances)
        
        # Map the local index back to the original index
        medoid_global_index = cluster_indices[medoid_local_index]
        medoid_indices.append(medoid_global_index)
        
    return np.array(medoid_indices)


def build_library():
    """
    Builds a pattern library by clustering historical market index sequences.
    Uses gpudtw and cuML.DBSCAN for GPU-accelerated DTW clustering if available.
    Falls back to tslearn.TimeSeriesKMeans on CPU if GPU libraries are not found.
    """
    logger.info("--- Starting Pattern Library Construction ---")
    
    # 1. Fetch and prepare data
    logger.info("Fetching historical BTC+ETH market index...")
    market_index_df = get_market_index()
    if market_index_df.empty or len(market_index_df) < config.Pattern.LENGTH * 10:
        logger.error(f"Not enough data. Need at least {config.Pattern.LENGTH * 10} data points.")
        return

    logger.info(f"Creating sequences of length {config.Pattern.LENGTH}...")
    sequences = create_sequences_from_index(market_index_df, config.Pattern.LENGTH)
    if sequences.shape[0] < config.Pattern.N_CLUSTERS:
        logger.error(f"Not enough sequences ({sequences.shape[0]}) to form {config.Pattern.N_CLUSTERS} clusters.")
        return

    logger.info(f"Successfully created {sequences.shape[0]} sequences. Scaling for clustering...")
    sequences_scaled = TimeSeriesScalerMeanVariance().fit_transform(sequences)
    
    representative_patterns = None

    # 4. Cluster the sequences
    if GPU_ENABLED:
        try:
            logger.info("GPU detected. Using 'gpudtw' for distance calculation and 'cuml.DBSCAN' for clustering.")
            
            # Reshape from 3D to 2D for gpudtw, which expects (n_samples, n_timesteps)
            n_samples, n_timesteps, n_features = sequences_scaled.shape
            if n_features > 1:
                logger.warning(f"Data has {n_features} features, but DTW will only use the first one. Reshaping.")
            sequences_2d = sequences_scaled.reshape(n_samples, n_timesteps)

            # --- Step 1: Compute pairwise DTW distance matrix on GPU ---
            logger.info(f"Computing pairwise DTW distance matrix for {n_samples} sequences on GPU... (This can take a while)")
            distance_matrix_gpu = dtw_pairwise(sequences_2d)
            
            # --- Step 2: Cluster using DBSCAN with the precomputed distance matrix ---
            # TODO: The 'eps' parameter is crucial and requires tuning. This is a default value.
            # A good starting point is to analyze the k-distance graph.
            DBSCAN_EPS = 3.5 
            DBSCAN_MIN_SAMPLES = 5
            logger.info(f"Clustering with DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
            
            dbscan = cuDBSCAN(metric='precomputed', eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            labels_gpu = dbscan.fit_predict(distance_matrix_gpu)
            labels_cpu = labels_gpu.get()
            
            n_found_clusters = len(np.unique(labels_cpu)) - (1 if -1 in np.unique(labels_cpu) else 0)
            logger.info(f"DBSCAN found {n_found_clusters} clusters and {np.sum(labels_cpu == -1)} noise points.")

            if n_found_clusters == 0:
                logger.error("DBSCAN did not find any clusters. Try adjusting 'eps' and 'min_samples'.")
                return

            # --- Step 3: Find medoids for each cluster to act as centroids ---
            logger.info("Finding medoids for each cluster...")
            distance_matrix_cpu = cupy.asnumpy(distance_matrix_gpu)
            medoid_indices = find_cluster_medoids(labels_cpu, distance_matrix_cpu)
            
            if len(medoid_indices) == 0:
                logger.error("Could not determine any medoids from the found clusters.")
                return

            # The original (unscaled) sequences corresponding to the medoids are our patterns
            representative_patterns = sequences[medoid_indices]
            logger.info(f"Extracted {len(representative_patterns)} representative patterns (medoids).")

        except Exception as e:
            logger.error(f"An error occurred during GPU DTW clustering: {e}", exc_info=True)
            logger.warning("Falling back to CPU due to GPU processing error.")
            GPU_ENABLED = False # Force fallback

    if not GPU_ENABLED: # Fallback to CPU if GPU is disabled or failed
        logger.warning("Using CPU-based tslearn K-Means with DTW. This will be very slow.")
        
        kmeans = TimeSeriesKMeans(
            n_clusters=config.Pattern.N_CLUSTERS,
            metric="dtw",
            max_iter=15, # Keep iterations low for this slow method
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        kmeans.fit(sequences_scaled)
        
        # The cluster centers are our representative patterns
        # We need to un-scale them, but tslearn does not easily provide the scaler's inverse transform for centroids.
        # For now, we accept the scaled centroids as a limitation of the fallback method.
        representative_patterns = kmeans.cluster_centers_
        logger.warning("CPU fallback complete. Note: Fallback patterns are scaled, not original data.")

    # 5. Save the final library of representative patterns
    if representative_patterns is not None:
        output_dir = os.path.dirname(config.General.DB_PATH) # Should be 'data'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path_npy = os.path.join(output_dir, "pattern_library.npy")
        np.save(output_path_npy, representative_patterns)

        logger.info(f"--- Pattern Library successfully built and saved to {output_path_npy} ---")
        logger.info(f"Final representative patterns shape: {representative_patterns.shape}")
    else:
        logger.error("Failed to generate any representative patterns.")


if __name__ == "__main__":
    build_library()
