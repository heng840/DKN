import numpy as np

loaded_neurons = np.load('neurons.npy', allow_pickle=True).tolist()  # Convert back to a regular list
loaded_adjacency_matrix = np.load('adjacency_matrix.npy')