import copy
import itertools
import json

import einops
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
import random
from knowledge_neurons.kn import KnowledgeNeurons
from knowledge_neurons.patch import patch_ff_layer, unpatch_ff_layer, unpatch_ff_layers
import math
from functools import partial

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from knowledge_neurons.utils import get_model_output
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
random.seed(42)


def visualize_accuracy_changes(accuracies):
    # Box plot
    labels = list(accuracies.keys())
    data = [accuracies[label] for label in labels]
    plt.boxplot(data, labels=labels)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Under Neuron Suppression')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_neuron_sets(neurons, clusters, min_size=2):
    # Mapping from cluster ID to neuron sets
    neuron_sets_map = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in neuron_sets_map:
            neuron_sets_map[cluster_id] = []
        neuron_sets_map[cluster_id].append(neurons[idx])

    # Extracting neuron sets as a list
    neuron_sets = [neuron_set for neuron_set in neuron_sets_map.values() if len(neuron_set) >= min_size]
    return neuron_sets


class Dkn(KnowledgeNeurons):
    def __init__(
            self,
            model: nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            model_type: str = "gpt",
            device: str = None,
            # Any other parameters for GARNs
    ):
        super().__init__(model, tokenizer, model_type, device)
        self.original_weights = None
        self.inf_distance = 1000
        self.inf_threshold = 100
        self.model = model
        if self.model_type == 'llama':
            self.num_layers = len(self.model.model.layers)
        else:
            self.num_layers = len(self.model.transformer.h)
        # Precompute and store c_proj_weights and c_fc_weights for all layers
        self.c_proj_weights_list = [] # 4096,11008
        self.c_fc_weights_list = []

        for layer in range(self.num_layers):
            if self.model_type == 'llama':
                c_proj_weights = self.model.model.layers[layer].mlp.gate_proj.weight.detach().cpu().numpy()
                c_fc_weights = self.model.model.layers[layer].mlp.down_proj.weight.detach().cpu().numpy()
            else:
                c_proj_weights = self.model.transformer.h[layer].mlp.c_proj.weight.detach().cpu().numpy()
                c_fc_weights = self.model.transformer.h[layer].mlp.c_fc.weight.detach().cpu().numpy()

            self.c_proj_weights_list.append(c_proj_weights)
            self.c_fc_weights_list.append(c_fc_weights)
    def scaled_input_src(self, activations=None, steps=20, device="cpu", baseline_vector_path=None,
                     layer_idx=0, encoded_input=None, mask_idx=None):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
        out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
        )
        return out
    def scaled_input(self, activations=None, steps=20, device="cpu", baseline_vector_path=None,
                     layer_idx=0, encoded_input=None, mask_idx=None):
        """
        SIG方法，最有效。
        """
        # emb: (1, ffn_size)
        if self.model_type == 'bert':
            replace_token_id = self.tokenizer.mask_token_id
        else:
            replace_token_id = self.tokenizer.eos_token_id

        all_res = []

        # get original activations for the complete input
        _, original_activations = self.get_baseline_with_activations(encoded_input,
                                                                     layer_idx=layer_idx, mask_idx=mask_idx)

        for idx in range(encoded_input['input_ids'].size(1)):
            # create a copy of the input and replace the idx-th word with mask token
            masked_input = copy.deepcopy(encoded_input)
            masked_input['input_ids'][0][idx] = replace_token_id

            # get masked activations to use as baseline
            _, baseline_activations = self.get_baseline_with_activations(masked_input,
                                                                         layer_idx=layer_idx, mask_idx=mask_idx)
            step = (original_activations - baseline_activations) / steps  # (1, ffn_size)

            res = torch.cat([torch.add(baseline_activations, step * i) for i in range(steps)], dim=0)
            all_res.append(res)
        # average
        mean_res = torch.stack(all_res).mean(dim=0)
        return mean_res


    def get_coarse_neurons(
            self,
            prompt: str,
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            threshold: float = None,
            adaptive_threshold: float = None,
            percentile: float = None,
            attribution_method: str = "integrated_grads",
            pbar: bool = True,
            baseline_vector_path=None,
            normalization=False,
            k_path=1
    ):
        """
        """
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
            baseline_vector_path=baseline_vector_path,
            normalization=normalization
        )
        # sorted_attribution_scores, \
        #     sorted_attribution_indice = torch.sort(attribution_scores.flatten(), descending=True)
        #
        # # Convert indices to original shape
        # original_shape_indices = np.unravel_index(sorted_attribution_indice.cpu().numpy(), attribution_scores.shape)

        assert (
                sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        threshold = attribution_scores.max().item() * adaptive_threshold

        selected_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
        # Sort neurons based on attribution scores
        selected_neurons.sort(key=lambda x: attribution_scores[x[0]][x[1]], reverse=True)
        return selected_neurons
        # if self.model_type == 'bert':
        #     return selected_neurons
        # else:
        #     selected_neurons = self.distill_attributions(selected_neurons=selected_neurons,
        #                                                  attribution_scores=attribution_scores)
        #     return selected_neurons

    def get_scores_for_layer(
            self,
            prompt,
            ground_truth,
            encoded_input=None,
            layer_idx=0,
            batch_size=10,
            steps=20,
            attribution_method="integrated_grads",
            baseline_vector_path=None,
            k_path=1
    ):
        """
        """

        encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(prompt, ground_truth)
        n_sampling_steps = len(target_label)
        baseline_outputs, baseline_activations = self.get_baseline_with_activations(encoded_input, layer_idx, mask_idx)

        n_batches = steps // batch_size

        # Initialize an accumulator for the Distilled Gradients
        D_accumulator = torch.zeros_like(baseline_activations.squeeze(0))

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
                    prompt, ground_truth
                )

            if n_sampling_steps > 1:
                argmax_next_token = (
                    baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                )
                next_token_str = self.tokenizer.decode(argmax_next_token)

            scaled_weights = self.scaled_input(encoded_input=encoded_input, steps=steps, layer_idx=layer_idx,
                                               mask_idx=mask_idx)
            scaled_weights.requires_grad_(True)

            integrated_grads_this_step = []  # to store the integrated gradients

            for batch_weights in scaled_weights.chunk(n_batches):
                inputs = {
                    "input_ids": einops.repeat(
                        encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                    ),
                    "attention_mask": einops.repeat(
                        encoded_input["attention_mask"],
                        "b d -> (r b) d",
                        r=batch_size,
                    ),
                }

                # then patch the model to replace the activations with the scaled activations
                patch_ff_layer(
                    self.model,
                    layer_idx=layer_idx,
                    mask_idx=mask_idx,
                    replacement_activations=batch_weights,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

                # then forward through the model to get the logits
                outputs = self.model(**inputs)

                # then calculate the gradients for each step w/r/t the inputs
                probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                target_idx = target_label[i]
                grad = torch.autograd.grad(torch.unbind(probs[:, target_idx]), batch_weights)[0]
                grad = grad.sum(dim=0)
                integrated_grads_this_step.append(grad)

                unpatch_ff_layer(
                    self.model,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

            integrated_grads_this_step = torch.stack(
                integrated_grads_this_step, dim=0
            ).sum(dim=0)
            integrated_grads_this_step *= baseline_activations.squeeze(0) / steps

            if n_sampling_steps > 1:
                prompt += next_token_str

            D_accumulator += integrated_grads_this_step

        A_i = D_accumulator

        return A_i

    def get_refined_neurons(
            self,
            prompts,
            ground_truth,
            negative_examples=None,
            p=0.5,
            batch_size=10,
            steps=20,
            coarse_adaptive_threshold=0.3,
            coarse_threshold=None,
            coarse_percentile=None,
            quiet=False,
            baseline_vector_path=None,
            normalization=False,
            k_path=1
    ):
        """
        TempLAMA依然是没有多种表达，因此这里的refined的步骤其实是多余的。
        """
        refined_neurons = []
        for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            refined_neurons.extend(
                self.get_coarse_neurons(
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                    baseline_vector_path=baseline_vector_path,
                    normalization=normalization,
                    k_path=k_path
                )
            )
        return refined_neurons

    def compute_connection_weight(self, neuron_A, neuron_B):
        layer_A, index_A = neuron_A
        layer_B, index_B = neuron_B
        # Use precomputed weights
        effects_of_A_on_hidden = self.c_proj_weights_list[layer_A][index_A, :]
        effects_on_B_from_hidden = self.c_fc_weights_list[layer_B][:, index_B]
        weight = np.dot(effects_of_A_on_hidden, effects_on_B_from_hidden)
        # distance = 1 / weight if weight > 0 else self.inf_distance  # todo 应该考虑负值吗
        distance = abs(1 / weight) if weight != 0 else self.inf_distance
        return distance

    def compute_distance_between_neurons(self, neuron_A, neuron_B, located_neurons, max_span):
        layer_A, index_A = neuron_A
        layer_B, index_B = neuron_B

        # Direct connection case
        if abs(layer_A - layer_B) == 1:
            return self.compute_connection_weight(neuron_A, neuron_B)

        min_distance = self.inf_distance

        # Get the located neurons in the intermediate layers
        intermediate_neurons = [neuron for neuron in located_neurons if layer_A < neuron[0] < layer_B]

        # If any intermediate layer is missing a located neuron, return infinite distance
        intermediate_layers = set([neuron[0] for neuron in intermediate_neurons])
        if len(intermediate_layers) < (layer_B - layer_A - 1):
            return min_distance

        # Calculate the distance considering each possible path
        for path in itertools.combinations(intermediate_neurons, max_span - 1):
            # Ensure the path neurons are in increasing order of layers
            path = sorted(path, key=lambda x: x[0])

            # Ensure the first neuron of the path is one layer above neuron_A
            # and the last neuron of the path is one layer below neuron_B
            if path[0][0] != layer_A + 1 or path[-1][0] != layer_B - 1:
                continue

            distance = self.compute_connection_weight(neuron_A, path[0])
            for i in range(len(path) - 1):
                distance += self.compute_connection_weight(path[i], path[i + 1])
            distance += self.compute_connection_weight(path[-1], neuron_B)

            min_distance = min(min_distance, distance)

        return min_distance

    def get_connection_weight(self, neuron_i, neuron_j, located_neurons, max_span=2):
        """
        为什么同层设置为距离无穷大？因为距离代表了相互之间的影响，而同层的神经元相互之间不进行交流。
        """
        start_layer, start_index = neuron_i
        end_layer, end_index = neuron_j

        # Ensure the start neuron is in an earlier layer
        if start_layer > end_layer:
            neuron_i, neuron_j = neuron_j, neuron_i
        # Check if neurons are in the same layer
        if start_layer == end_layer:
            return self.inf_distance

        # If layer span is greater than max_span, return infinity
        if abs(start_layer - end_layer) > max_span:
            return self.inf_distance

        # Compute the distance using the intermediary neurons
        distance = self.compute_distance_between_neurons(neuron_i, neuron_j, located_neurons, max_span)
        return distance

    def get_adjacency_matrix(self, group, max_span=3):
        n = len(group)
        adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                neuron_i = group[i]
                neuron_j = group[j]
                distance = self.get_connection_weight(neuron_i, neuron_j, located_neurons=group, max_span=max_span)
                # distance = 1 / weight if weight != 0 else 0
                adjacency_matrix[i, j] = distance
                adjacency_matrix[j, i] = distance  # Assuming undirected graph
        return adjacency_matrix

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.size = [1] * n

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            rootX = self.find(x)
            rootY = self.find(y)

            if rootX != rootY:
                if self.rank[rootX] > self.rank[rootY]:
                    rootX, rootY = rootY, rootX
                self.parent[rootX] = rootY
                if self.rank[rootX] == self.rank[rootY]:
                    self.rank[rootY] += 1
                self.size[rootY] += self.size[rootX]
        def get_size(self, x):
            return self.size[self.find(x)]

    def persistent_homology_clustering(self, adjacency_matrix, knowledge_neurons,
                                       distance_quantile, persistence_quantile, fraction_of_edges):
        """第一步聚类，仅仅使用持久同调性特征
            distance_quantile, persistence_quantile,:都是分位数，设置动态阈值
        """
        num_points = adjacency_matrix.shape[0]
        uf = self.UnionFind(num_points)
        distances = adjacency_matrix[np.triu_indices(num_points, k=1)]
        finite_distances = distances[distances < self.inf_threshold]
        if finite_distances.size == 0:
            return []
        distance_threshold = np.quantile(finite_distances, distance_quantile)

        # Create a list of edges sorted by weight
        edges = [(i, j, adjacency_matrix[i, j]) for i in range(num_points) for j in range(i + 1, num_points)
                 if adjacency_matrix[i, j] < distance_threshold]
        edges.sort(key=lambda x: x[2])
        if not edges:
            print(f"No edges found with the current distance threshold: {distance_threshold}")
            return []

        # Track birth times of components
        birth_time = {i: 0 for i in range(num_points)}
        # Use only a fraction of the shortest edges for clustering
        num_edges_to_use = int(fraction_of_edges * len(edges))
        for i, j, weight in edges[:num_edges_to_use]:
            if uf.find(i) != uf.find(j):
                # If the two clusters have not been merged yet
                if uf.get_size(i) == 1:
                    birth_time[i] = weight
                if uf.get_size(j) == 1:
                    birth_time[j] = weight
                uf.union(i, j)

        # Extract clusters with persistence greater than the threshold
        clusters = {}
        for i in range(num_points):
            root = uf.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        birth_times = np.array(list(birth_time.values()))

        if birth_times.size == 0:
            return []
        persistence_threshold = np.quantile(birth_times, persistence_quantile)  # median of the birth times

        # Filter out clusters with low persistence
        filtered_clusters = {}
        for root, points in clusters.items():
            persistence_time = edges[-1][2] - birth_time[root]
            if persistence_time > persistence_threshold and len(points) > 1:
                filtered_clusters[root] = points
        dkn_cluster_1 = []
        for cluster in filtered_clusters.values():
            neuron_positions = [knowledge_neurons[i] for i in cluster]
            dkn_cluster_1.append(neuron_positions)

        return dkn_cluster_1

    def hierarchical_clustering(self, adjacency_matrix, knowledge_neurons, percentile_threshold):
        try:
            # Ensure the diagonal is zero
            np.fill_diagonal(adjacency_matrix, 0)

            # Check if adjacency_matrix is not empty
            if adjacency_matrix.size == 0:
                print("Adjacency matrix is empty")
                return []

            # Convert the adjacency matrix to a condensed distance matrix
            condensed_matrix = squareform(adjacency_matrix, checks=False)

            # Check if condensed_matrix is empty
            if condensed_matrix.size == 0:
                print("Condensed matrix is empty")
                return []

            # Check percentile threshold
            if not (0 <= percentile_threshold <= 1):
                print(f"Invalid percentile_threshold: {percentile_threshold}")
                return []

            # Calculate the distance threshold based on percentile
            distance_threshold = np.quantile(condensed_matrix, percentile_threshold)

            # Hierarchical clustering using the condensed distance matrix
            Z = linkage(condensed_matrix, method='average')

            # Form clusters
            labels = fcluster(Z, distance_threshold, criterion='distance')

            # Cluster the neurons based on the clustering results
            clustered_neurons = [[] for _ in range(max(labels))]
            for neuron_idx, cluster_label in enumerate(labels):
                clustered_neurons[cluster_label - 1].append(knowledge_neurons[neuron_idx])

            return clustered_neurons

        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def dbscan_clustering(self, adjacency_matrix, knowledge_neurons, percentile_eps):
        if len(knowledge_neurons) != adjacency_matrix.shape[0]:
            raise ValueError("Length of knowledge_neurons must match the dimensions of adjacency_matrix")

        # Calculate epsilon value for DBSCAN
        # nearest_neighbor_distances = np.sort([np.min(row[np.nonzero(row)]) for row in adjacency_matrix])
        # Calculate epsilon value for DBSCAN
        nearest_neighbor_distances = []
        for row in adjacency_matrix:
            non_zero_elements = row[np.nonzero(row)]
            if non_zero_elements.size > 0:
                nearest_neighbor_distances.append(np.min(non_zero_elements))

        # Check if nearest_neighbor_distances is empty
        if not nearest_neighbor_distances:
            return []

        nearest_neighbor_distances = np.sort(nearest_neighbor_distances)
        eps = np.quantile(nearest_neighbor_distances, percentile_eps)
        min_samples = max(2, int(len(adjacency_matrix) * 0.03))  # For example, setting it to 3%

        # Perform DBSCAN Clustering with precomputed distances
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(adjacency_matrix)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

        # Initialize lists to hold clustered neurons
        clustered_neurons = {i: [] for i in range(n_clusters)}
        for neuron_idx, label in enumerate(clustering.labels_):
            if label != -1:
                clustered_neurons[label].append(knowledge_neurons[neuron_idx])

        # Convert dictionary to list
        return list(clustered_neurons.values())

    def kmeans_clustering(self, adjacency_matrix, knowledge_neurons, variance_threshold=0.95):
        try:
            if len(knowledge_neurons) != adjacency_matrix.shape[0]:
                print("Length of knowledge_neurons does not match the dimensions of adjacency_matrix")
                return []

            # Check if adjacency_matrix is empty or too small
            if adjacency_matrix.size == 0 or len(adjacency_matrix) < 2:
                print("Adjacency matrix is empty or too small for clustering")
                return []

            # Dimensionality Reduction using PCA
            n_clusters = max(2, int(len(adjacency_matrix) * 0.1))
            pca4components = PCA().fit(adjacency_matrix)
            cumulative_variance = np.cumsum(pca4components.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(adjacency_matrix)

            # Check if the reduced features are suitable for KMeans
            if len(reduced_features) < n_clusters:
                print("Not enough features for the number of clusters")
                return []

            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reduced_features)

            # Initialize lists to hold clustered neurons
            clustered_neurons = {i: [] for i in range(n_clusters)}

            # Assign each neuron to a cluster
            for index, label in enumerate(kmeans.labels_):
                clustered_neurons[label].append(knowledge_neurons[index])

            # Convert dictionary to list
            return list(clustered_neurons.values())

        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def get_dkn_src(self, prompt, ground_truth, all_neurons, threshold_low, threshold_high):
        """
        Method based on original assumptions. Structural properties are not considered.
        """
        # todo 原来都是针对某一个query进行调整。
        dk_neurons_src = []
        used_neurons = set()  # Set to keep track of used neurons

        # First pass: identify neurons whose removal doesn't impact accuracy significantly
        potential_neurons = []
        original_accuracy = self.get_predict_acc(prompt, ground_truth, [])
        if len(all_neurons) > 15:
            all_neurons = all_neurons[:15]
        for neuron in all_neurons:
            neuron_tuple = tuple(neuron)  # Convert neuron list to tuple
            if neuron_tuple not in used_neurons:  # Check if the neuron has already been used
                neurons_to_suppress = [neuron]
                new_accuracy = self.get_predict_acc(prompt, ground_truth, neurons_to_suppress)

                T1 = (original_accuracy - new_accuracy) / original_accuracy
                if T1 < threshold_low:
                    potential_neurons.append(neuron)

        # Second pass: dynamically identify pairs and update potential neurons
        while potential_neurons:
            for pair in itertools.combinations(potential_neurons, 2):
                pair_tuple = tuple(map(tuple, pair))
                if all(neuron_tuple not in used_neurons for neuron_tuple in pair_tuple):
                    neurons_to_suppress = list(pair)
                    new_accuracy_2 = self.get_predict_acc(prompt, ground_truth, neurons_to_suppress)
                    T2 = (original_accuracy - new_accuracy_2) / original_accuracy

                    if T2 > threshold_high:
                        dk_neurons_src.append(neurons_to_suppress)
                        used_neurons.update(pair_tuple)
                    elif T2 < threshold_low:
                        # Remove individual neurons from potential neurons
                        for neuron in pair:
                            if neuron in potential_neurons:
                                potential_neurons.remove(neuron)
                        break  # Break to regenerate combinations with updated potential_neurons
            else:
                # Exit the while loop if no more combinations can be formed
                break

        return dk_neurons_src

    def filter_neuron_sets(self, prompt, ground_truth, dkn_cluster_1, threshold_low, use_predict_acc_2=False):
        dkn_cluster_2 = []
        if len(dkn_cluster_1) <= 1:
            return dkn_cluster_2, {}, 0
        if use_predict_acc_2:
            original_acc = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth, neurons=[])
        else:
            original_acc = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth, neurons=[])

        # Step 1: Evaluate each set in dkn_cluster_1 individually
        #如果ABCD都是知识神经元，只不过有一部分不是简并知识神经元。那么它们的性质就是，不能单独的表达这个事实知识。那么，我只需要对每一个进行研究即可，即限定size=1，这样我在过滤时也容易执行。
        for neuron_set in dkn_cluster_1:
            if use_predict_acc_2:
                accuracy_1 = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth, neurons=neuron_set)
            else:
                accuracy_1 = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth, neurons=neuron_set)

            accuracy_drop_1 = (original_acc - accuracy_1) / original_acc
            if accuracy_drop_1 < threshold_low:
                # todo 负值可以解释。（即抑制神经元，准确率反而上升）：
                #  1.因此抑制一个途径可能会导致激活另一条可能更有效或噪音更少的途径，以实现正确的响应。
                #  2.简并的代偿性质，使得剩余的神经元更加活跃。
                #  3.神经元本身的定位结果是不够准确的，但是这个问题直接从定位角度出发难以解决（甚至于难以评价），反而可以从简并和过滤的角度出发进行解决。
                dkn_cluster_2.append(neuron_set)

        # Step 2: Traverse all proper subsets of dkn_cluster_2 and record accuracy drops
        dkn_acc_drops = {}
        for size in range(1, len(dkn_cluster_2)):
            for subset in itertools.combinations(dkn_cluster_2, size):
                suppressed_neurons_2 = [neuron for group in subset for neuron in group]
                if use_predict_acc_2:
                    accuracy_2 = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth, neurons=suppressed_neurons_2)
                else:
                    accuracy_2 = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth, neurons=suppressed_neurons_2)
                accuracy_drop_2 = (original_acc - accuracy_2) / original_acc
                dkn_acc_drops.setdefault(size, []).append({'subset': subset, 'accuracy_drop': accuracy_drop_2})
        all_neurons_in_dkn_cluster_2 = [neuron for subset in dkn_cluster_2 for neuron in subset]
        if use_predict_acc_2:
            all_neurons_accuracy = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth,
                                                        neurons=all_neurons_in_dkn_cluster_2)
        else:
            all_neurons_accuracy = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth,
                                                        neurons=all_neurons_in_dkn_cluster_2)
        all_neurons_acc_drop = (original_acc - all_neurons_accuracy) / original_acc
        return dkn_cluster_2, dkn_acc_drops, all_neurons_acc_drop


    def filter_neuron_sets_second_only(self, prompt, ground_truth, dkn_cluster_2, use_predict_acc_2=False):
        original_acc = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth, neurons=[])
        dkn_acc_drops = {}
        for size in range(1, len(dkn_cluster_2)):
            for subset in itertools.combinations(dkn_cluster_2, size):
                suppressed_neurons_2 = [neuron for group in subset for neuron in group]
                if use_predict_acc_2:
                    accuracy_2 = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth, neurons=suppressed_neurons_2)
                else:
                    accuracy_2 = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth, neurons=suppressed_neurons_2)
                accuracy_drop_2 = (original_acc - accuracy_2) / original_acc
                dkn_acc_drops.setdefault(size, []).append({'subset': subset, 'accuracy_drop': accuracy_drop_2})
        all_neurons_in_dkn_cluster_2 = [neuron for subset in dkn_cluster_2 for neuron in subset]
        if use_predict_acc_2:
            all_neurons_accuracy = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth,
                                                        neurons=all_neurons_in_dkn_cluster_2)
        else:
            all_neurons_accuracy = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth,
                                                        neurons=all_neurons_in_dkn_cluster_2)
        all_neurons_acc_drop = (original_acc - all_neurons_accuracy) / original_acc
        return dkn_acc_drops, all_neurons_acc_drop


    def filter_neuron_sets_only_first(self, prompt, ground_truth, dkn_cluster_1, threshold_low, use_predict_acc_2=False):
        dkn_cluster_2 = []
        if len(dkn_cluster_1) <= 1:
            return dkn_cluster_2
        if use_predict_acc_2:
            original_acc = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth, neurons=[])
        else:
            original_acc = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth, neurons=[])

        # Step 1: Evaluate each set in dkn_cluster_1 individually
        # 如果ABCD都是知识神经元，只不过有一部分不是简并知识神经元。那么它们的性质就是，不能单独的表达这个事实知识。那么，我只需要对每一个进行研究即可，即限定size=1，这样我在过滤时也容易执行。
        for neuron_set in dkn_cluster_1:
            if len(neuron_set) == 1:
                continue
            if use_predict_acc_2:
                accuracy_1 = self.get_predict_acc_2(prompt=prompt, ground_truth=ground_truth, neurons=neuron_set)
            else:
                accuracy_1 = self.get_predict_acc(prompt=prompt, ground_truth=ground_truth, neurons=neuron_set)

            accuracy_drop_1 = (original_acc - accuracy_1) / original_acc
            if accuracy_drop_1 < threshold_low:
                dkn_cluster_2.append(neuron_set)

        return dkn_cluster_2

    def zero_out_weights(self, neurons):
        self.original_weights = {}
        # for sublist in neurons:
        for layer_index, neuron_index in neurons:
            # Store original weights and biases
            if self.model_type == 'llama':
                self.original_weights[(layer_index, neuron_index)] = (
                    self.model.model.layers[layer_index].mlp.down_proj.weight.data[:, neuron_index].clone(),
                    self.model.model.layers[layer_index].mlp.gate_proj.weight.data[neuron_index, :].clone()
                )
                # Zero out weights
                self.model.model.layers[layer_index].mlp.down_proj.weight.data[:, neuron_index] = torch.zeros_like(self.model.model.layers[layer_index].mlp.down_proj.weight[:, neuron_index])
                self.model.model.layers[layer_index].mlp.gate_proj.weight.data[neuron_index, :] = torch.zeros_like(self.model.model.layers[layer_index].mlp.gate_proj.weight[neuron_index, :])
                # Zero out biases is not directly applicable here
            else:
                self.original_weights[(layer_index, neuron_index)] = (
                    self.model.transformer.h[layer_index].mlp.c_fc.weight.data[:, neuron_index].clone(),
                    self.model.transformer.h[layer_index].mlp.c_proj.weight.data[neuron_index, :].clone()
                )
                # Zero out weights
                self.model.transformer.h[layer_index].mlp.c_fc.weight.data[:, neuron_index] = torch.zeros_like(self.model.transformer.h[layer_index].mlp.c_fc.weight[:, neuron_index])
                self.model.transformer.h[layer_index].mlp.c_proj.weight.data[neuron_index, :] = torch.zeros_like(self.model.transformer.h[layer_index].mlp.c_proj.weight[neuron_index, :])
                    # Zero out biases is not directly applicable here

    def change_weights(self, neurons, value):
        self.original_weights = {}

        for layer_index, neuron_index in neurons:
            # Store original weights and biases
            if self.model_type == 'llama':
                original_down_proj_weights = self.model.model.layers[layer_index].mlp.down_proj.weight.data[:, neuron_index].clone()
                original_gate_proj_weights = self.model.model.layers[layer_index].mlp.gate_proj.weight.data[neuron_index, :].clone()

                self.original_weights[(layer_index, neuron_index)] = (
                    original_down_proj_weights,
                    original_gate_proj_weights
                )

                # Adjust weights to 2 times the original values
                self.model.model.layers[layer_index].mlp.down_proj.weight.data[:, neuron_index] = value * original_down_proj_weights
                self.model.model.layers[layer_index].mlp.gate_proj.weight.data[neuron_index, :] = value * original_gate_proj_weights
            else:
                original_c_fc_weights = self.model.transformer.h[layer_index].mlp.c_fc.weight.data[:, neuron_index].clone()
                original_c_proj_weights = self.model.transformer.h[layer_index].mlp.c_proj.weight.data[neuron_index, :].clone()

                self.original_weights[(layer_index, neuron_index)] = (
                    original_c_fc_weights,
                    original_c_proj_weights
                )

                # Adjust weights to 2 times the original values
                self.model.transformer.h[layer_index].mlp.c_fc.weight.data[:, neuron_index] = value * original_c_fc_weights
                self.model.transformer.h[layer_index].mlp.c_proj.weight.data[neuron_index, :] = value * original_c_proj_weights

    def restore_weights(self):
        for (layer_index, neuron_index), (fc_weight, proj_weight) in self.original_weights.items():
            if self.model_type == 'llama':
                self.model.model.layers[layer_index].mlp.down_proj.weight.data[:, neuron_index] = fc_weight
                self.model.model.layers[layer_index].mlp.gate_proj.weight.data[neuron_index, :] = proj_weight
            else:
                self.model.transformer.h[layer_index].mlp.c_fc.weight.data[:, neuron_index] = fc_weight
                self.model.transformer.h[layer_index].mlp.c_proj.weight.data[neuron_index, :] = proj_weight
        self.original_weights = {}


    def get_predict_acc_2(self, prompt, ground_truth, neurons, mode='suppress'):
        _, mask_idx, _, prompt = self._prepare_inputs(prompt, ground_truth)

        # Patch the model to suppress neurons
        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # Get the probability of the ground truth
        # combined_input = prompt + ' ' + ground_truth
        # Tokenize the combined input

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        ground_truth_tokens = self.tokenizer.encode(ground_truth, add_special_tokens=False)

        # Prepare input for the model
        input_ids = torch.tensor([prompt_tokens + ground_truth_tokens]).to(self.device)
        # inputs = self.tokenizer(combined_input, return_tensors="pt").to(self.device)
        # Get model output
        with torch.no_grad():
            # outputs = self.model(**inputs)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits

        # Calculate the probability of each token in the ground truth sequence
        gt_probs = []
        for i, gt_token_id in enumerate(ground_truth_tokens):
            # Position of the token in the logits is after the prompt
            token_idx = len(prompt_tokens) + i

            # Convert logits to probabilities
            probs = torch.softmax(logits[0, token_idx], dim=-1)
            token_prob = probs[gt_token_id].item()

            # Decode for debugging
            # decoded_token = self.tokenizer.decode([gt_token_id])
            gt_probs.append(token_prob)

        # Compute the joint probability of the ground truth sequence
        new_gt_prob = math.prod(gt_probs)
        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=set([n[0] for n in neurons]),
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
        unpatch_fn()
        unpatch_fn = lambda *args: args

        return new_gt_prob


    def get_score_for_neuron(self, prompt, ground_truth, layer_idx, neuron_idx, batch_size=10, steps=20,
                             baseline_vector_path=None):
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(prompt, ground_truth)

        # Scaling weights
        scaled_weights = self.scaled_input(encoded_input=encoded_input, steps=steps, layer_idx=layer_idx,
                                           mask_idx=mask_idx)
        scaled_weights.requires_grad_(True)

        integrated_grads = []

        for batch_weights in scaled_weights.chunk(n_batches):
            # Patching the model for scaled activations
            patch_ff_layer(self.model, layer_idx=layer_idx, mask_idx=mask_idx, replacement_activations=batch_weights,
                           transformer_layers_attr=self.transformer_layers_attr, ff_attrs=self.input_ff_attr)

            # Forward pass
            outputs = self.model(**{
                "input_ids": einops.repeat(encoded_input["input_ids"], "b d -> (r b) d", r=batch_size),
                "attention_mask": einops.repeat(encoded_input["attention_mask"], "b d -> (r b) d", r=batch_size),
                # Add other inputs needed by your model
            })

            # Computing gradients
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[0]  # Assuming single target label for simplicity
            grad = torch.autograd.grad(torch.unbind(probs[:, target_idx]), batch_weights)[0]

            # Extracting the gradient for the specific neuron
            neuron_grad = grad[:, neuron_idx].sum(dim=0)
            integrated_grads.append(neuron_grad)

            # Unpatching the model
            unpatch_ff_layer(self.model, layer_idx=layer_idx, transformer_layers_attr=self.transformer_layers_attr,
                             ff_attrs=self.input_ff_attr)
        integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
            integrated_grads
        )
        return integrated_grads

    def if_fact_wrong_new(self, prompt, label, batch_size, steps, baseline_vector_path, neurons, threshold, relation_name,
                      score_path):
        sn_scores = []

        for neuron in neurons:
            # Calculate score for each specified neuron
            score = self.get_score_for_neuron(prompt=prompt, ground_truth=label, layer_idx=neuron[0],neuron_idx=neuron[1], batch_size=batch_size, steps=steps,baseline_vector_path=baseline_vector_path)
            sn_scores.append(score)

        # Convert list of scores to a tensor
        sn_scores = torch.tensor(sn_scores)

        # Check if the mean score is below the threshold
        if sn_scores.mean().item() < threshold:
            return True
        else:
            return False

    def if_fact_wrong(self, prompt, label,
                      batch_size,
                      steps,
                      baseline_vector_path,
                      neurons,
                      threshold,
                      relation_name,
                      score_path, ):
        scores = self.get_scores(prompt=prompt, ground_truth=label, batch_size=batch_size,
                                 steps=steps, baseline_vector_path=baseline_vector_path, )

        sn_scores = []
        for neuron in neurons:
            score = scores[neuron[0], neuron[1]]
            sn_scores.append(score)

        sn_scores = torch.tensor(sn_scores)
        if sn_scores.mean().item() < threshold:
            return True
        else:
            return False

    def test_detection_system(self, item, neurons, threshold,
                              batch_size, steps, baseline_vector_path, score_path):
        # Test the correct fact
        prompt = item['sentences'][0]
        true_label = item['obj_label']
        relation_name = item['relation_name']
        if self.if_fact_wrong(prompt=prompt, label=true_label,
                                  neurons=neurons,
                                  threshold=threshold, batch_size=batch_size, steps=steps,
                                  baseline_vector_path=baseline_vector_path,
                                  relation_name=f'True_{relation_name}',
                                  score_path=score_path):
            correct_true = 0
        else:
            correct_true = 1  # 正确结果判断为正：1

        # Randomly select and test one incorrect fact
        random.seed(42)
        wrong_label = random.choice(item['wrong_fact'])
        if self.if_fact_wrong(prompt=prompt, label=wrong_label, neurons=neurons,
                              threshold=threshold, batch_size=batch_size, steps=steps,
                              baseline_vector_path=baseline_vector_path,
                              relation_name=f'False_{relation_name}',
                              score_path=score_path):
            correct_false = 1  # 错误结果判断为负：1
        else:
            correct_false = 0
        return correct_true, correct_false

    def test_detection_system_PLMs(self, item):
        # Format the input item into two statements
        correct_statement = None
        wrong_statement = None
        if '_X_' in item['sentences'][0]:
            correct_statement = item['sentences'][0].replace('_X_', item['obj_label'])
            wrong_statement = item['sentences'][0].replace('_X_', item['wrong_fact'][0])
        elif 'MASK' in item['sentences'][0]:
            correct_statement = item['sentences'][0].replace('[MASK]', item['obj_label'])
            wrong_statement = item['sentences'][0].replace('[MASK]', item['wrong_fact'][0])
        else:
            return 0, 0

        def get_model_judgement(statement, is_correct):
            prompt = f"Statement: {statement}\nIs this true or false? Just answer 'true' or answer 'false'."
            response = get_model_output(model=self.model, tokenizer=self.tokenizer, prompt=prompt)
            response = response.split('\n')[-1].strip().lower()
            if is_correct:
                return 1 if 'true' in response.lower() else 0
            else:
                return 1 if 'false' in response.lower() else 0
        # Get GPT-2's judgement for both statements
        correct_judgement = get_model_judgement(correct_statement, True)
        wrong_judgement = get_model_judgement(wrong_statement, False)

        return correct_judgement, wrong_judgement


    def enhance_or_suppress_dkn_predict_answer(self, query, correct_answer, neurons_list_2d, mode='enhance'):
        _, mask_idx, _ , query= self._prepare_inputs(query, correct_answer)
        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons_list_2d,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        predicted_answer = get_model_output(model=self.model, tokenizer=self.tokenizer, prompt=query)

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=set([n[0] for n in neurons_list_2d]),
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
        unpatch_fn()
        # unpatch_fn = lambda *args: args

        return predicted_answer

    def change_weights_dkn_predict_answer(self, query, neurons_list_2d, value=0):
        self.change_weights(neurons=neurons_list_2d, value=value)

        predicted_answer = get_model_output(model=self.model, tokenizer=self.tokenizer, prompt=query)
        self.restore_weights()
        return predicted_answer


