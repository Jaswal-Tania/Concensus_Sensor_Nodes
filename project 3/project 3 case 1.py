# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Define constants
iteration = 200  # Number of iterations
num_node = 50  # Number of nodes
dimension = 2  # Dimension of each node
area = 3  # Area limit for random node generation
variance = 2  # Variance for measurement noise
gt = 50  # Ground truth value

# Generate random node positions within the specified area
nodes = np.random.uniform(size=(num_node, dimension), high=area)

# Generate ground truth array
gt_array = np.full((num_node, 1), gt)

# Generate noisy measurements
measure = gt + np.random.uniform(size=(num_node, 1), high=variance, low=-variance)

# Initialize array to store measurements for each iteration
measure_all = np.zeros([num_node, iteration])

# Initialize weight matrix
w = np.zeros([num_node, num_node])

# Initialize initial active range
r = 1


# Function to compute adjacency matrix and degree of neighbors
def num_neighbors():
    # Compute distance matrix between nodes
    matrix = distance_matrix(nodes, nodes)

    # Convert distances to adjacency matrix
    matrix[(matrix <= r) & (matrix > 0)] = 1
    matrix[np.where(matrix != 1)] = 0

    # Compute degree of neighbors
    return matrix, np.sum(matrix, axis=1)


# Function to plot node connections
def plot_neighbors():
    plt.plot(nodes[:, 0], nodes[:, 1], 'o', color='black')
    for i in range(0, num_node):
        for j in range(0, num_node):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= r:
                plt.plot([nodes[i, 0], nodes[j, 0]],
                         [nodes[i, 1], nodes[j, 1]],
                         'b-', lw=0.5)
    plt.show()


# Function to compute weights using max degree
def max_degree_weights(i, j, adj_matrix, degree_neighbors):
    if adj_matrix[i, j] == 1:
        return 1 / num_node
    elif i == j:
        return 1 - degree_neighbors[i] / num_node
    return 0


# Function to compute weights using Metropolis criterion
def metropolis(i, j, adj_matrix, degree_neighbors):
    if i == j:
        return 1 - np.sum([1 / (1 + np.maximum(degree_neighbors[i], degree_neighbors[k]))
                           for k in range(num_node) if adj_matrix[i, k] == 1])
    elif adj_matrix[i, j] == 1:
        return 1 / (1 + np.maximum(degree_neighbors[i], degree_neighbors[j]))
    return 0


# Function to plot initial and final measurements
def plot_value(measure):
    x = np.arange(0, num_node)
    plt.plot(x, measure, 'o', color='black')
    plt.plot(x, measure, color='black')
    plt.plot(x, gt_array, 'o', color='blue')
    plt.plot(x, gt_array, color='blue')
    plt.show()


# Function to plot convergence error
def plot_convergence_error():
    x = np.arange(0, iteration)
    plt.title("Error Convergence")
    for i in range(0, num_node):
        plt.plot(x, gt - measure_all[i, :])
    plt.show()
    plt.title("Mean Square Error Convergence")
    for i in range(0, num_node):
        plt.plot(x, (gt - measure_all[i, :]) ** 2)
    plt.show()


# Function to plot convergence for max and min neighbor errors
def plot_min_max(degree_neighbors):
    x = np.arange(0, iteration)
    plt.title("Max and Min Neighbor Error Convergence")
    plt.plot(x, (gt - measure_all[np.argmax(degree_neighbors), :]), label="Max Neighbor")
    plt.plot(x, (gt - measure_all[np.argmin(degree_neighbors), :]), label="Min Neighbor")
    leg = plt.legend(loc='upper right')
    plt.show()
    plt.title("Max and Min Neighbor Mean Square Error Convergence")
    plt.plot(x, (gt - measure_all[np.argmax(degree_neighbors), :]) ** 2, label="Max Neighbor")
    plt.plot(x, (gt - measure_all[np.argmin(degree_neighbors), :]) ** 2, label="Min Neighbor")
    leg = plt.legend(loc='upper right')
    plt.show()


# Main function
def main():
    # Compute adjacency matrix and degree of neighbors
    adj_matrix, degree_neighbors = num_neighbors()

    # Compute weights using Metropolis criterion
    for i in range(num_node):
        for j in range(num_node):
            w[i, j] = metropolis(i, j, adj_matrix, degree_neighbors)
            # Uncomment the following line to use Max-Degree weights instead
            # w[i, j] = max_degree_weights(i, j, adj_matrix, degree_neighbors)

    # Perform iterations
    for iter in range(iteration):
        if iter == 0:
            # Plot initial node connections and measurements
            plot_neighbors()
            for n in range(0, num_node):
                measure_all[n, iter] = measure[n][0]
            plot_value(measure)
        else:
            # Update measurements using weights and previous measurements
            for i in range(num_node):
                total = 0
                if degree_neighbors[n] != 0:
                    for j in range(num_node):
                        if np.linalg.norm(nodes[j] - nodes[i]) < r and i != j:
                            total += w[i, j] * measure_all[j, iter - 1]
                    adj_n = adj_matrix[i]
                    weight_n = w[i]
                    neighbor_weight = weight_n[adj_n == 1]
                    neighbor_measure = (((measure_all[:, iter - 1])[:, np.newaxis])[adj_n == 1])[:, 0]
                    element_mul = neighbor_weight * neighbor_measure
                    sum1 = np.sum(element_mul)
                    new_measure = measure_all[i, iter - 1] * w[i, i] + total
                    measure[i] = new_measure
                    measure_all[i, iter] = new_measure

    # Plot final measurements and convergence plots
    plot_value(measure_all[:, -1])
    plot_convergence_error()
    plot_min_max(degree_neighbors)


# Execute main function
if __name__ == "__main__":
    main()
