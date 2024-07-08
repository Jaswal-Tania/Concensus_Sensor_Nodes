import matplotlib.pyplot as plt
import numpy as np

# Constants
numNodes = 50  # Number of sensor nodes
r = 15  # Active range of sensor nodes
area = 50  # Distribution area of sensor nodes
iteration = 300  # Number of iterations
neighborList = [0] * numNodes  # List to store neighbors of each node
V1 = [0] * numNodes  # List to store noise variance for each node
n1 = [0] * numNodes  # List to store noise values for each node
zi = [0] * numNodes  # List to store noisy measurements for each node
errorc1 = np.zeros([iteration, numNodes])  # Array to store errors for consensus 1
errorc2 = np.zeros([iteration, numNodes])  # Array to store errors for consensus 2
errorAvgc1 = np.zeros([iteration, numNodes])  # Array to store errors between average for consensus 1
errorAvgc2 = np.zeros([iteration, numNodes])  # Array to store errors between average for consensus 2
groundTruth = 40  # Ground truth value


# Function to find neighbors
def find_neighbors(n, Xi, r):
    # Find neighbors within a specified radius for each node.
    allNi = [[] for _ in range(n)]
    for j in range(n):
        neighbors = []
        for i in range(n):
            if np.linalg.norm(Xi[j] - Xi[i]) <= r:
                neighbors.append(i)
        allNi[j] = neighbors
    return allNi


# Consensus function 1
def consensus1(Xi, Ni, A, e):
    # Perform consensus operation for each node using Consensus 1.
    newXi = np.zeros_like(Xi)
    for i in range(len(Xi)):
        sum_neighbors = sum(A[i, j] * (Xi[j] - Xi[i]) for j in Ni[i])
        newXi[i] = Xi[i] + e * sum_neighbors
    return newXi


# Consensus function 2
def consensus2(Xi, Ni):
    # Perform consensus operation for each node using Consensus 2.
    newXi = np.zeros_like(Xi)
    for i in range(len(Xi)):
        sum_neighbors = sum(Xi[j] for j in Ni[i])
        newXi[i] = (1 / (1 + len(Ni[i]))) * (Xi[i] + sum_neighbors)
    return newXi


# Function to create adjacency matrix
def adjacency_matrix(Ni):
    # Create adjacency matrix based on neighbor lists.
    AMatrix = np.zeros((numNodes, numNodes))
    for i in range(len(Ni)):
        for j in Ni[i]:
            AMatrix[i][j] = 1
    return AMatrix


def main():
    # Generate random values
    Xi = np.random.random((numNodes, 2)) * area

    # Set epsilon step-rate
    e = 0.02

    # Target position
    target = np.array([25, 25])

    # Find neighbors and store in a list
    neighborList = find_neighbors(numNodes, Xi, r)

    # Make noisy measurement
    for j in range(numNodes):
        cov_mat = np.stack((Xi[j], target), axis=1)
        Calccov = np.cov(cov_mat)
        cv = (Calccov[0, 0] - Calccov[1, 1]) / 100

        V1[j] = ((np.linalg.norm(Xi[j] - target) ** 2) + cv) / (r ** 2)
        n1[j] = np.random.normal(0.0, V1[j])
        zi[j] = groundTruth + n1[j]

    # Initial measurement
    xic1 = zi.copy()
    xic2 = zi.copy()
    initial_average = np.mean(zi)

    iter = 0
    while iter < iteration:
        # Update adjacency matrix
        A = adjacency_matrix(neighborList)

        # Consensus 1 and 2
        xic1 = consensus1(xic1, neighborList, A, e)
        xic2 = consensus2(xic2, neighborList)

        for x in range(numNodes):
            # Record error
            errorc1[iter, x] = xic1[x] - groundTruth
            errorc2[iter, x] = xic2[x] - groundTruth
            errorAvgc1[iter, x] = xic1[x] - initial_average
            errorAvgc2[iter, x] = xic2[x] - initial_average

        iter += 1

    # Plot the errors and initial vs final measurements
    plt.figure(figsize=(10, 8))

    # Plot error consensus 1
    plt.subplot(3, 2, 1)
    plt.plot(errorc1)
    plt.title("Error of all nodes consensus 1")

    # Plot error consensus 2
    plt.subplot(3, 2, 2)
    plt.plot(errorc2)
    plt.title("Error of all nodes consensus 2")

    # Plot error between average consensus 1
    plt.subplot(3, 2, 3)
    plt.plot(errorAvgc1)
    plt.title("Error between average of all nodes consensus 1")

    # Plot error between average consensus 2
    plt.subplot(3, 2, 4)
    plt.plot(errorAvgc2)
    plt.title("Error between average of all nodes consensus 2")

    # Plot initial and final measurement consensus 1
    plt.subplot(3, 2, 5)
    plt.plot(zi, label='initial', color='blue', marker='o', markerfacecolor='blue', markersize=4)
    plt.plot(xic1, label='last', color='orange', marker='o', markerfacecolor='orange', markersize=4)
    plt.legend()
    plt.title("Initial and final measurement consensus 1")

    # Plot initial and final measurement consensus 2
    plt.subplot(3, 2, 6)
    plt.plot(zi, label='initial', color='blue', marker='o', markerfacecolor='blue', markersize=4)
    plt.plot(xic2, label='last', color='orange', marker='o', markerfacecolor='orange', markersize=4)
    plt.legend()
    plt.title("Initial and final measurement consensus 2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
