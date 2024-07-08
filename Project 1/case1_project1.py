# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters definition
X = 50  # Range of X-axis
Y = 50  # Range of Y-axis
EPSILON = 0.1  # Small positive constant
H = 0.2  # Threshold parameter
C1_ALPHA = 30  # Coefficient for calculating new position
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)  # Coefficient for calculating new velocity
N = 100  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor nodes
K = 1.2  # Scaling factor
R = K * D  # Interaction range
DELTA_T = 0.009  # Time step
A = 5  # Parameter for potential function
B = 5  # Parameter for potential function
C = np.abs(A - B) / np.sqrt(4 * A * B)  # Parameter for potential function
ITERATION = 300  # Number of iterations
SNAPSHOT_INTERVAL = 20  # Interval for snapshot
POSITION_X = np.zeros([N, ITERATION])  # Array to store X positions of nodes
POSITION_Y = np.zeros([N, ITERATION])  # Array to store Y positions of nodes

# Generating random initial positions for sensor nodes
nodes = np.random.rand(N, M) * X
nodes_old = nodes  # Copy of initial node positions
nodes_velocity_p = np.zeros([N, M])  # Initial velocities of nodes
a_ij_matrix = np.zeros([N, N])  # Adjacency matrix for the network
velocity_magnitudes = np.zeros([N, ITERATION])  # Array to store velocity magnitudes of nodes
connectivity = np.zeros([ITERATION, 1])  # Array to store connectivity at each iteration
fig = plt.figure()  # Figure for plotting
fig_counter = 0  # Counter for figures


# Function to create adjacency matrix for the network
def create_adjacency_matrix():
    adjacency_matrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                distance = np.linalg.norm(nodes[i] - nodes[j])
                if distance <= R:
                    adjacency_matrix[i, j] = 1
    return adjacency_matrix


# Function to plot initial deployment of sensor nodes
def plot_deployment():
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    plt.show()


# Function to plot neighbors within interaction range
def plot_neighbors():
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                plt.plot([nodes[i, 0], nodes[j, 0]],
                         [nodes[i, 1], nodes[j, 1]],
                         'b-', lw=2)
    plt.show()


# Functions defining mathematical operations and potential functions
def sigma_norm(z):
    val = EPSILON * (z ** 2)
    val = np.sqrt(1 + val) - 1
    val = val / EPSILON
    return val


def bump_function(z):
    if 0 <= z < H:
        return 1
    elif H <= z < 1:
        val = (z - H) / (1 - H)
        val = np.cos(np.pi * val)
        val = (1 + val) / 2
        return val
    else:
        return 0


def sigma_1(z):
    val = 1 + z ** 2
    val = np.sqrt(val)
    val = z / val
    return val


def phi(z):
    val_1 = A + B
    val_2 = sigma_1(z + C)
    val_3 = A - B
    val = val_1 * val_2 + val_3
    val = val / 2
    return val


def phi_alpha(z):
    input_1 = z / sigma_norm(R)  # Sigma norm of R is R_alpha
    input_2 = z - sigma_norm(D)  # Sigma norm of D is D_alpha
    val_1 = bump_function(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val


# Functions to calculate parameters for movement and position updates
def get_a_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm) / sigma_norm(R)
    val = bump_function(val_2)
    return val


def get_n_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm ** 2
    val = val_1 / np.sqrt(val_2)
    return val


def get_u_i(i):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        if i == j:
            pass
        else:
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                val_1 = nodes[j] - nodes[i]
                norm = np.linalg.norm(val_1)
                phi_alpha_val = phi_alpha(sigma_norm(norm))
                val = phi_alpha_val * get_n_ij(i, j)
                sum_1 += val

                val_2 = nodes_velocity_p[j] - nodes_velocity_p[i]
                sum_2 += get_a_ij(i, j) * val_2
    val = C1_ALPHA * sum_1 + C2_ALPHA * sum_2
    return val


# Function to update positions of nodes
def get_positions():
    for t in range(0, ITERATION):
        adjacency_matrix = create_adjacency_matrix()
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)
        if t == 0:
            plot_neighbors()
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
        else:
            for i in range(0, N):
                u_i = get_u_i(i)
                old_velocity = nodes_velocity_p[i, :]
                old_position = nodes[i]
                new_velocity = old_velocity + u_i * DELTA_T
                new_position = old_position + DELTA_T * new_velocity + (DELTA_T ** 2 / 2) * u_i
                POSITION_X[i, t] = new_position[0]
                POSITION_Y[i, t] = new_position[1]
                nodes_velocity_p[i, :] = new_velocity
                nodes[i, :] = new_position
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)
        if (t + 1) % SNAPSHOT_INTERVAL == 0:
            plot_neighbors()


# Function to plot trajectory of nodes
def plot_trajectory():
    for i in range(0, N):
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Trajectory of Sensor Nodes')
    plt.grid(True)
    plt.show()


# Function to plot velocity magnitude of nodes over time
def plot_velocity():
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]
        plt.plot(velocity_i)
    plt.xlabel('Time')
    plt.ylabel('Velocity Magnitude')
    plt.title('Velocity Magnitude of Sensor Nodes')
    plt.grid(True)
    plt.show()


# Function to plot connectivity of the network over time
def plot_connectivity():
    plt.plot(connectivity)
    plt.xlabel('Time')
    plt.ylabel('Connectivity')
    plt.title('Connectivity of Sensor Network')
    plt.grid(True)
    plt.show()


# Initial deployment plotting
plot_deployment()

# Simulation and plotting
get_positions()
plot_trajectory()
plot_velocity()
plot_connectivity()
