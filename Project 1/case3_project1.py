import numpy as np
import matplotlib.pyplot as plt

# Parameters start
X = 150  # X-dimension of the space
Y = 150  # Y-dimension of the space
EPSILON = 0.1  # Small value for numerical stability
H = 0.2  # Threshold for the bump function
C1_ALPHA = 70  # Weighting factor for potential field
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)  # Weighting factor for potential field
N = 100  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor nodes
K = 1.2  # Scaling factor
R = K * D  # Interaction range
DELTA_T = 0.009  # Time step
A = 5  # Parameter for potential function
B = 5  # Parameter for potential function
C = np.abs(A - B) / np.sqrt(4 * A * B)  # Parameter for potential function
ITERATION_VALUES = np.arange(0, 7, DELTA_T)  # Time steps for simulation
ITERATION = ITERATION_VALUES.shape[0]  # Number of iterations
SNAPSHOT_INTERVAL = 50  # Interval for plotting snapshots
POSITION_X = np.zeros([N, ITERATION])  # Array to store x-coordinates of nodes over time
POSITION_Y = np.zeros([N, ITERATION])  # Array to store y-coordinates of nodes over time

# Generate random initial positions for nodes
n_x = np.random.rand(N) * X
n_y = np.random.rand(N) * X
nodes = np.array([n_x, n_y]).T
nodes_old = nodes  # Store old node positions
nodes_velocity_p = np.zeros([N, M])  # Initialize velocities of nodes
a_ij_matrix = np.zeros([N, N])  # Adjacency matrix for connectivity
velocity_magnitudes = np.zeros([N, ITERATION])  # Magnitude of velocities for each node
connectivity = np.zeros([ITERATION, 1])  # Connectivity of the network over time
fig = plt.figure()  # Create a figure for plotting
fig_counter = 0  # Counter for saving figures
c1_mt = 20  # Weighting factor for target attraction
c2_mt = 2 * np.sqrt(c1_mt)  # Weighting factor for target attraction
q_mt_x1 = 50  # Initial x-coordinate of target point
q_mt_y1 = 295  # Initial y-coordinate of target point
q_mt_x1_old = q_mt_x1  # Store old x-coordinate of target point
q_mt_y1_old = q_mt_y1  # Store old y-coordinate of target point
target_points = np.zeros([ITERATION, M])  # Target points over time
center_of_mass = np.zeros([ITERATION, M])  # Center of mass of the network over time


# Parameters end

# Function to create adjacency matrix based on current node positions
def create_adjacency_matrix():
    adjacency_matrix = np.zeros([N, N])  # Initialize adjacency matrix
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                distance = np.linalg.norm(nodes[i] - nodes[j])  # Calculate distance between nodes
                if distance <= R:  # If within interaction range
                    adjacency_matrix[i, j] = 1  # Set adjacency value to 1
    return adjacency_matrix


# Function to plot initial deployment of nodes
def plot_deployment():
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot nodes
    plt.show()  # Show plot


# Function to plot neighbors of nodes at a given time
def plot_neighbors(t):
    plt.plot(target_points[0:t, 0], target_points[0:t, 1])  # Plot target points over time
    plt.plot(q_mt_x1_old, q_mt_y1_old, marker='o', color='green')  # Plot old target point
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot nodes
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])  # Calculate distance between nodes
            if distance <= R:  # If within interaction range
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-', lw=0.5)  # Plot connection
    plt.show()  # Show plot


# Functions for potential field and interaction
def sigma_norm(z):
    val = EPSILON * (z ** 2)
    val = np.sqrt(1 + val) - 1
    val = val / EPSILON
    return val


def bump_function(z):
    if 0 <= z <= H:
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
    input_1 = z / sigma_norm(R)
    input_2 = z - sigma_norm(D)
    val_1 = bump_function(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val


# Functions to calculate interaction forces
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


def get_u_i(i, q_mt, p_mt):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        distance = np.linalg.norm(nodes[j] - nodes[i])
        if distance <= R:
            val_1 = nodes[j] - nodes[i]
            norm = np.linalg.norm(val_1)
            phi_alpha_val = phi_alpha(sigma_norm(norm))
            val = phi_alpha_val * get_n_ij(i, j)
            sum_1 += val

            val_2 = nodes_velocity_p[j] - nodes_velocity_p[i]
            sum_2 += get_a_ij(i, j) * val_2
    val = C1_ALPHA * sum_1 + C2_ALPHA * sum_2 - c1_mt * (nodes[i] - q_mt) - \
          c2_mt * (nodes_velocity_p[i] - p_mt)
    return val


# Function to update node positions over time
def get_positions():
    global q_mt_x1_old, q_mt_y1_old
    for t in range(0, ITERATION):
        adjacency_matrix = create_adjacency_matrix()  # Create adjacency matrix
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)  # Calculate connectivity
        center_of_mass[t] = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1])])  # Calculate center of mass
        if t == 0:
            target_points[t] = np.array([q_mt_x1_old, q_mt_y1_old])  # Set initial target point
            plot_neighbors(t)  # Plot initial neighbors
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]  # Record x-coordinate
                POSITION_Y[i, t] = nodes[i, 1]  # Record y-coordinate
        else:
            q_mt_x1 = 50 + 50 * ITERATION_VALUES[t]  # Update x-coordinate of target point
            q_mt_y1 = 295 - 50 * np.sin(ITERATION_VALUES[t])  # Update y-coordinate of target point
            q_mt = np.array([q_mt_x1, q_mt_y1])  # New target point
            target_points[t] = q_mt  # Record target point
            q_mt_old = np.array([q_mt_x1_old, q_mt_y1_old])  # Old target point
            p_mt = (q_mt - q_mt_old) / DELTA_T  # Calculate velocity of target point
            q_mt_x1_old = q_mt_x1  # Update old x-coordinate of target point
            q_mt_y1_old = q_mt_y1  # Update old y-coordinate of target point
            for i in range(0, N):
                u_i = get_u_i(i, q_mt, p_mt)  # Calculate interaction force
                old_velocity = nodes_velocity_p[i, :]  # Old velocity of node
                old_position = np.array([POSITION_X[i, t - 1], POSITION_Y[i, t - 1]])  # Old position of node
                new_velocity = old_velocity + u_i * DELTA_T  # Update velocity
                new_position = old_position + DELTA_T * new_velocity + (DELTA_T ** 2 / 2) * u_i  # Update position
                [POSITION_X[i, t], POSITION_Y[i, t]] = new_position  # Record new position
                nodes_velocity_p[i, :] = new_velocity  # Record new velocity
                nodes[i, :] = new_position  # Update node position
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)  # Record velocity magnitude
        if t % SNAPSHOT_INTERVAL == 0:
            plot_neighbors(t)  # Plot neighbors at certain intervals


# Functions to plot simulation results
def plot_trajectory():
    for i in range(0, N):
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])  # Plot trajectory of each node
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Trajectory of Sensor Nodes')
    plt.grid(True)
    plt.show()

# Function to plot velocity of nodes over time
def plot_velocity():
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]  # Get velocity magnitude over time
        plt.plot(velocity_i)  # Plot velocity magnitude
    plt.xlabel('Time')
    plt.ylabel('Velocity Magnitude')
    plt.title('Velocity Magnitude of Sensor Nodes')
    plt.grid(True)
    plt.show()

# Function to plot connectivity of the network over time
def plot_connectivity():
    plt.plot(connectivity)  # Plot connectivity over time
    plt.xlabel('Time')
    plt.ylabel('Connectivity')
    plt.title('Connectivity of Sensor Network')
    plt.grid(True)
    plt.show()

# Function to plot center of mass of the network over time
def plot_center_of_mass():
    plt.plot(target_points[:, 0], target_points[:, 1])  # Plot target points
    plt.plot(center_of_mass[:, 0], center_of_mass[:, 1])  # Plot center of mass
    plt.xlabel('Time')
    plt.ylabel('Center of Mass')
    plt.title('Center of Mass of Sensor Network')
    plt.grid(True)
    plt.show()


plot_deployment()  # Plot initial deployment
get_positions()  # Simulate positions of nodes over time
plot_trajectory()  # Plot trajectory of nodes
plot_velocity()  # Plot velocity of nodes
plot_connectivity()  # Plot connectivity of network
plot_center_of_mass()  # Plot center of mass of network
