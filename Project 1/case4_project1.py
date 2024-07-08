import numpy as np
import matplotlib.pyplot as plt

# Parameters start
X = 150  # Width of the area
Y = 150  # Height of the area
EPSILON = 0.1  # Small constant used in calculations
H = 0.2  # Threshold for bump function
C1_ALPHA = 70  # Coefficient for attraction towards target
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)  # Coefficient for alignment with neighbors

# Constants defining the simulation environment
N = 100  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor node
K = 1.2  # Scaling factor
R = K * D  # Interaction range
DELTA_T = 0.009  # Time step for simulation
A = 5  # Parameter for potential function
B = 5  # Parameter for potential function
C = np.abs(A - B) / np.sqrt(4 * A * B)  # Parameter for potential function
ITERATION_VALUES = np.arange(0, 7, DELTA_T)  # Time steps for iteration
ITERATION = ITERATION_VALUES.shape[0]  # Total number of iterations
SNAPSHOT_INTERVAL = 50  # Interval for snapshotting the simulation
POSITION_X = np.zeros([N, ITERATION])  # X-coordinates of nodes over time
POSITION_Y = np.zeros([N, ITERATION])  # Y-coordinates of nodes over time
# Initialize random positions for nodes
nodes = np.random.rand(N, M) * X
nodes_old = nodes  # Store initial positions for reference
nodes_velocity_p = np.zeros([N, M])  # Velocity of nodes
a_ij_matrix = np.zeros([N, N])  # Matrix to store adjacency information
velocity_magnitudes = np.zeros([N, ITERATION])  # Magnitude of velocities of nodes over time
connectivity = np.zeros([ITERATION, 1])  # Connectivity of the network over time
fig = plt.figure()  # Create a figure for plotting
fig_counter = 0  # Counter for figure names
c1_mt = 50  # Coefficient for attraction towards global target
c2_mt = 2 * np.sqrt(c1_mt)  # Coefficient for alignment with global target
q_mt_x1 = 310  # X-coordinate of global target
q_mt_y1 = 255  # Y-coordinate of global target
q_mt_x1_old = q_mt_x1  # Store old X-coordinate for velocity calculation
q_mt_y1_old = q_mt_y1  # Store old Y-coordinate for velocity calculation
target_points = np.zeros([ITERATION, M])  # Store target positions over time
center_of_mass = np.zeros([ITERATION, M])  # Store center of mass over time


# Parameters end

# Function to create adjacency matrix based on node positions and interaction range
def create_adjacency_matrix():
    adjacency_matrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                distance = np.linalg.norm(nodes[i] - nodes[j])  # Calculate distance between nodes
                if distance <= R:  # If distance is within interaction range
                    adjacency_matrix[i, j] = 1  # Set corresponding entry in adjacency matrix to 1
    return adjacency_matrix


# Function to plot initial deployment of nodes
def plot_deployment():
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot nodes
    plt.show()


# Function to plot neighbor relationships among nodes
def plot_neighbors(t):
    plt.plot(target_points[0:t, 0], target_points[0:t, 1])  # Plot target trajectory
    plt.plot(q_mt_x1_old, q_mt_y1_old, marker='o', color='green')  # Plot old global target position
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot nodes
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])  # Calculate distance between nodes
            if distance <= R:  # If distance is within interaction range
                plt.plot([nodes[i, 0], nodes[j, 0]],
                         [nodes[i, 1], nodes[j, 1]],
                         'b-', lw=0.5)  # Plot connection between nodes
    plt.show()


# Function for sigmoidal normalization
def sigma_norm(z):
    val = EPSILON * (z ** 2)
    val = np.sqrt(1 + val) - 1
    val = val / EPSILON
    return val


# Function for bump function
def bump_function(z):
    if 0 <= z < H:
        return 1
    elif H <= z <= 1:
        val = (z - H) / (1 - H)
        val = np.cos(np.pi * val)
        val = (1 + val) / 2
        return val
    else:
        return 0


# Function for potential function sigma_1
def sigma_1(z):
    val = 1 + z ** 2
    val = np.sqrt(val)
    val = z / val
    return val


# Function for potential function phi
def phi(z):
    val_1 = A + B
    val_2 = sigma_1(z + C)
    val_3 = A - B
    val = val_1 * val_2 + val_3
    val = val / 2
    return val


# Function for potential function phi_alpha
def phi_alpha(z):
    input_1 = z / sigma_norm(R)  # Sigma norm of R is R_alpha
    input_2 = z - sigma_norm(D)  # Sigma norm of D is D_alpha
    val_1 = bump_function(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val


# Function to calculate a_ij for a pair of nodes
def get_a_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm) / sigma_norm(R)
    val = bump_function(val_2)
    return val


# Function to calculate normalized vector n_ij for a pair of nodes
def get_n_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm ** 2
    val = val_1 / np.sqrt(val_2)
    return val


# Function to calculate control input for a node
def get_u_i(i, q_mt, p_mt):
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        distance = np.linalg.norm(nodes[j] - nodes[i])  # Calculate distance between nodes
        if distance <= R:  # If distance is within interaction range
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


# Function to calculate positions of nodes over time
def get_positions():
    global q_mt_x1_old, q_mt_y1_old
    for t in range(0, ITERATION):
        adjacency_matrix = create_adjacency_matrix()  # Create adjacency matrix
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)  # Calculate connectivity
        center_of_mass[t] = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1])])  # Calculate center of mass
        if t == 0:
            q_mt_x1 = 310 - 160 * np.cos(ITERATION_VALUES[t])  # Update target X-coordinate
            q_mt_y1 = 255 - 160 * np.sin(ITERATION_VALUES[t])  # Update target Y-coordinate
            q_mt = np.array([q_mt_x1, q_mt_y1])  # Combine X and Y to form target point
            target_points[t] = q_mt  # Store target point
            plot_neighbors(t)  # Plot neighbor relationships
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]  # Store X-coordinate of node
                POSITION_Y[i, t] = nodes[i, 1]  # Store Y-coordinate of node
        else:
            q_mt_x1 = 310 - 160 * np.cos(ITERATION_VALUES[t])  # Update target X-coordinate
            q_mt_y1 = 255 - 160 * np.sin(ITERATION_VALUES[t])  # Update target Y-coordinate
            q_mt = np.array([q_mt_x1, q_mt_y1])  # Combine X and Y to form target point
            target_points[t] = q_mt  # Store target point
            q_mt_old = np.array([q_mt_x1_old, q_mt_y1_old])  # Store old target point
            p_mt = (q_mt - q_mt_old) / DELTA_T  # Calculate velocity of global target
            q_mt_x1_old = q_mt_x1  # Update old target X-coordinate
            q_mt_y1_old = q_mt_y1  # Update old target Y-coordinate
            for i in range(0, N):
                u_i = get_u_i(i, q_mt, p_mt)  # Calculate control input for node
                old_velocity = nodes_velocity_p[i, :]  # Get old velocity of node
                old_position = np.array([POSITION_X[i, t - 1],
                                         POSITION_Y[i, t - 1]])  # Get old position of node
                new_velocity = old_velocity + u_i * DELTA_T  # Calculate new velocity of node
                new_position = old_position + DELTA_T * new_velocity + (
                            DELTA_T ** 2 / 2) * u_i  # Calculate new position of node
                [POSITION_X[i, t], POSITION_Y[i, t]] = new_position  # Store new position of node
                nodes_velocity_p[i, :] = new_velocity  # Store new velocity of node
                nodes[i, :] = new_position  # Update position of node
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)  # Store magnitude of new velocity
        if (t + 1) % SNAPSHOT_INTERVAL == 0:
            plot_neighbors(t)  # Plot neighbor relationships at snapshot intervals


# Function to plot trajectory of nodes over time
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
        velocity_i = velocity_magnitudes[i, :]  # Get velocity of node
        plt.plot(velocity_i)  # Plot velocity over time for each node
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
    plt.plot(target_points[:, 0], target_points[:, 1])  # Plot target trajectory
    plt.plot(center_of_mass[:, 0], center_of_mass[:, 1])  # Plot center of mass trajectory
    plt.xlabel('Time')
    plt.ylabel('Center of Mass')
    plt.title('Center of Mass of Sensor Network')
    plt.grid(True)
    plt.show()


# Plot initial deployment of nodes
plot_deployment()
# Perform simulation to get positions of nodes over time
get_positions()
# Plot trajectory of nodes
plot_trajectory()
# Plot velocity of nodes
plot_velocity()
# Plot connectivity of the network
plot_connectivity()
# Plot center of mass of the network
plot_center_of_mass()
