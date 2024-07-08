import numpy as np
import matplotlib.pyplot as plt

# Parameters start
X = 50  # Width of the area where sensor nodes are deployed
Y = 50  # Height of the area where sensor nodes are deployed
EPSILON = 0.1  # Small constant used in calculations
H = 0.2  # Threshold value for a bump function
C1_ALPHA = 30  # Coefficient for position adjustment
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)  # Coefficient for velocity adjustment
N = 100  # Number of sensor nodes
M = 2  # Number of dimensions in space
D = 15  # Desired distance among sensor nodes
K = 1.2  # Scaling factor for interaction range
R = K * D  # Interaction range
DELTA_T = 0.009  # Time step for simulation
A = 5  # Constant parameter for calculation
B = 5  # Constant parameter for calculation
C = np.abs(A - B) / np.sqrt(4 * A * B)  # Constant parameter for calculation
ITERATION = 500  # Number of simulation iterations
SNAPSHOT_INTERVAL = 70  # Interval for capturing snapshots during simulation
POSITION_X = np.zeros([N, ITERATION])  # Array to store x-coordinates of sensor nodes over time
POSITION_Y = np.zeros([N, ITERATION])  # Array to store y-coordinates of sensor nodes over time

# Initialize sensor nodes randomly within the deployment area
nodes = np.random.rand(N, M) * X
nodes_old = nodes  # Store initial positions of sensor nodes
nodes_velocity_p = np.zeros([N, M])  # Initialize velocity of sensor nodes
a_ij_matrix = np.zeros([N, N])  # Matrix to store pairwise coefficients
velocity_magnitudes = np.zeros([N, ITERATION])  # Array to store velocity magnitudes of sensor nodes over time
connectivity = np.zeros([ITERATION, 1])  # Array to store network connectivity over time
fig = plt.figure()  # Create a figure for plotting
fig_counter = 0  # Counter for figure number
q_mt = np.array([150, 150])  # Target position for sensor nodes
c1_mt = 1.1  # Coefficient for position adjustment towards the target
c2_mt = 2 * np.sqrt(c1_mt)  # Coefficient for velocity adjustment towards the target

def create_adjacency_matrix():
    """
    Create adjacency matrix based on distance between sensor nodes.
    """
    adjacency_matrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                distance = np.linalg.norm(nodes[i] - nodes[j])
                if distance <= R:
                    adjacency_matrix[i, j] = 1  # If within interaction range, set connection to 1
    return adjacency_matrix


def plot_deployment():
    """
    Plot the initial deployment of sensor nodes.
    """
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot sensor nodes as red circles
    plt.show()


def plot_neighbors():
    """
    Plot sensor nodes and their connections.
    """
    plt.plot(q_mt[0], q_mt[1], marker='o', color='green')  # Plot target position as green circle
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot sensor nodes as red circles
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                plt.plot([nodes[i, 0], nodes[j, 0]],
                         [nodes[i, 1], nodes[j, 1]],
                         'b-', lw=1)  # Plot connection lines between nodes within interaction range
    plt.show()


def sigma_norm(z):
    """
    Calculate sigma norm function.
    """
    val = EPSILON * (z ** 2)
    val = np.sqrt(1 + val) - 1
    val = val / EPSILON
    return val


def bump_function(z):
    """
    Calculate bump function.
    """
    if 0 <= z < H:
        return 1
    elif H <= z <= 1:
        val = (z - H) / (1 - H)
        val = np.cos(np.pi * val)
        val = (1 + val) / 2
        return val
    else:
        return 0


def sigma_1(z):
    """
    Calculate sigma_1 function.
    """
    val = 1 + z ** 2
    val = np.sqrt(val)
    val = z / val
    return val


def phi(z):
    """
    Calculate phi function.
    """
    val_1 = A + B
    val_2 = sigma_1(z + C)
    val_3 = A - B
    val = val_1 * val_2 + val_3
    val = val / 2
    return val


def phi_alpha(z):
    """
    Calculate phi_alpha function.
    """
    input_1 = z / sigma_norm(R)  # Sigma norm of R is R_alpha
    input_2 = z - sigma_norm(D)  # Sigma norm of D is D_alpha
    val_1 = bump_function(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val


def get_a_ij(i, j):
    """
    Calculate coefficient a_ij for pair of nodes.
    """
    val_1 = nodes_old[j] - nodes_old[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm) / sigma_norm(R)
    val = bump_function(val_2)
    return val


def get_n_ij(i, j):
    """
    Calculate unit vector n_ij for pair of nodes.
    """
    val_1 = nodes_old[j] - nodes_old[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm ** 2
    val = val_1 / np.sqrt(val_2)
    return val


def get_u_i(i):
    """
    Calculate velocity adjustment for sensor node.
    """
    sum_1 = np.array([0.0, 0.0])
    sum_2 = np.array([0.0, 0.0])
    for j in range(0, N):
        distance = np.linalg.norm(nodes_old[j] - nodes_old[i])
        if distance <= R:
            val_1 = nodes_old[j] - nodes_old[i]
            norm = np.linalg.norm(val_1)
            phi_alpha_val = phi_alpha(sigma_norm(norm))
            val = phi_alpha_val * get_n_ij(i, j)
            sum_1 += val

            val_2 = nodes_velocity_p[j] - nodes_velocity_p[i]
            sum_2 += get_a_ij(i, j) * val_2
    val = C1_ALPHA * sum_1 + C2_ALPHA * sum_2 - c1_mt * (nodes_old[i] - q_mt)
    return val


def get_positions():
    """
    Update positions of sensor nodes over time.
    """
    for t in range(0, ITERATION):
        adjacency_matrix = create_adjacency_matrix()  # Generate adjacency matrix
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)  # Calculate network connectivity
        if t == 0:
            plot_neighbors()  # Plot initial deployment and connections
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
        else:
            for i in range(0, N):
                u_i = get_u_i(i)  # Get velocity adjustment for node
                old_velocity = nodes_velocity_p[i, :]
                old_position = np.array([POSITION_X[i, t - 1],
                                         POSITION_Y[i, t - 1]])
                new_position = old_position + DELTA_T * old_velocity + (DELTA_T ** 2 / 2) * u_i
                new_velocity = (new_position - old_position) / DELTA_T
                [POSITION_X[i, t], POSITION_Y[i, t]] = new_position
                nodes_velocity_p[i, :] = new_velocity
                nodes[i, :] = new_position
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)
        if (t + 1) % SNAPSHOT_INTERVAL == 0:
            plot_neighbors()  # Plot intermediate deployment and connections


def plot_trajectory():
    """
    Plot trajectories of sensor nodes over time.
    """
    for i in range(0, N):
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Trajectory of Sensor Nodes')
    plt.grid(True)
    plt.show()


def plot_velocity():
    """
    Plot velocity magnitudes of sensor nodes over time.
    """
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]
        plt.plot(velocity_i)
    plt.xlabel('Time')
    plt.ylabel('Velocity Magnitude')
    plt.title('Velocity Magnitude of Sensor Nodes')
    plt.grid(True)
    plt.show()


def plot_connectivity():
    """
    Plot network connectivity over time.
    """
    plt.plot(connectivity)
    plt.xlabel('Time')
    plt.ylabel('Connectivity')
    plt.title('Connectivity of Sensor Network')
    plt.grid(True)
    plt.show()


plot_deployment()  # Plot initial deployment of sensor nodes
get_positions()  # Run simulation to update positions of sensor nodes over time
plot_trajectory()  # Plot trajectories of sensor nodes
plot_velocity()  # Plot velocity magnitudes of sensor nodes
plot_connectivity()  # Plot network connectivity over time
