import numpy as np
import matplotlib.pyplot as plt

# Parameters start
X = 70  # Width of the area
Y = 70  # Height of the area
EPSILON = 0.1  # Small constant for sigma functions
H = 0.2  # Parameter for bump function
C1_ALPHA = 70  # Coefficient for attraction towards neighbors
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)  # Coefficient for alignment with neighbors
N = 150  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor nodes
K = 1.2  # Scaling factor
R = K * D  # Interaction range
DELTA_T = 0.009  # Time step for simulation
A = 5  # Parameter for phi function
B = 5  # Parameter for phi function
C = np.abs(A - B) / np.sqrt(4 * A * B)  # Parameter for phi function
ITERATION_VALUES = np.arange(0, 7, DELTA_T)  # Time steps for simulation
ITERATION = ITERATION_VALUES.shape[0]  # Total number of time steps
SNAPSHOT_INTERVAL = 30  # Interval for taking snapshots
POSITION_X = np.zeros([N, ITERATION])  # X-coordinates of sensor nodes over time
POSITION_Y = np.zeros([N, ITERATION])  # Y-coordinates of sensor nodes over time

# Initial deployment of sensor nodes
nodes = np.random.rand(N, M) * X  # Random positions within the area
nodes_old = nodes  # Store old positions for reference
nodes_velocity_p = np.zeros([N, M])  # Initial velocities of sensor nodes
a_ij_matrix = np.zeros([N, N])  # Adjacency matrix for connectivity
velocity_magnitudes = np.zeros([N, ITERATION])  # Magnitude of velocities over time
connectivity = np.zeros([ITERATION, 1])  # Connectivity of the network over time
fig = plt.figure()  # Create a figure for plotting
fig_counter = 0  # Counter for figures

# Initial target point and parameters for moving target
q_mt_x1 = 40  # Initial x-coordinate of target point
q_mt_y1 = 25  # Initial y-coordinate of target point
q_mt = ([q_mt_x1, q_mt_y1])  # Current target point
q_mt_x1_old = q_mt_x1  # Store old x-coordinate of target point
q_mt_y1_old = q_mt_y1  # Store old y-coordinate of target point
target_points = np.zeros([ITERATION, M])  # Target points over time
c1_mt = 1.1  # Coefficient for attraction towards target
c2_mt = 2 * np.sqrt(c1_mt)  # Coefficient for alignment with target

# Parameters for obstacles
obstacles = np.array([[20, 20], [30, 60]])  # Positions of obstacles
Rk = np.array([10, 10])  # Radii of obstacles
num_obstacles = obstacles.shape[0]  # Number of obstacles
c1_beta = 10000  # Coefficient for obstacle avoidance
c2_beta = 2 * np.sqrt(c1_beta)  # Coefficient for obstacle avoidance
r_prime = 0.22 * K * R  # Prime distance for sigma functions
d_prime = 15  # Prime distance for obstacle avoidance

center_of_mass = np.zeros([ITERATION, M])  # Center of mass of sensor network over time


# Parameters end

# Function to compute sigma function
def sigma_norm(z):
    val = EPSILON * (z ** 2)
    val = np.sqrt(1 + val) - 1
    val = val / EPSILON
    return val


r_beta = sigma_norm(np.linalg.norm(r_prime))
d_beta = sigma_norm(np.linalg.norm(d_prime))
s = 1


# Function to create adjacency matrix for connectivity
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
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot sensor nodes
    plt.plot(q_mt[0], q_mt[1], marker='o', color='green')  # Plot the target
    for i in range(num_obstacles):
        plt.gca().add_patch(plt.Circle((obstacles[i, 0], obstacles[i, 1]), Rk[i], color='red'))  # Plot obstacles
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, 100)
    plt.ylim(0, 150)
    plt.show()


# Function to plot neighbors of sensor nodes
def plot_neighbors(t):
    plt.plot(target_points[0:t, 0], target_points[0:t, 1])  # Plot target points over time
    plt.plot(q_mt_x1_old, q_mt_y1_old, marker='o', color='green')  # Plot old target point
    for i in range(0, num_obstacles):
        plt.gcf().gca().add_artist(plt.Circle((obstacles[i, 0], obstacles[i, 1]), Rk[i], color='red'))
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # Plot sensor nodes
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-', lw=0.5)
    plt.show()


# Function for the bump function
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


# Function for the sigma_1 function
def sigma_1(z):
    val = 1 + z ** 2
    val = np.sqrt(val)
    val = z / val
    return val


# Function for the phi function
def phi(z):
    val_1 = A + B
    val_2 = sigma_1(z + C)
    val_3 = A - B
    val = val_1 * val_2 + val_3
    val = val / 2
    return val


# Function for the phi_alpha function
def phi_alpha(z):
    input_1 = z / sigma_norm(R)
    input_2 = z - sigma_norm(D)
    val_1 = bump_function(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val


# Function for the phi_beta function
def phi_beta(z):
    val1 = bump_function(z / d_beta)
    val2 = sigma_1(z - d_beta) - 1
    return val1 * val2


# Function to get a_ij (bump function) for neighboring nodes
def get_a_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm) / sigma_norm(R)
    val = bump_function(val_2)
    return val


# Function to get normalized vector n_ij (unit vector) for neighboring nodes
def get_n_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm ** 2
    val = val_1 / np.sqrt(val_2)
    return val


# Function to compute the total force acting on node i
def get_u_i(i, mu, a_k, P, p_i_k, q_i_k, b_i_k, n_i_k, old_position):
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
    val = C1_ALPHA * sum_1 + C2_ALPHA * sum_2 - c1_mt * (nodes[i] - q_mt) + \
          c1_beta * phi_beta(sigma_norm(np.linalg.norm(q_i_k - old_position))) * n_i_k + \
          c2_beta * b_i_k * (p_i_k - nodes_velocity_p[i])
    return val


# Function to compute new positions of sensor nodes and target point over time
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
            # Update target point position over time
            q_mt_x1 = 40 + 40 * ITERATION_VALUES[t]
            q_mt_y1 = 25 - 40 * np.sin(ITERATION_VALUES[t])
            q_mt = np.array([q_mt_x1, q_mt_y1])  # New target point
            target_points[t] = q_mt  # Record target point
            q_mt_old = np.array([q_mt_x1_old, q_mt_y1_old])  # Old target point
            p_mt = (q_mt - q_mt_old) / DELTA_T  # Calculate velocity of target point
            q_mt_x1_old = q_mt_x1  # Update old x-coordinate of target point
            q_mt_y1_old = q_mt_y1  # Update old y-coordinate of target point
            for i in range(0, N):
                old_velocity = nodes_velocity_p[i, :]  # Old velocity of node
                old_position = np.array([POSITION_X[i, t - 1], POSITION_Y[i, t - 1]])  # Old position of node
                u_i_sum = np.array([0.0, 0.0])  # Initialize total force vector for node i

                # Calculate direction towards the target point
                direction_to_target = q_mt - old_position
                distance_to_target = np.linalg.norm(direction_to_target)
                if distance_to_target != 0:
                    direction_to_target /= distance_to_target

                # Iterate over all obstacles for obstacle avoidance
                for k in range(num_obstacles):
                    mu = Rk[k] / np.linalg.norm((old_position - obstacles[k]))
                    a_k = (old_position - obstacles[k]) / np.linalg.norm((old_position - obstacles[k]))
                    P = np.eye(2) - np.outer(a_k, a_k)  # Projection matrix onto orthogonal complement
                    p_i_k = mu * np.matmul(P, old_velocity)
                    q_i_k = mu * old_position + (1 - mu) * obstacles[k]
                    b_i_k = bump_function(sigma_norm(np.linalg.norm(q_i_k - old_position)) / d_beta)
                    n_i_k = (q_i_k - old_position) / np.sqrt(1 + EPSILON * (np.linalg.norm(q_i_k - old_position)) ** 2)

                    # Calculate total force acting on node i from all obstacles
                    u_i_sum += get_u_i(i, mu, a_k, P, p_i_k, q_i_k, b_i_k, n_i_k, old_position)

                # Adjust velocity towards the target point
                desired_velocity = direction_to_target * np.linalg.norm(old_velocity)
                new_velocity = old_velocity + (desired_velocity - old_velocity) * 0.1  # Adjusting factor for stability

                # Update positions and velocities of nodes
                new_position = old_position + DELTA_T * new_velocity + (DELTA_T ** 2 / 2) * u_i_sum
                new_velocity = (new_position - old_position) / DELTA_T

                POSITION_X[i, t] = new_position[0]
                POSITION_Y[i, t] = new_position[1]
                nodes_velocity_p[i, :] = new_velocity
                nodes[i, :] = new_position
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)

        # Take snapshots of neighbors at certain intervals
        if (t + 1) % SNAPSHOT_INTERVAL == 0:
            plot_neighbors(t)


# Function to plot trajectory of sensor nodes
def plot_trajectory():
    for i in range(0, N):
        plt.plot(POSITION_X[i, :], POSITION_Y[i, :])
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Trajectory of Sensor Nodes')
    plt.grid(True)
    plt.show()


# Function to plot velocity magnitude of sensor nodes
def plot_velocity():
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]
        plt.plot(velocity_i)
    plt.xlabel('Time')
    plt.ylabel('Velocity Magnitude')
    plt.title('Velocity Magnitude of Sensor Nodes')
    plt.grid(True)
    plt.show()


# Function to plot connectivity of the network
def plot_connectivity():
    plt.plot(connectivity)
    plt.xlabel('Time')
    plt.ylabel('Connectivity')
    plt.title('Connectivity of Sensor Network')
    plt.grid(True)
    plt.show()


# Function to plot center of mass of the sensor network
def plot_center_of_mass():
    plt.plot(target_points[:, 0], target_points[:, 1])  # Plot target points
    plt.plot(center_of_mass[:, 0], center_of_mass[:, 1])
    plt.xlabel('Time')
    plt.ylabel('Center of Mass')
    plt.title('Center of Mass of Sensor Network')
    plt.grid(True)
    plt.show()


# Main script
plot_deployment()  # Plot initial deployment
get_positions()  # Compute positions of sensor nodes over time
plot_trajectory()  # Plot trajectory of sensor nodes
plot_velocity()  # Plot velocity magnitude of sensor nodes
plot_connectivity()  # Plot connectivity of the network
plot_center_of_mass()  # Plot center of mass of the sensor network
