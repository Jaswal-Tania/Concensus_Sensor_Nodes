import numpy as np
import matplotlib.pyplot as plt

# Parameters start
X = 70  # X dimension of the space
Y = 70  # Y dimension of the space
EPSILON = 0.1  # Small constant epsilon for calculations
H = 0.2  # Threshold value for the bump function
C1_ALPHA = 20  # Constant for the alignment term in the force equation
C2_ALPHA = 2 * np.sqrt(C1_ALPHA)  # Constant for the velocity matching term in the force equation
N = 150  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor nodes
K = 1.2  # Scaling factor
R = K * D  # Interaction range
DELTA_T = 0.009  # Time step
A = 5  # Parameter for phi function
B = 5  # Parameter for phi function
C = np.abs(A - B) / np.sqrt(4 * A * B)  # Parameter for phi function
ITERATION = 500  # Number of iterations
SNAPSHOT_INTERVAL = 50  # Interval for generating snapshots
POSITION_X = np.zeros([N, ITERATION])  # Array to store X positions of sensor nodes over time
POSITION_Y = np.zeros([N, ITERATION])  # Array to store Y positions of sensor nodes over time

# Initialize sensor nodes randomly within the space
nodes = np.random.rand(N, M) * X
nodes_old = nodes  # Store initial positions of sensor nodes
nodes_velocity_p = np.zeros([N, M])  # Array to store velocity of sensor nodes
a_ij_matrix = np.zeros([N, N])  # Adjacency matrix for connectivity analysis
velocity_magnitudes = np.zeros([N, ITERATION])  # Array to store velocity magnitudes of sensor nodes
connectivity = np.zeros([ITERATION, 1])  # Array to store connectivity over time
fig = plt.figure()  # Initialize figure for plotting
fig_counter = 0  # Counter for figure indexing
q_mt = np.array([250, 25])  # Target position
c1_mt = 1.1  # Constant for target attraction
c2_mt = 2 * np.sqrt(c1_mt)  # Constant for target attraction

# Define obstacle parameters
obstacles = np.array([[100, 10], [100, 80]])  # Obstacle positions
Rk = np.array([10, 10])  # Obstacle radii
num_obstacles = obstacles.shape[0]  # Number of obstacles
c1_beta = 5000  # Constant for obstacle avoidance
c2_beta = 2 * np.sqrt(c1_beta)  # Constant for obstacle avoidance
r_prime = 0.22 * K * R  # Parameter for obstacle avoidance
d_prime = 15  # Parameter for obstacle avoidance

center_of_mass = np.zeros([ITERATION, M])  # Array to store center of mass trajectory


# Parameters end

# Function to compute sigma norm
def sigma_norm(z):
    val = EPSILON * (z ** 2)
    val = np.sqrt(1 + val) - 1
    val = val / EPSILON
    return val


r_beta = sigma_norm(np.linalg.norm(r_prime))
d_beta = sigma_norm(np.linalg.norm(d_prime))
s = 1


# Function to create adjacency matrix
def create_adjacency_matrix():
    adjacency_matrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                distance = np.linalg.norm(nodes[i] - nodes[j])
                if distance <= R:
                    adjacency_matrix[i, j] = 1
    return adjacency_matrix


# Function to plot initial deployment with obstacles
def plot_deployment_with_obstacles():
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    plt.plot(q_mt[0], q_mt[1], marker='o', color='green')  # Plot the target
    for i in range(num_obstacles):
        plt.gca().add_patch(plt.Circle((obstacles[i, 0], obstacles[i, 1]), Rk[i], color='red'))  # Plot obstacles
    plt.xlim(0, X)
    plt.ylim(0, Y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, 270)
    plt.ylim(0, 100)
    plt.show()


# Function to plot neighboring nodes
def plot_neighbors(t):
    plt.plot(q_mt[0], q_mt[1], marker='o', color='green')
    for i in range(0, num_obstacles):
        plt.gcf().gca().add_artist(plt.Circle((obstacles[i, 0], obstacles[i, 1]), Rk[i], color='red'))
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-', lw=0.5)
    plt.show()


# Bump function to model obstacle avoidance
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


# Sigma_1 function
def sigma_1(z):
    val = 1 + z ** 2
    val = np.sqrt(val)
    val = z / val
    return val


# Phi function
def phi(z):
    val_1 = A + B
    val_2 = sigma_1(z + C)
    val_3 = A - B
    val = val_1 * val_2 + val_3
    val = val / 2
    return val


# Phi_alpha function
def phi_alpha(z):
    input_1 = z / sigma_norm(R)
    input_2 = z - sigma_norm(D)
    val_1 = bump_function(input_1)
    val_2 = phi(input_2)
    val = val_1 * val_2
    return val


# Phi_beta function
def phi_beta(z):
    val1 = bump_function(z / d_beta)
    val2 = sigma_1(z - d_beta) - 1
    return val1 * val2


# Function to compute a_ij
def get_a_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = sigma_norm(norm) / sigma_norm(R)
    val = bump_function(val_2)
    return val


# Function to compute n_ij
def get_n_ij(i, j):
    val_1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val_1)
    val_2 = 1 + EPSILON * norm ** 2
    val = val_1 / np.sqrt(val_2)
    return val


# Function to compute force acting on node i
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


# Function to update positions of sensor nodes
def get_positions():
    for t in range(0, ITERATION):
        adjacency_matrix = create_adjacency_matrix()
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(adjacency_matrix)
        center_of_mass[t] = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1])])

        if t == 0:
            plot_neighbors(t)
            for i in range(0, N):
                POSITION_X[i, t] = nodes[i, 0]
                POSITION_Y[i, t] = nodes[i, 1]
        else:
            for i in range(0, N):
                old_velocity = nodes_velocity_p[i, :]
                old_position = np.array([POSITION_X[i, t - 1], POSITION_Y[i, t - 1]])
                u_i_sum = np.array([0.0, 0.0])  # Initialize total force vector for node i

                for k in range(num_obstacles):  # Iterate over all obstacles
                    mu = Rk[k] / np.linalg.norm((old_position - obstacles[k]))
                    a_k = (old_position - obstacles[k]) / np.linalg.norm((old_position - obstacles[k]))
                    P = np.eye(2) - np.outer(a_k, a_k)  # Projection matrix onto orthogonal complement
                    p_i_k = mu * np.matmul(P, old_velocity)
                    q_i_k = mu * old_position + (1 - mu) * obstacles[k]
                    b_i_k = bump_function(sigma_norm(np.linalg.norm(q_i_k - old_position)) / d_beta)
                    n_i_k = (q_i_k - old_position) / np.sqrt(1 + EPSILON * (np.linalg.norm(q_i_k - old_position)) ** 2)

                    # Calculate total force acting on node i from all obstacles
                    u_i_sum += get_u_i(i, mu, a_k, P, p_i_k, q_i_k, b_i_k, n_i_k, old_position)

                new_position = old_position + DELTA_T * old_velocity + (DELTA_T ** 2 / 2) * u_i_sum
                new_velocity = (new_position - old_position) / DELTA_T

                POSITION_X[i, t] = new_position[0]
                POSITION_Y[i, t] = new_position[1]
                nodes_velocity_p[i, :] = new_velocity
                nodes[i, :] = new_position
                velocity_magnitudes[i, t] = np.linalg.norm(new_velocity)

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


# Function to plot velocity of sensor nodes
def plot_velocity():
    for i in range(0, N):
        velocity_i = velocity_magnitudes[i, :]
        plt.plot(velocity_i)
    plt.xlim(0, 600)
    plt.xlabel('Time')
    plt.ylabel('Velocity Magnitude')
    plt.title('Velocity Magnitude of Sensor Nodes')
    plt.grid(True)
    plt.show()


# Function to plot connectivity over time
def plot_connectivity():
    plt.plot(connectivity)
    plt.xlim(0, 600)
    plt.xlabel('Time')
    plt.ylabel('Connectivity')
    plt.title('Connectivity of Sensor Network')
    plt.grid(True)
    plt.show()


# Function to plot center of mass trajectory with target trajectory
def plot_center_of_mass_with_target_trajectory():
    plt.plot(center_of_mass[:, 0], center_of_mass[:, 1], label='Center of Mass')
    plt.plot(q_mt[0], q_mt[1], marker='o', color='green', label='Target')
    plt.legend()
    plt.xlim(0, 400)
    plt.ylim(0, 200)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Center of Mass and Target Trajectory')
    plt.show()


# Plot initial deployment with obstacles
plot_deployment_with_obstacles()

# Compute positions of sensor nodes over time
get_positions()

# Plot trajectory of sensor nodes
plot_trajectory()

# Plot velocity of sensor nodes
plot_velocity()

# Plot connectivity over time
plot_connectivity()

# Plot center of mass trajectory with target trajectory
plot_center_of_mass_with_target_trajectory()
