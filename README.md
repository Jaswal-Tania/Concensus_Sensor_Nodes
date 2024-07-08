## Project 1: MSN Flocking Formation Control

**Project Overview:**
This project explores the dynamics of Multi-Sensor Networks (MSN) in achieving flocking formation control. By implementing various algorithms, we investigate how MSN nodes interact to exhibit different flocking behaviors in different scenarios.

**Project Parameters:**
- Number of sensor nodes: 100
- Space dimensions: 2D
- Desired distance among sensor nodes: 15
- Scaling factor: 1.2, interaction range: 1.2 * 15
- Epsilon: 0.1, Delta_t: 0.009

**Cases:**

1. **MSN Fragmentation:**
   - **Implementation:** Randomly generate a network of 100 sensor nodes in a 50x50 area. Apply Algorithm 1 to simulate node fragmentation.
   - **Results:** Visuals and plots showing initial deployment, fragmentation snapshots, node trajectories, connectivity status, and velocity patterns.

2. **MSN Quasi-Lattice Formation with Static Target:**
   - **Implementation:** Generate a network of 100 sensor nodes in a 50x50 area with a static target at (150, 150). Use Algorithm 2 for quasi-lattice formation.
   - **Results:** Visuals and plots depicting initial deployment, flocking snapshots, node trajectories, connectivity status, and velocity patterns.

3. **MSN Quasi-Lattice Formation with Dynamic Target (Sine Wave):**
   - **Implementation:** Create a network of 100 sensor nodes in a 150x150 area with a target moving along a sine wave. Use Algorithm 2 for quasi-lattice formation.
   - **Results:** Visuals and plots showing initial deployment, dynamic flocking snapshots, node trajectories, center of mass tracking, and connectivity status.

4. **MSN Quasi-Lattice Formation with Dynamic Target (Circular Trajectory):**
   - **Implementation:** Generate a network of 100 sensor nodes in a 150x150 area with a target moving along a circular trajectory. Use Algorithm 2 for quasi-lattice formation.
   - **Results:** Visuals and plots illustrating initial deployment, dynamic flocking snapshots, node trajectories, center of mass tracking, and connectivity status.

**How to Run:**
1. Extract the files and open them in an IDE.
2. Install required libraries (`numpy`, `matplotlib`).
3. Run the script using `python filename.py`.

## Project 2: MSN Flocking Formation Control and Obstacle Avoidance

**Project Overview:**
This project investigates MSN dynamics in achieving flocking formation control with obstacle avoidance. Using Algorithm 3, we explore MSN node interactions to exhibit flocking behaviors and adapt to avoid obstacles.

**Project Parameters:**
- Number of sensor nodes: 150
- Space dimensions: 2D
- Desired distance among sensor nodes: 15
- Scaling factor: 1.2, interaction range: 1.2 * 15
- Epsilon: 0.1, Delta_t: 0.009

**Cases:**

1. **MSN Quasi-Lattice Formation with Obstacle Avoidance:**
   - **Implementation:** Generate a network of 150 sensor nodes in a 70x70 area with a static target at (250, 25). Apply Algorithm 3 for quasi-lattice formation and obstacle avoidance.
   - **Results:** Visuals and plots showing initial deployment, flocking snapshots, node trajectories, connectivity status, velocity patterns, and center of mass tracking.

2. **MSN Quasi-Lattice Formation with Dynamic Target (Sine Wave) and Obstacle Avoidance:**
   - **Implementation:** Create a network of 150 sensor nodes in a 70x70 area with a target moving along a sine wave. Use Algorithm 3 for quasi-lattice formation and obstacle avoidance.
   - **Results:** Visuals and plots illustrating initial deployment, dynamic flocking snapshots, node trajectories, center of mass tracking, and connectivity status.

**How to Run:**
1. Extract the files and open them in an IDE.
2. Install required libraries (`numpy`, `matplotlib`).
3. Run the script using `python filename.py`.

## Project 4: Consensus Filters for Sensor Networks

**Project Overview:**
This project implements consensus filters for sensor networks, focusing on static and dynamic sensor networks. The consensus filters estimate values based on noisy sensor measurements while leveraging communication and collaboration among neighboring nodes.

**Project Specifications:**
- Sensor network with 50 nodes
- Each sensor node measures zi = 40 + Gaussian noise (zero mean)
- Two consensus algorithms: Consensus 1 and Consensus 2

**Cases:**

1. **Static Sensor Network:**
   - **Analysis:** Plot the estimated error between the estimated value and the ground truth (40), the estimated error between the estimated value and the average of all measurements, and initial measurements versus final estimates for all sensor nodes.
   - **Results:** Visualizations show convergence of estimates, consensus among sensor nodes, and filtering process.

2. **Dynamic Sensor Network:**
   - **Analysis:** Similar analyses as in the static case, with the added complexity of changing network topology.
   - **Results:** Visualizations demonstrate the adaptation and convergence of consensus filters in dynamic environments.

**Conclusion:**
This project successfully implemented consensus filters for both static and dynamic sensor networks, demonstrating effective estimation, noise mitigation, and adaptation to changing network topologies.
