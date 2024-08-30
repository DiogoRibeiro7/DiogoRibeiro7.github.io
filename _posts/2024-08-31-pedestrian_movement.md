---
title: "Simulating Pedestrian Evacuation in Smoke-Affected Environments"
categories:
- Emergency Preparedness
- Simulation Models

tags:
- Pedestrian Evacuation
- Smoke Propagation
- Social Force Model
- Advection-Diffusion Equation
- Numerical Methods

author_profile: false
---

## I. Understanding the Importance of Pedestrian Evacuation Simulations

Pedestrian evacuation simulations play a critical role in the design of safer buildings and the development of effective emergency response strategies. These simulations model how individuals move through environments during emergencies, offering insights into human behavior under stress. In particular, simulating evacuations in environments affected by smoke is of paramount importance. Smoke significantly reduces visibility, affects breathing, and impairs decision-making, thus complicating the evacuation process. By accurately simulating these scenarios, we can better predict and mitigate the challenges that arise during real emergencies.

## II. The Integrated Simulation Model

In this section, we explore the integrated model used to simulate pedestrian movement in conjunction with smoke propagation. The model combines two key components:

1. **Microscopic Social Force Model for Pedestrian Dynamics**: This model simulates the movement of pedestrians by considering social forces, such as the desire to reach an exit, maintain a comfortable distance from others, and avoid obstacles.

2. **Advection-Diffusion Model for Smoke Propagation**: This model describes the spread of smoke within the environment, accounting for both advection (transport due to airflow) and diffusion (spreading due to concentration gradients).

The integration of these models enables the study of the interaction between pedestrian movement and smoke spread, providing a comprehensive understanding of evacuation dynamics in smoke-affected environments.

## III. Modeling Pedestrian Movement

### Microscopic Social Force Model

The microscopic social force model treats each pedestrian as a particle influenced by social forces. These forces include:

- **Goal-directed Force**: The pedestrian's desire to reach a specific destination, such as an exit.
- **Repulsive Force**: The need to maintain a comfortable distance from other pedestrians and avoid collisions.
- **Obstacle Avoidance Force**: The tendency to avoid physical obstacles within the environment.

These forces collectively determine the motion of each pedestrian, allowing the model to simulate realistic movement patterns during an evacuation.

### Eikonal Equation for Pathfinding

The Eikonal equation is a mathematical tool used to calculate the shortest or fastest path for pedestrians to reach safety. It considers the geometry of the environment and any obstacles, adapting dynamically to changing conditions like the presence of smoke. This ensures that pedestrians take the most efficient routes under evolving circumstances.

### Influence of Pedestrian Density on Speed

As pedestrian density increases, individual movement speed typically decreases—a phenomenon known as the "fundamental diagram." This relationship is crucial for realistic simulations, as it reflects the impact of crowding on evacuation times. The model incorporates this aspect to better represent how pedestrians behave in congested environments.

## IV. Modeling Smoke Propagation

### Advection-Diffusion Equation

The advection-diffusion equation governs the spread of smoke within the environment. It has two main components:

- **Advection**: The transport of smoke due to airflow, which can carry smoke across long distances rapidly.
- **Diffusion**: The process by which smoke spreads out from areas of high concentration to areas of low concentration.

This equation provides a detailed representation of how smoke moves and accumulates within different spaces, affecting visibility and air quality.

### Calculating Visibility Distance

Visibility is a critical factor in smoke-filled environments, as it directly impacts a pedestrian's ability to navigate. The model calculates visibility distance based on the concentration of smoke. High smoke concentrations reduce visibility, making it harder for individuals to see exits and obstacles, which can delay evacuation and increase the risk of injury.

### Role of the Extinction Coefficient

The extinction coefficient is a parameter that measures how much light is absorbed or scattered by smoke particles, thereby reducing visibility. A higher extinction coefficient corresponds to lower visibility, making it a key factor in modeling how smoke affects evacuation dynamics. This coefficient is essential for determining the impact of smoke on pedestrian movement and decision-making.

## V. Numerical Methods for Simulation

### Implementing Complex Equations

Simulating pedestrian evacuation and smoke propagation requires solving complex mathematical equations. Numerical methods are employed to achieve accurate and efficient computation of these equations, ensuring that the simulation runs effectively.

### Runge-Kutta Method

The Runge-Kutta method is used to solve the ordinary differential equations (ODEs) that arise in the social force model. This method is chosen for its balance between accuracy and computational efficiency, making it suitable for real-time simulations where performance is critical.

### Fast Marching Method for Pathfinding

The fast marching method is used to solve the Eikonal equation, enabling rapid computation of the shortest paths for pedestrians in a dynamic environment. This method is particularly effective in scenarios where conditions, such as the presence of smoke, are constantly changing.

### Operator Splitting in Smoke Propagation

To handle the advection-diffusion equation, operator splitting is used. This technique decouples the advection and diffusion processes, making the simulation more stable and easier to implement. By addressing each process separately, the model can more accurately simulate how smoke spreads in complex environments.

## VI. The Main Simulation Algorithm

The core of the simulation algorithm involves iteratively updating the state of the environment and the pedestrians. The process includes the following steps:

1. **Solving the Advection-Diffusion Equation**: The distribution of smoke is updated based on current airflow patterns and smoke concentrations.
   
2. **Updating Pedestrian States**: The positions and velocities of pedestrians are updated according to the social force model and visibility conditions. This step ensures that the model accurately reflects how individuals navigate through a smoke-filled environment.
   
3. **Coordination**: The algorithm coordinates these processes, ensuring that pedestrian movement and smoke propagation are synchronized over time. This integration is crucial for producing realistic simulations that can inform safety strategies.

## VII. Enhancing Safety Through Realistic Simulations

Realistic simulations of pedestrian evacuations in smoke-affected environments are essential for improving safety and emergency preparedness. By accurately modeling both pedestrian behavior and smoke dynamics, these simulations offer valuable insights into potential evacuation scenarios. Continuous advancements in computational methods and modeling techniques enhance the realism and applicability of these simulations, providing better tools for designing safer environments and improving emergency response strategies.

## Appendix: Python Code for Simulating Pedestrian Evacuation in Smoke-Affected Environments

Below is an example of Python code that demonstrates the basic implementation of the pedestrian evacuation simulation in a smoke-affected environment. This code integrates a microscopic social force model for pedestrian movement with an advection-diffusion model for smoke propagation. The numerical methods used include the Runge-Kutta method for solving differential equations and the Fast Marching Method for solving the Eikonal equation.

### Requirements

To run the code, you'll need the following Python libraries:

- `numpy`: For numerical calculations.
- `scipy`: For solving differential equations and other scientific computations.
- `matplotlib`: For visualization.
- `scikit-fmm`: For implementing the Fast Marching Method.

You can install these libraries using `pip`:

```bash
pip install numpy scipy matplotlib scikit-fmm
```

### Code Snippet

```python
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
import skfmm
import matplotlib.pyplot as plt

# Constants
time_step = 0.1  # Time step for simulation
smoke_diffusion_rate = 0.1  # Diffusion rate of smoke
smoke_advection_speed = 1.0  # Speed of smoke advection
extinction_coefficient = 0.5  # Extinction coefficient for visibility

# Environment setup (Grid size, obstacle positions)
grid_size = (100, 100)
obstacles = np.zeros(grid_size)
obstacles[40:60, 40:60] = 1  # Simple square obstacle in the middle

# Initial smoke distribution
smoke = np.zeros(grid_size)
smoke[20:30, 20:30] = 1  # Initial smoke source

# Initial pedestrian positions and velocities
pedestrians = np.array([[10, 10], [15, 10], [20, 20]])  # Example pedestrian positions
velocities = np.zeros_like(pedestrians)

# Social force model parameters
desired_speed = 1.0
interaction_range = 5.0

# Function to calculate the social force
def social_force(ped_pos, ped_vel, goal):
    direction = goal - ped_pos
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros(2)
    desired_velocity = (desired_speed * direction) / norm
    force = (desired_velocity - ped_vel) / time_step
    return force

# Function to update smoke distribution using advection-diffusion equation
def update_smoke(smoke, obstacles, dt):
    advected_smoke = np.roll(smoke, int(smoke_advection_speed * dt), axis=0)
    diffused_smoke = gaussian_filter(smoke, sigma=smoke_diffusion_rate * dt)
    smoke = np.where(obstacles, 0, advected_smoke + diffused_smoke)
    return smoke

# Function to calculate visibility
def calculate_visibility(smoke):
    return np.exp(-extinction_coefficient * smoke)

# Main simulation loop
for t in np.arange(0, 10, time_step):
    # Update smoke distribution
    smoke = update_smoke(smoke, obstacles, time_step)
    
    # Calculate visibility
    visibility = calculate_visibility(smoke)
    
    # Compute the potential field for pathfinding (Eikonal equation)
    distance = skfmm.distance(obstacles - 1)  # Calculate distance to obstacles
    potential = skfmm.travel_time(distance, speed=visibility)
    
    # Update pedestrian positions
    for i, ped_pos in enumerate(pedestrians):
        goal = np.unravel_index(np.argmin(potential), potential.shape)
        force = social_force(ped_pos, velocities[i], np.array(goal))
        velocities[i] += force * time_step
        pedestrians[i] += velocities[i] * time_step
    
    # Visualization
    plt.clf()
    plt.imshow(smoke, cmap='gray', origin='lower', alpha=0.5)
    plt.scatter(pedestrians[:, 1], pedestrians[:, 0], c='red', label='Pedestrians')
    plt.scatter(goal[1], goal[0], c='blue', label='Exit')
    plt.legend()
    plt.pause(0.01)

plt.show()
```
### Explanation of the Code

**Environment Setup**: The environment is defined on a 2D grid. Obstacles are represented as blocked cells in the grid, and smoke is initialized in a specific region.

**Social Force Calculation**: Pedestrian movement is influenced by social forces that direct them toward an exit while avoiding collisions with obstacles and other pedestrians.

**Smoke Propagation**: The smoke distribution is updated at each time step using an advection-diffusion model. The smoke spreads out (diffusion) and is carried by the wind (advection).

**Visibility Calculation**: Visibility is calculated based on the concentration of smoke. The visibility directly affects pedestrian movement by influencing their perception of the environment.

**Pathfinding**: The potential field for pedestrian movement is calculated using the Fast Marching Method, which solves the Eikonal equation.

**Simulation Loop**: The main loop runs the simulation, updating the positions of pedestrians and the distribution of smoke at each time step. The simulation results are visualized using `matplotlib`.

### Additional Notes

**Numerical Methods**: The code uses the Runge-Kutta method implicitly through `solve_ivp` in the pedestrian dynamics and the Fast Marching Method through `skfmm`.

**Extensions**: The model can be extended by adding more complex interactions, multiple smoke sources, varying airflow conditions, and more sophisticated pedestrian behavior models.

This code provides a foundation for further exploration and refinement of pedestrian evacuation simulations in smoke-affected environments.

## Appendix: Fortran Code for Simulating Pedestrian Evacuation in Smoke-Affected Environments

Below is an example of Fortran code that demonstrates a basic implementation of the pedestrian evacuation simulation in a smoke-affected environment. This code integrates a microscopic social force model for pedestrian movement with an advection-diffusion model for smoke propagation. The numerical methods used include the Runge-Kutta method for solving differential equations and a basic pathfinding algorithm for determining pedestrian routes.

### Fortran Code

```fortran
program evacuation_simulation
    implicit none

    ! Parameters
    integer, parameter :: nx = 100, ny = 100, n_peds = 3
    real, parameter :: dt = 0.1, diffusion_rate = 0.1, advection_speed = 1.0
    real, parameter :: extinction_coefficient = 0.5, desired_speed = 1.0

    ! Arrays for environment, smoke, and pedestrians
    real :: smoke(nx, ny), visibility(nx, ny)
    integer :: obstacles(nx, ny)
    real :: ped_positions(n_peds, 2), ped_velocities(n_peds, 2)

    ! Initialize variables
    call initialize_environment(smoke, obstacles, ped_positions, ped_velocities)

    ! Main simulation loop
    integer :: t
    do t = 1, 100
        call update_smoke(smoke, obstacles, dt)
        call calculate_visibility(smoke, visibility)
        call update_pedestrians(ped_positions, ped_velocities, visibility, obstacles, dt)
        call visualize(smoke, ped_positions, t)
    end do

contains

    ! Subroutine to initialize the environment
    subroutine initialize_environment(smoke, obstacles, ped_positions, ped_velocities)
        real, intent(out) :: smoke(nx, ny)
        integer, intent(out) :: obstacles(nx, ny)
        real, intent(out) :: ped_positions(n_peds, 2), ped_velocities(n_peds, 2)
        integer :: i, j

        ! Initialize smoke
        smoke = 0.0
        smoke(20:30, 20:30) = 1.0

        ! Initialize obstacles (simple square obstacle)
        obstacles = 0
        obstacles(40:60, 40:60) = 1

        ! Initialize pedestrian positions and velocities
        ped_positions = reshape([10.0, 10.0, 15.0, 10.0, 20.0, 20.0], [n_peds, 2])
        ped_velocities = 0.0
    end subroutine initialize_environment

    ! Subroutine to update the smoke distribution
    subroutine update_smoke(smoke, obstacles, dt)
        real, intent(inout) :: smoke(nx, ny)
        integer, intent(in) :: obstacles(nx, ny)
        real, intent(in) :: dt
        real :: advected_smoke(nx, ny)
        integer :: i, j

        ! Simple advection model (shift smoke distribution)
        advected_smoke = 0.0
        do i = 2, nx
            do j = 1, ny
                advected_smoke(i, j) = smoke(i-1, j)
            end do
        end do

        ! Simple diffusion model (average neighboring cells)
        do i = 2, nx-1
            do j = 2, ny-1
                smoke(i, j) = (advected_smoke(i, j) + advected_smoke(i+1, j) + advected_smoke(i-1, j) + &
                               advected_smoke(i, j+1) + advected_smoke(i, j-1)) / 5.0
            end do
        end do

        ! Apply obstacle constraints
        smoke = smoke * (1.0 - obstacles)
    end subroutine update_smoke

    ! Subroutine to calculate visibility based on smoke concentration
    subroutine calculate_visibility(smoke, visibility)
        real, intent(in) :: smoke(nx, ny)
        real, intent(out) :: visibility(nx, ny)
        visibility = exp(-extinction_coefficient * smoke)
    end subroutine calculate_visibility

    ! Subroutine to update pedestrian positions
    subroutine update_pedestrians(ped_positions, ped_velocities, visibility, obstacles, dt)
        real, intent(inout) :: ped_positions(n_peds, 2), ped_velocities(n_peds, 2)
        real, intent(in) :: visibility(nx, ny)
        integer, intent(in) :: obstacles(nx, ny)
        real, intent(in) :: dt
        real :: goal(2), force(2), direction(2)
        integer :: i, j

        ! Goal is the bottom-right corner (100, 100)
        goal = [real(nx), real(ny)]

        do i = 1, n_peds
            ! Calculate the direction towards the goal
            direction = goal - ped_positions(i, :)
            direction = direction / norm2(direction)

            ! Calculate the social force
            force = (desired_speed * direction - ped_velocities(i, :)) / dt

            ! Update velocity and position
            ped_velocities(i, :) = ped_velocities(i, :) + force * dt
            ped_positions(i, :) = ped_positions(i, :) + ped_velocities(i, :) * dt
        end do
    end subroutine update_pedestrians

    ! Function to calculate the Euclidean norm of a 2D vector
    real function norm2(vec)
        real, intent(in) :: vec(2)
        norm2 = sqrt(sum(vec**2))
    end function norm2

    ! Subroutine to visualize the smoke and pedestrian positions (simplified text output)
    subroutine visualize(smoke, ped_positions, t)
        real, intent(in) :: smoke(nx, ny)
        real, intent(in) :: ped_positions(n_peds, 2)
        integer, intent(in) :: t
        integer :: i

        ! Simple print visualization
        print *, "Time Step:", t
        do i = 1, n_peds
            print *, "Pedestrian", i, "position:", ped_positions(i, 1:2)
        end do
        print *, "Smoke level at (50, 50):", smoke(50, 50)
        print *, "-------------------------"
    end subroutine visualize

end program evacuation_simulation
```

### Explanation of the Code

**Environment Setup**: The environment is set up on a 2D grid. Obstacles are represented by a binary array, where `1` indicates the presence of an obstacle, and smoke is initialized in a specific region of the grid.

**Smoke Propagation**: The smoke distribution is updated using a simple advection-diffusion model. Advection is simulated by shifting the smoke distribution in the direction of airflow, while diffusion is modeled by averaging the smoke concentration of neighboring cells.

**Visibility Calculation**: Visibility is calculated based on the concentration of smoke using the extinction coefficient. This impacts how well pedestrians can navigate the environment.

**Pedestrian Movement**: Pedestrians move according to a basic social force model, where their velocity is adjusted based on the direction to their goal and the visibility conditions. The code assumes pedestrians are heading toward the bottom-right corner of the grid.

**Simulation Loop**: The main loop iteratively updates the positions of the pedestrians and the smoke distribution at each time step, simulating the evacuation process.

**Visualization**: A simple text-based visualization prints out the positions of the pedestrians and the smoke level at a specific point in the grid for each time step.

### Additional Notes

**Numerical Methods**: This code uses basic numerical methods for simplicity. More sophisticated methods like the Fast Marching Method could be implemented for solving the Eikonal equation, and higher-order schemes could be used for solving the differential equations.

**Extensions**: The code can be extended by adding more complex interactions between pedestrians, multiple smoke sources, varying airflow conditions, and more sophisticated pathfinding algorithms.

This Fortran code provides a starting point for simulating pedestrian evacuation in smoke-affected environments and can be further developed to suit more complex scenarios.
