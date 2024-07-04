import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from matplotlib.animation import FuncAnimation


# Function to calculate h_fluid using CoolProp
def calculate_h_fluid(T_film, P_fluid):
    Pr = PropsSI('Prandtl', 'T', T_film, 'P', P_fluid, 'CO2')
    k = PropsSI('L', 'T', T_film, 'P', P_fluid, 'CO2')
    mu = PropsSI('V', 'T', T_film, 'P', P_fluid, 'CO2')
    rho = PropsSI('D', 'T', T_film, 'P', P_fluid, 'CO2')
    
    velocity = mass_flow_rate / (rho * area)  # m/s
    Re = (rho * velocity * inner_diameter) / mu
    Nu = 0.023 * Re**0.8 * Pr**0.4  # Dittus-Boelter equation for turbulent flow
    
    h = Nu * k / inner_diameter  # Convective heat transfer coefficient
    return h


# Constants
length_pipe = 3  # Convert 40 ft to meters
diameter_pipe = 141/1000  # Convert 6 inches to meters
wall_thickness = 0.034  # Pipe wall thickness in meters

inner_diameter = diameter_pipe - 2 * wall_thickness  # m
outer_diameter = diameter_pipe  # m

inner_radius = diameter_pipe / 2
outer_radius = inner_radius + wall_thickness
time_total = 8 * 60  # Total simulation time in seconds
dt = 1  # Time step in seconds
num_time_steps = int(time_total / dt)

T_initial = 34  # Initial temperature in Celsius
T_initial_K = T_initial + 273.15  # Convert to Kelvin
T_fluid = -55 + 273.15  # Fluid temperature in Kelvin

V_air = 3.5  # average wind speed in m/s
T_air = 28 + 273.15  # K

# https://www.matweb.com/search/datasheet.aspx?matguid=7b95733c08f4422a8113a1a931f716ab&ckck=1
k = 17  # Thermal conductivity of 25 Cr UNS S32750 (W/m·K)
rho_wall = 7820  # Density of 25 Cr UNS S32750 (kg/m³)
cp_wall = 450  # Specific heat capacity of 25 Cr UNS S32750 (J/kg·K)
alpha = k / (rho_wall * cp_wall)  # Thermal diffusivity

mass_flow_rate = 20  # kg/s

# Properties for CO2 at (initial condition)
T_fluid_initial = (T_fluid + T_initial_K) / 2 # K
P_fluid = PropsSI('P', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Saturation pressure at
rho_fluid = PropsSI('D', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Density
cp_fluid = PropsSI('C', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Specific heat capacity
mu_fluid = PropsSI('V', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Dynamic viscosity
k_fluid = PropsSI('L', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Thermal conductivity
Pr_fluid = PropsSI('Prandtl', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Prandtl number

# Reynolds number calculation
area = (inner_diameter / 2) ** 2 * np.pi  # m^2
velocity = mass_flow_rate / (rho_fluid * area)  # m/s
Re_fluid = (rho_fluid * velocity * inner_diameter) / mu_fluid

# Nusselt number using Dittus-Boelter equation
Nu_fluid = 0.023 * (Re_fluid ** 0.8) * (Pr_fluid ** 0.4)
h_inner = Nu_fluid * k_fluid / inner_diameter  # W/m^2-K

# Free convection heat transfer coefficient (h_outer) estimation
Pr_air = PropsSI('Prandtl', 'T', T_air, 'P', 101325, 'Air')
k_air = PropsSI('L', 'T', T_air, 'P', 101325, 'Air')
mu_air = PropsSI('V', 'T', T_air, 'P', 101325, 'Air')
rho_air = PropsSI('D', 'T', T_air, 'P', 101325, 'Air')

# Assuming characteristic length = outer diameter of the pipe

Re_air = (rho_air * V_air * outer_diameter) / mu_air
Nu_air = 0.3 + (0.62 * Re_air ** 0.5 * Pr_air ** (1/3)) / (1 + (0.4 / Pr_air) ** (2/3)) ** 0.25
h_outer = Nu_air * k_air / outer_diameter  # W/m^2-K

print(f"h_inner: {h_inner}, h_outer: {h_outer}, Re_fluid: {Re_fluid}, Pr_fluid: {Pr_fluid}, Nu_fluid: {Nu_fluid}, Re_air: {Re_air}, Pr_air: {Pr_air}, Nu_air: {Nu_air}")

h_fluid = h_inner
h_air = h_outer

# Discretize the radial coordinate
dr = 0.005  # Radial step size in meters
r = np.arange(inner_radius, outer_radius + dr, dr)
num_radial_steps = len(r)

# Initialize temperature distribution
T = np.full((num_radial_steps, num_time_steps + 1), T_initial_K)
h_over_time = np.full(num_time_steps+1, h_fluid)

# Simulation loop
for t in range(1, num_time_steps + 1):
    # Update heat transfer coefficients based on film temperature
    film_temp_inner = (T[0, t - 1] + T_fluid) / 2
    h_fluid = calculate_h_fluid(film_temp_inner, P_fluid=P_fluid+10)  # Update h_fluid using CoolProp  
    h_over_time[t] = h_fluid

    # Inner surface boundary condition (convective with fluid)
    T[0, t] = T[0, t-1] + 2 * alpha * dt / dr**2 * (T[1, t-1] - T[0, t-1]) + 2 * h_fluid * dt / (rho_wall * cp_wall * dr) * (T_fluid - T[0, t-1])  
    
    for i in range(1, num_radial_steps - 1):  #Conduction through pipe wall
        T[i, t] = T[i, t-1] + alpha * dt * ((T[i+1, t-1] - 2*T[i, t-1] + T[i-1, t-1]) / dr**2 + (T[i+1, t-1] - T[i-1, t-1]) / (r[i] * dr))
    

    # Outer surface boundary condition (convective with air)
    T[-1, t] = T[-1, t-1] + 2 * alpha * dt / dr**2 * (T[-2, t-1] - T[-1, t-1]) + 2 * h_air * dt / (rho_wall * cp_wall * dr) * (T_air - T[-1, t-1])

# Convert temperature to Celsius for plotting
T_C = T - 273.15

# Plot results
plt.figure(figsize=(10, 6))
for i in range(0, num_time_steps + 1, int(num_time_steps / 10)):
    plt.plot(r, T_C[:, i], label=f'Time = {i*dt} s')
plt.xlabel('Radius (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('Temperature Distribution in Pipe Wall Over Time')
plt.grid(True)
plt.show()

# """
# Animated Line
# """

# # Set up the figure and axis
# fig, ax = plt.subplots()
# line, = ax.plot(r, T_C[:, 0], color='blue')
# ax.set_xlabel('Radius (m)')
# ax.set_ylabel('Temperature (°C)')
# ax.set_title('Temperature Distribution in Pipe Wall Over Time')
# ax.grid(True)
# ax.set_ylim(-55, 38)  # Set y-axis range

# # Animation update function
# def update(frame):
#     line.set_ydata(T_C[:, frame])
#     ax.set_title(f'Temperature Distribution at Time = {frame*dt:.0f} s')  # Ensure title updates correctly
#     return line,

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_time_steps + 1, blit=False)

# # Display the animation
# plt.show()

"""
Animated Heatmap
"""
# # Set up the figure and axis for heatmap
# fig, ax = plt.subplots()
# heatmap = ax.imshow(T_C[:, 0].reshape(1, -1), aspect='auto', cmap='hot', origin='lower',
#                     extent=[r.min(), r.max(), 0, 1], vmin=-55, vmax=35)
# ax.set_xlabel('Radius (m)')
# ax.get_yaxis().set_visible(False)  # Remove y-axis
# fig.colorbar(heatmap, ax=ax, label='Temperature (°C)')
# ax.set_title('Temperature Distribution in Pipe Wall Over Time')

# # Animation update function
# def update(frame):
#     heatmap.set_array(T_C[:, frame].reshape(1, -1))
#     ax.set_title(f'Temperature Distribution at Time = {frame*dt:.0f} s')  # Ensure title updates correctly
#     return heatmap,

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_time_steps + 1, interval=50, blit=False)  # Set blit to False

# # Display the animation
# plt.show()