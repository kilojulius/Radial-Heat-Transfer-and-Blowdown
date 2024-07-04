import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# Properties for CO2 at -20°C (initial condition)
T_fluid_initial = -20 + 273.15  # K
P_fluid = PropsSI('P', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Saturation pressure at -20°C
rho_fluid = PropsSI('D', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Density
cp_fluid = PropsSI('C', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Specific heat capacity
mu_fluid = PropsSI('V', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Dynamic viscosity
k_fluid = PropsSI('L', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Thermal conductivity
Pr_fluid = PropsSI('Prandtl', 'T', T_fluid_initial, 'Q', 1, 'CO2')  # Prandtl number

# Reynolds number calculation
mass_flow_rate = 10  # kg/s
inner_diameter = 0.1055  # m
area = (inner_diameter / 2) ** 2 * np.pi  # m^2
velocity = mass_flow_rate / (rho_fluid * area)  # m/s
Re_fluid = (rho_fluid * velocity * inner_diameter) / mu_fluid

# Nusselt number using Dittus-Boelter equation
Nu_fluid = 0.023 * (Re_fluid ** 0.8) * (Pr_fluid ** 0.4)
h_inner = Nu_fluid * k_fluid / inner_diameter  # W/m^2-K

# Free convection heat transfer coefficient (h_outer) estimation
V_air = 3.5  # average wind speed in m/s
T_air = 25 + 273.15  # K
Pr_air = PropsSI('Prandtl', 'T', T_air, 'P', 101325, 'Air')
k_air = PropsSI('L', 'T', T_air, 'P', 101325, 'Air')
mu_air = PropsSI('V', 'T', T_air, 'P', 101325, 'Air')
rho_air = PropsSI('D', 'T', T_air, 'P', 101325, 'Air')

# Assuming characteristic length = outer diameter of the pipe
outer_diameter = 0.114  # m
Re_air = (rho_air * V_air * outer_diameter) / mu_air
Nu_air = 0.3 + (0.62 * Re_air ** 0.5 * Pr_air ** (1/3)) / (1 + (0.4 / Pr_air) ** (2/3)) ** 0.25
h_outer = Nu_air * k_air / outer_diameter  # W/m^2-K

print(f"h_inner: {h_inner}, h_outer: {h_outer}, Re_fluid: {Re_fluid}, Pr_fluid: {Pr_fluid}, Nu_fluid: {Nu_fluid}, Re_air: {Re_air}, Pr_air: {Pr_air}, Nu_air: {Nu_air}")

# Given data and constants
alpha = 17.0 / (7820 * 0.450)  # Thermal diffusivity (m^2/s)
r_i = 0.05275  # Inner radius (m)
r_o = 0.057  # Outer radius (m)
T_initial = 30 + 273.15  # Initial temperature (K)
T_fluid = -20 + 273.15  # Fluid temperature (K)
T_env = 25 + 273.15  # Environmental temperature (K)
k = 17.0  # Thermal conductivity (W/m-K)
rho_wall = 7820  # Density of wall material (kg/m^3)
cp_wall = 450  # Specific heat capacity of wall material (J/kg-K)

# Discretization
dr = 0.001  # Radial step size (m)
max_dt = (dr**2) / (2 * alpha)  # Maximum allowable time step (s)
print(f'Maximum allowable time step for stability: {max_dt} s')
dt = 0.1  # Choosing a time step smaller than the maximum allowable time step
r = np.arange(r_i, r_o + dr, dr)
N = len(r)
num_time_steps = 1000  # Number of time steps

# Initialize temperature array
T = np.ones((N, num_time_steps + 1)) * T_initial

# Simulation loop
for t in range(1, num_time_steps + 1):
    for i in range(1, N - 1):
        T[i, t] = T[i, t-1] + alpha * dt * (
            (T[i+1, t-1] - 2*T[i, t-1] + T[i-1, t-1]) / dr**2 +
            (T[i+1, t-1] - T[i-1, t-1]) / (r[i] * dr)
        )
    
    # Inner surface boundary condition (convective with fluid)
    T[0, t] = T[0, t-1] + 2 * alpha * dt / dr**2 * (T[1, t-1] - T[0, t-1]) + 2 * h_inner * dt / (rho_wall * cp_wall * dr) * (T_fluid - T[0, t-1])
    
    # Outer surface boundary condition (convective with air)
    T[-1, t] = T[-1, t-1] + 2 * alpha * dt / dr**2 * (T[-2, t-1] - T[-1, t-1]) + 2 * h_outer * dt / (rho_wall * cp_wall * dr) * (T_env - T[-1, t-1])

# Convert temperature back to °C
T_Celsius = T[:, -1] - 273.15

# Plotting the temperature distribution
plt.plot(r, T_Celsius, label='Temperature distribution')
plt.xlabel('Radius (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
