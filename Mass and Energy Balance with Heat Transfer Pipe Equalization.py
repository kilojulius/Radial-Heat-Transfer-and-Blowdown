import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Constants
length_pipe = 40 * 0.3048  # Convert 40 ft to meters
diameter_pipe = 6 * 0.0254  # Convert 6 inches to meters
volume_pipe = np.pi * (diameter_pipe / 2)**2 * length_pipe
time_total = 200  # Total simulation time in seconds
dt = 1  # Time step in seconds
num_time_steps = int(time_total / dt)
initial_temp_F = 68  # Ambient temperature
initial_temp_K = (initial_temp_F - 32) * 5.0 / 9.0 + 273.15
initial_pressure_Pa = 101325  # Atmospheric pressure
upstream_pressure_psig = 2700
upstream_pressure_Pa = upstream_pressure_psig * 6894.76 + 101325  # Convert PSIG to Pa
P_atm = 101325  # Atmospheric pressure in Pa
fluid = "CO2"
C_d = 0.8  # Discharge coefficient
orifice_diameter = 0.05  # Orifice diameter in meters (example value)
area_orifice = np.pi * (orifice_diameter / 2)**2  # Area of the orifice

# Calculate constant properties using CoolProp
rho_initial = CP.PropsSI('D', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
cp = CP.PropsSI('CPMASS', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
cv = CP.PropsSI('CVMASS', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
gamma = cp / cv

# Initial total mass in the pipe
initial_mass = rho_initial * volume_pipe

# Initialize arrays
P = np.full(num_time_steps + 1, initial_pressure_Pa)
T = np.full(num_time_steps + 1, initial_temp_K)
mass_flow_rate = np.zeros(num_time_steps + 1)
mass_in_pipe = np.full(num_time_steps + 1, initial_mass)
time = np.arange(0, time_total + dt, dt)

# Simulation loop
for t in range(1, num_time_steps + 1):
    if mass_in_pipe[t-1] <= 0:
        print("The pipe has run out of mass at time step:", t)
        break

    # Calculate mass flow rate from upstream to downstream
    if P[t-1] < upstream_pressure_Pa * (2 / (gamma + 1))**(gamma / (gamma - 1)):
        # Subsonic flow
        mass_flow_rate[t] = C_d * area_orifice * np.sqrt(2 * CP.PropsSI('D', 'P', upstream_pressure_Pa, 'T', T[t-1], fluid) * (upstream_pressure_Pa - P[t-1]))
    else:
        # Choked flow
        mass_flow_rate[t] = C_d * area_orifice * upstream_pressure_Pa * np.sqrt(gamma / (rho_initial * T[t-1]) * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)))

    # Mass balance
    d_mass = mass_flow_rate[t] * dt
    mass_in_pipe[t] = mass_in_pipe[t-1] + d_mass

    # Update pressure using ideal gas law
    rho_new = mass_in_pipe[t] / volume_pipe
    P[t] = rho_new * CP.PropsSI('R', fluid) * T[t-1]

    # Print intermediate results for the first few time steps
    if t <= 5 or t == num_time_steps:
        print(f"Time step {t}")
        print(f"Pressure (Pa): {P[t]}")
        print(f"Temperature (K): {T[t]}")
        print(f"Mass Flow Rate (kg/s): {mass_flow_rate[t]}")
        print(f"Mass in Pipe (kg): {mass_in_pipe[t]}")
        print("------------------------------------------------")

# Convert to PSI and Fahrenheit for plotting
P_psi = P * 0.000145037737797  # Convert Pa to PSI
T_F = (T - 273.15) * 9/5 + 32  # Convert K to Fahrenheit

# Trim arrays to actual simulation length
P_psi = P_psi[:t]
T_F = T_F[:t]
mass_flow_rate = mass_flow_rate[:t]
mass_in_pipe = mass_in_pipe[:t]
time = time[:t]

# Plot results
plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(time, P_psi, label='Pressure (PSI)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (PSI)')
plt.title('Pressure, Temperature, Mass Flow Rate, and Mass in Pipe over Time')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time, T_F, label='Temperature (°F)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time, mass_flow_rate, label='Mass Flow Rate (kg/s)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Mass Flow Rate (kg/s)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time, mass_in_pipe, label='Mass in Pipe (kg)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Mass in Pipe (kg)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
