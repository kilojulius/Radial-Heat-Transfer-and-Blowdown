import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt


# Function to calculate h_fluid using CoolProp
def calculate_h_fluid(T_film, P_fluid, m_flow):
    Pr = CP.PropsSI('Prandtl', 'T', T_film, 'P', P_fluid, 'CO2')
    k = CP.PropsSI('L', 'T', T_film, 'P', P_fluid, 'CO2')
    mu = CP.PropsSI('V', 'T', T_film, 'P', P_fluid, 'CO2')
    rho = CP.PropsSI('D', 'T', T_film, 'P', P_fluid, 'CO2')
    
    velocity = m_flow / (rho * area)  # m/s
    Re = (rho * velocity * (diameter_pipe - (2 * wall_thickness))) / mu
    Nu = 0.023 * Re**0.8 * Pr**0.4  # Dittus-Boelter equation for turbulent flow
    
    h = Nu * k / (diameter_pipe - (2 * wall_thickness))  # Convective heat transfer coefficient
    return h


# Constants
length_pipe = 15 #m
diameter_pipe = 6 * 0.0254  # Convert 6 inches to meters
wall_thickness = 0.005  # Pipe wall thickness in meters
volume_pipe = np.pi * (diameter_pipe / 2)**2 * length_pipe
surface_area_inner = np.pi * diameter_pipe * length_pipe  # Inner surface area of the pipe
surface_area_wall = np.pi * (diameter_pipe + 2 * wall_thickness) * length_pipe  # Outer surface area of the pipe wall
time_total = 10 * 60  # Total simulation time in seconds
dt = 0.5  # Time step in seconds
num_time_steps = int(time_total / dt)
initial_temp_F = 120
initial_temp_K = (initial_temp_F - 32) * 5.0 / 9.0 + 273.15
initial_pressure_psig = 2700
initial_pressure_Pa = initial_pressure_psig * 6894.76 + 101325  # Convert PSIG to Pa
P_atm = 101325  # Atmospheric pressure in Pa
fluid = "CO2"
C_d = 0.8  # Discharge coefficient
orifice_diameter = diameter_pipe/3 # Orifice diameter in meters (example value)
area_orifice = np.pi * (orifice_diameter / 2)**2  # Area of the orifice

area = np.pi * (diameter_pipe**2/4)

# Heat transfer parameters
k = 16  # Thermal conductivity of 316 stainless steel (W/m·K)
rho_wall = 8000  # Density of 316 stainless steel (kg/m³)
cp_wall = 500  # Specific heat capacity of 316 stainless steel (J/kg·K)

# Initial wall temperature
initial_wall_temp_K = initial_temp_K

# Calculate constant properties using CoolProp
rho_initial = CP.PropsSI('D', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
h_initial = CP.PropsSI('H', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
u_initial = CP.PropsSI('UMASS', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
cp = CP.PropsSI('CPMASS', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
cv = CP.PropsSI('CVMASS', 'P', initial_pressure_Pa, 'T', initial_temp_K, fluid)
gamma = cp / cv

# JT = CP.PropsSI('d(T)/d(P)|H', 'P',initial_pressure_Pa,'T', initial_temp_K,fluid)

# print(JT)

# Initial total mass in the pipe
initial_mass = rho_initial * volume_pipe

# Initialize arrays
P = np.full(num_time_steps + 1, initial_pressure_Pa)
T = np.full(num_time_steps + 1, initial_temp_K)
T_wall = np.full(num_time_steps + 1, initial_wall_temp_K)
mass_flow_rate = np.zeros(num_time_steps + 1)
mass_in_pipe = np.full(num_time_steps + 1, initial_mass)
internal_energy = np.full(num_time_steps + 1, initial_mass * u_initial)
enthalpy = np.full(num_time_steps + 1, initial_mass * h_initial)
entropy = np.zeros(num_time_steps + 1)
time = np.arange(0, time_total + dt, dt)

# Simulation loop
for t in range(1, num_time_steps + 1):
    if mass_in_pipe[t-1] <= 0:
        print("The pipe has run out of mass at time step:", t)
        break

    # Calculate mass flow rate
    if P[t-1] > P_atm * (2 / (gamma + 1))**(gamma / (gamma - 1)):
        # Choked flow
        mass_flow_rate[t] = C_d * area_orifice * P[t-1] * np.sqrt(gamma / (rho_initial * T[t-1]) * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)))
    else:
        # Subsonic flow
        mass_flow_rate[t] = C_d * area_orifice * np.sqrt(2 * rho_initial * (P[t-1] - P_atm))
    
    # Mass balance
    d_mass = -mass_flow_rate[t] * dt
    mass_in_pipe[t] = mass_in_pipe[t-1] + d_mass

    if mass_in_pipe[t] < 0:
        mass_in_pipe[t] = 0

    # Check for saturation before calculating h_out
    try:
        Q = CP.PropsSI('Q', 'P', P[t-1], 'T', T[t-1], fluid)
        if 0 <= Q <= 1:
            if Q < 1e-4:
                h_out = CP.PropsSI('H', 'P', P[t-1], 'Q', 0, fluid)
            elif Q > 1 - 1e-4:
                h_out = CP.PropsSI('H', 'P', P[t-1], 'Q', 1, fluid)
            else:
                h_out = CP.PropsSI('H', 'P', P[t-1], 'Q', Q, fluid)
        else:
            h_out = CP.PropsSI('H', 'P', P[t-1], 'T', T[t-1], fluid)
    except ValueError as e:
        print(f"Saturation check issue at time step {t}: {e}")
        h_out = h_initial  # Fallback to initial value

    # Energy balance
    dU = -mass_flow_rate[t] * h_out * dt
    internal_energy[t] = internal_energy[t-1] + dU
    enthalpy[t] = enthalpy[t-1] + dU

    try:
        if 0 <= Q <= 1:
            if Q < 1e-4:
                entropy[t] = CP.PropsSI('S', 'P', P[t], 'Q', 0, fluid)
            elif Q > 1 - 1e-4:
                entropy[t] = CP.PropsSI('S', 'P', P[t], 'Q', 1, fluid)
            else:
                entropy[t] = CP.PropsSI('S', 'P', P[t], 'Q', Q, fluid)
        else:
            entropy[t] = CP.PropsSI('S', 'P', P[t], 'T', T[t], fluid)
    except ValueError as e:
        print(f"Entropy calculation issue at time step {t}: {e}")
        entropy[t] = entropy[t-1]  # Keep the previous entropy
    
    
    # Update pressure and temperature using CoolProp with GERG-2008
    if mass_in_pipe[t] > 0:
        # Calculate new internal energy per unit mass
        u_new = internal_energy[t] / mass_in_pipe[t]
        # Calculate density
        rho_new = mass_in_pipe[t] / volume_pipe
        
        # Check for phase transition
        try:
            Q = CP.PropsSI('Q', 'UMASS', u_new, 'D', rho_new, fluid)
            print(f"Time step {t}, Quality (Q): {Q}, Density: {rho_new}, Internal Energy: {u_new}")
            if 0 <= Q <= 1:
                # If in two-phase region, handle saturation explicitly
                if Q < 1e-4:
                    P[t] = CP.PropsSI('P', 'T', T[t-1], 'Q', 0, fluid)
                    T[t] = CP.PropsSI('T', 'P', P[t], 'Q', 0, fluid)
                elif Q > 1 - 1e-4:
                    P[t] = CP.PropsSI('P', 'T', T[t-1], 'Q', 1, fluid)
                    T[t] = CP.PropsSI('T', 'P', P[t], 'Q', 1, fluid)
                else:
                    P[t] = CP.PropsSI('P', 'UMASS', u_new, 'D', rho_new, fluid)
                    T[t] = CP.PropsSI('T', 'P', P[t], 'Q', Q, fluid)
            else:
                # Single-phase region
                T[t] = CP.PropsSI('T', 'UMASS', u_new, 'D', rho_new, fluid)
                P[t] = CP.PropsSI('P', 'UMASS', u_new, 'D', rho_new, fluid)
        except ValueError as e:
            print(f"Phase transition or numerical issue at time step {t}: {e}")
            T[t] = T[t-1]  # Keep the previous temperature
            P[t] = P[t-1]  # Keep the previous pressure
            
        # try:
        #     Q = CP.PropsSI('Q', 'UMASS', u_new, 'D', rho_new, fluid)
        #     if 0 <= Q <= 1:
        #         h_liquid = CP.PropsSI('H', 'Q', 0, 'P', P[t-1], fluid)
        #         h_vapor = CP.PropsSI('H', 'Q', 1, 'P', P[t-1], fluid)
        #         h_out = Q * h_vapor + (1 - Q) * h_liquid
        #         T[t] = CP.PropsSI('T', 'Q', Q, 'P', P[t-1], fluid)
        #         P[t] = P[t-1]
        #     else:
        #         h_out = CP.PropsSI('H', 'UMASS', u_new, 'D', rho_new, fluid)
        #         T[t] = CP.PropsSI('T', 'UMASS', u_new, 'D', rho_new, fluid)
        #         P[t] = CP.PropsSI('P', 'UMASS', u_new, 'D', rho_new, fluid)
        # except ValueError as e:
        #     print(f"Phase transition or numerical issue at time step {t}: {e}")

        #     T[t] = T[t-1]  # Keep the previous temperature
        #     P[t] = P[t-1]  # Keep the previous pressure
    else:
        P[t] = P_atm
        T[t] = T[t-1]


    film_temp_inner = (T_wall[t] + T[t]) / 2
    h = calculate_h_fluid(film_temp_inner, P[t], mass_flow_rate[t])
    
    print(film_temp_inner, h, sep='\n')
    
    # Update wall temperature
    m_wall = rho_wall * (surface_area_wall * wall_thickness)
    Q_conv = h * surface_area_inner * (T[t] - T_wall[t-1])
    delta_T_wall = Q_conv * dt / (m_wall * cp_wall)
    T_wall[t] = T_wall[t-1] + delta_T_wall

    # Print intermediate results for the first few time steps
    if t <= 5 or t == num_time_steps:
        print(f"Time step {t}")
        print(f"Pressure (Pa): {P[t]}")
        print(f"Temperature (K): {T[t]}")
        print(f"Wall Temperature (K): {T_wall[t]}")
        print(f"Mass Flow Rate (kg/s): {mass_flow_rate[t]}")
        print(f"Mass in Pipe (kg): {mass_in_pipe[t]}")
        print(f"Internal Energy (J): {internal_energy[t]}")
        print("------------------------------------------------")

# Convert to PSI and Fahrenheit for plotting
P_psi = P * 0.000145037737797  # Convert Pa to PSI
T_F = (T - 273.15) * 9/5 + 32  # Convert K to Fahrenheit
T_wall_F = (T_wall - 273.15) * 9/5 + 32  # Convert K to Fahrenheit

# Trim arrays to actual simulation length
P_psi = P_psi[:t]
T_F = T_F[:t]
T_wall_F = T_wall_F[:t]
mass_flow_rate = mass_flow_rate[:t]
mass_in_pipe = mass_in_pipe[:t]
internal_energy = internal_energy[:t]
enthalpy = enthalpy[:t]
entropy = entropy[:t]
time = time[:t]

# Plot results
plt.figure(figsize=(15, 15))

plt.subplot(3, 2, 1)
plt.plot(time, P_psi, label='Pressure (PSI)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (PSI)')
plt.title('Pressure over Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(time, T_F, label='Fluid Temperature (°F)', color='orange')
plt.plot(time, T_wall_F, label='Wall Temperature (°F)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°F)')
plt.title('Temperature over Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(time, mass_flow_rate, label='Mass Flow Rate (kg/s)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Mass Flow Rate (kg/s)')
plt.title('Mass Flow Rate over Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(time, mass_in_pipe, label='Mass in Pipe (kg)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Mass in Pipe (kg)')
plt.title('Mass in Pipe over Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(time, enthalpy, label='Enthalpy (J)', color='purple')
plt.plot(time, internal_energy, label='Internal Energy (J)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Enthalpy and Internal Energy over Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(time, entropy, label='Entropy (J/K)', color='brown')
plt.xlabel('Time (s)')
plt.ylabel('Entropy (J/K)')
plt.title('Entropy over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
