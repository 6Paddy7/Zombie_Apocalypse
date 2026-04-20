import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model import szr_system, get_jacobian_dfe
from analysis import check_stability

#1. Scenario Setup
#Parameters
P = 0.5       #Birth/Recruitment rate
delta = 0.01  #Natural death rate
beta = 0.03   #Infection rate
zeta = 0.1    #Re-animation rate (The "Feedback Loop")
alpha = 0.6  #Resistance

#Initial Populations
S0 = P / delta  #Start at equilibrium
Z0 = 1.0        #One single zombie enters the room
R0 = 0.0
y0 = [S0, Z0, R0]

#Time settings
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)

#2. Stability Analysis (The Math Part)
J_dfe = get_jacobian_dfe(P, beta, alpha, delta, zeta)
tr, det, stable = check_stability(J_dfe)

print(f"--- Stability Analysis for alpha = {alpha} ---")
print(f"Trace: {tr:.4f}")
print(f"Determinant: {det:.4f}")
print(f"Humanity is Stable: {stable}")
print("-" * 40)

#3. Numerical Simulation
#We pass the parameters using the 'args' tuple
solution = solve_ivp(
    szr_system, 
    t_span, 
    y0, 
    t_eval=t_eval, 
    args=(P, beta, alpha, delta, zeta)
)

#4. Plotting the Results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='Susceptible (S)', color='blue')
plt.plot(solution.t, solution.y[1], label='Zombies (Z)', color='red')
plt.plot(solution.t, solution.y[2], label='Removed (R)', color='green')

plt.title(f"SZR Zombie Simulation (Alpha={alpha}, Stable={stable})")
plt.xlabel("Time (Days)")
plt.ylabel("Population Count")
plt.legend()
plt.grid(True)
plt.show() 