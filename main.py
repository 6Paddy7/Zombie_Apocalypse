import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model import szr_system, get_jacobian_dfe
from analysis import check_stability

P = 1.0       
delta = 0.0001   
beta = 0.00005     
zeta = 0.004     
alpha = 0.0021

S0 = 10000.0     
Z0 = 10.0        
R0 = 0.0
y0 = [S0, Z0, R0]

t_span = (0, 1000)            
t_eval = np.linspace(0, 1000, 2000)

J_dfe = get_jacobian_dfe(P, beta, alpha, delta, zeta)
tr, det, stable = check_stability(J_dfe)

print(f"--- Stability Analysis for alpha = {alpha} ---")
print(f"Trace: {tr:.4f}")
print(f"Determinant: {det:.4f}")
print(f"Humanity is Stable: {stable}")
print("-" * 40)

solution = solve_ivp(
    szr_system, 
    t_span, 
    y0, 
    t_eval=t_eval, 
    args=(P, beta, alpha, delta, zeta)
)

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