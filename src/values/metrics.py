import sympy as sp

#for futer use in symbolic calculations
M = sp.symbols("M")
r = sp.symbols("r")
L = sp.symbols("L")
G = sp.symbols("G")
c = sp.symbols("c")

#defining all the metrics
metrics = {"s":[1-2*M*G/(c**2*r),r, "Schwarzschild"], 
           "h_LCC":[1-2*M*G*r**2/(c**2*r**3+2*M*G*L**2),r,"Hayward LCC"], 
           "h_REG":[1-2*M*G*r**2/(c**2*(r**3+L**3)),r, "Hayward REG"], 
           "b_REG":[1-2*M*G*r**2/(c**2*(r**2+L**2)**(3/2)),r, "Bardeen REG"], 
           "b_LCC":[1-2*M*G*r**2/(c**2*(r**2+(2*M*G/c**2)**(2/3)*L**(4/3))**(3/2)),r, "Bardeen LCC"], 
           "d_REG":[1-2*M*G*(1-sp.exp(-r**3/L**3))/(c**2*r),r, "Dymnikova REG"]}
