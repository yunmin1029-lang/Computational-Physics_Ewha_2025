import matplotlib.pyplot as plt
import numpy as np 
import os
from scipy.optimize import curve_fit

os.chdir(os.path.abspath(os.path.dirname(__file__)))

bins, count = [], []
with open("hist2.csv", "r") as f:
    for line in f.readlines():
        _b, _c = [float(i) for i in line.split(",")]
        bins.append(_b)
        count.append(_c)

plt.plot(bins, count)
plt.show()

x_data = np.array(bins)
y_data = np.array(count)

def particle(x, a, b, c):
    return a*np.exp(-((x - b)**2)/(2*c**2))

def two_particles(x, a1, b1, c1, a2, b2, c2):
    return (a1*np.exp(-((x - b1)**2)/(2*c1**2)) +
            a2*np.exp(-((x - b2)**2)/(2*c2**2)))
    
popt, pcov = curve_fit(two_particles, x_data, y_data, 
                       p0 = [1200, 1.0, 0.5, 500, 2.0, 1.0])

a1, b1, c1, a2, b2, c2 = popt

plt.figure(figsize=(8,6))
plt.bar(x_data, y_data, width=0.05, color='black', alpha=0.3, label='Data Histogram')

x_fit = np.linspace(min(bins), max(bins), 1000)

peak1 = particle(x_fit, a1, b1, c1)
peak2 = particle(x_fit, a2, b2, c2)
total_fit = two_particles(x_fit, a1, b1, c1, a2, b2, c2)

plt.plot(x_fit, peak1, 'r', label='Peak 1 Fit')
plt.plot(x_fit, peak2, 'b', label='Peak 2 Fit')
plt.plot(x_fit, total_fit, 'black', label='Total Fit')
plt.xlabel('x')
plt.ylabel('Counts')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("----- Optimized Parameters -----")
print(f"Peak 1: A={a1: .2f}, B={b1: .2f}, C={c1: .2f}")
print(f"Peak 2: A={a2: .2f}, B={b2: .2f}, C={c2: .2f}")
