import pickle
import matplotlib.pyplot as plt
import numpy as np

# Su-Olson Surface Source Numerical Results (1996)
radiation_at_1ns = [0.90895, 0.90107, 0.88926, 0.86965, 0.85011, 0.83067, 0.71657, 0.54080, 0.38982, 0.26789, 0.10906, 0.03624]
radiation_at_p1ns = [0.73611, 0.71338, 0.67978, 0.62523, 0.57274, 0.52255, 0.27705, 0.07075, 0.01271, 0.00167, 0.00002]
radiation_at_p01ns = [0.46599, 0.42133, 0.36020, 0.27323, 0.20332, 0.14837, 0.01441, 0.00005, 0.00001]
material_at_1ns = [0.90849, 0.90057, 0.88871, 0.86900, 0.84937, 0.82983, 0.71521, 0.53877, 0.38745, 0.26551, 0.10732, 0.03534]
material_at_p1ns = [0.72328, 0.69946, 0.66432, 0.60749, 0.55308, 0.50134, 0.25413, 0.05936, 0.00968, 0.00115, 0.00001]
material_at_p01ns = [0.24762, 0.21614, 0.17530, 0.12182, 0.08306, 0.05556, 0.00324, 0.00001]

x_values = [0, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 15, 20]

fig = plt.figure(figsize=(12, 6))

# Plot analytical results
plt.plot(x_values, radiation_at_1ns, label='Analytical RadTemp @ 1 ns', linestyle='--', color='red')
plt.plot(x_values, material_at_1ns, label='Analytical Temp @ 1 ns', linestyle='--', color='blue')
plt.plot(x_values[:len(radiation_at_p1ns)], radiation_at_p1ns, label='Analytical RadTemp @ 0.1 ns', linestyle='--', color='red')
plt.plot(x_values[:len(material_at_p1ns)], material_at_p1ns, label='Analytical Temp @ 0.1 ns', linestyle='--', color='blue')
plt.plot(x_values[:len(radiation_at_p01ns)], radiation_at_p01ns, label='Analytical RadTemp at 0.01 ns', linestyle='--', color='red')
plt.plot(x_values[:len(material_at_p01ns)], material_at_p01ns, label='Analytical Temp at 0.01 ns', linestyle='--', color='blue')

# Open the output file
with open("debug.out", "rb") as fname:
    # Loop to read and plot data from the file
    times = ["0.01 ns", "0.1 ns", "1 ns"]  # Corresponds to plot times
    for i in range(3):
        fname.readline()  # Skip the time line
        xdata = pickle.load(fname)  # cellpos
        temp = pickle.load(fname)   # temp
        radtemp = pickle.load(fname)  # radtemp

        plt.plot(xdata, temp, label=f"Temp @ {times[i]}", color='blue')
        plt.plot(xdata, radtemp, label=f"RadTemp @ {times[i]}", color='red')

# Set axis ticks, labels, and limits
plt.xticks(np.arange(0, 21, 1))
plt.yticks(np.arange(0, 1.3, 0.1))
plt.xlim(0, 20.0)
plt.ylim(0, 1.2)
plt.xlabel("x - cm")
plt.ylabel("T - keV")

# Add legend
plt.legend(loc='upper right', frameon=False)

# Save figure
plt.savefig("debug.png", bbox_inches="tight", dpi=900)

# Optionally show the plot
plt.show()
