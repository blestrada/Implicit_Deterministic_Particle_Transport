import pickle
import matplotlib.pyplot as plt
import numpy as np

"""Benchmark Solutions"""
x_bench = [0.01000, 0.10000, 0.17783, 0.31623, 0.45000, 0.50000, 0.56234, 0.75000, 1.00000, 1.33352, 1.77828, 3.16228, 5.62341, 10.00000, 17.78279]

t_bench = [0.10000, 0.31623, 1.00000, 3.16228, 10.00000, 31.6228, 100.000]

rad_benchone = [[0.09531, 0.09531, 0.09532, 0.09529, 0.08823, 0.04765, 0.00375, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], 
             [0.27526, 0.27526, 0.27527, 0.26262, 0.20312, 0.13762, 0.06277, 0.00280, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
             [0.64308, 0.63585, 0.61958, 0.56187, 0.44711, 0.35801, 0.25374, 0.11430, 0.03648, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
             [1.20052, 1.18869, 1.16190, 1.07175, 0.90951, 0.79902, 0.66678, 0.44675, .027540, 0.14531, 0.05968, 0.00123, 0.00000, 0.00000, 0.00000],
             [2.23575, 2.21944, 2.18344, 2.06448, 1.86072, 1.73178, 1.57496, 1.27398, 0.98782, 0.70822, 0.45016, 0.09673, 0.00375, 0.00000, 0.00000],
             [0.69020, 0.68974, 0.68878, 0.68569, 0.68111, 0.67908, 0.67619, 0.66548, 0.64691, 0.61538, 0.56353, 0.36965, 0.10830, 0.00390, 0.00000],
             [0.35720, 0.35714, 0.35702, 0.35664, 0.35599, 0.35574, 0.35538, 0.35393, 0.35141, 0.34697, 0.33924, 0.30346, 0.21382, 0.07200, 0.00272]
            ] # ca = 1, cs = 0 

rad_benchtwo = [[0.09757, 0.09757, 0.09758, 0.09756, 0.09033, 0.04878, 0.00383, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], 
                [0.29363, 0.29365, 0.29364, 0.28024, 0.21573, 0.14681, 0.06783, 0.00292, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.72799, 0.71888, 0.69974, 0.63203, 0.50315, 0.40769, 0.29612, 0.13756, 0.04396, 0.00324, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [1.28138, 1.26929, 1.24193, 1.15018, 0.98599, 0.87477, 0.74142, 0.51563, 0.33319, 0.18673, 0.08229, 0.00160, 0.00000, 0.00000, 0.00000],
                [2.26474, 2.24858, 2.21291, 2.09496, 1.89259, 1.76429, 1.60822, 1.30947, 1.02559, 0.74721, 0.48739, 0.11641, 0.00554, 0.00000, 0.00000],
                [0.68703, 0.68656, 0.68556, 0.68235, 0.67761, 0.67550, 0.67252, 0.66146, 0.64239, 0.61024, 0.55789, 0.36631, 0.11177, 0.00491, 0.00000],
                [0.35675, 0.35668, 0.35654, 0.35618, 0.35552, 0.35527, 0.35491, 0.35346, 0.35092, 0.34646, 0.33868, 0.30281, 0.21323, 0.07236, 0.00296]
               ] # ca = 0.5, cs = 0.5

mat_benchone = [[0.00468, 0.00468, 0.00468, 0.00468, 0.00455, 0.00234, 0.00005, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], 
                [0.04093, 0.04093, 0.04093, 0.04032, 0.03314, 0.02046, 0.00635, 0.00005, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.27126, 0.26839, 0.26261, 0.23978, 0.18826, 0.14187, 0.08838, 0.03014, 0.00625, 0.00017, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.94670, 0.93712, 0.91525, 0.84082, 0.70286, 0.60492, 0.48843, 0.30656, 0.17519, 0.08352, 0.02935, 0.00025, 0.00000, 0.00000, 0.00000],
                [2.11186, 2.09585, 2.06052, 1.94365, 1.74291, 1.61536, 1.46027, 1.16591, 0.88992, 0.62521, 0.38688, 0.07642, 0.00253, 0.00000, 0.00000],
                [0.70499, 0.70452, 0.70348, 0.70020, 0.69532, 0.69308, 0.68994, 0.67850, 0.65868, 0.62507, 0.57003, 0.36727, 0.10312, 0.00342, 0.00000],
                [0.35914, 0.35908, 0.35895, 0.35854, 0.35793, 0.35766, 0.35728, 0.35581, 0.35326, 0.34875, 0.34086, 0.30517, 0.21377, 0.07122, 0.00261]
               ] # ca = 1, cs = 0
       
mat_benchtwo = [[0.00242, 0.00242, 0.00242, 0.00242, 0.00235, 0.00121, 0.00003, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.02255, 0.02253, 0.02256, 0.02223, 0.01826, 0.01128, 0.00350, 0.00003, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.17609, 0.17420, 0.17035, 0.15520, 0.12164, 0.09194, 0.05765, 0.01954, 0.00390, 0.00009, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.77654, 0.76878, 0.75108, 0.69082, 0.57895, 0.49902, 0.40399, 0.25610, 0.14829, 0.07161, 0.02519, 0.00018, 0.00000, 0.00000, 0.00000],
                [2.00183, 1.98657, 1.95286, 1.84104, 1.64778, 1.52383, 1.37351, 1.09216, 0.83248, 0.58640, 0.36629, 0.07658, 0.00290, 0.00000, 0.00000],
                [0.71860, 0.71805, 0.71687, 0.71312, 0.70755, 0.70499, 0.70144, 0.68851, 0.66637, 0.62937, 0.57001, 0.36066, 0.10181, 0.00385, 0.00000],
                [0.36067, 0.36065, 0.36047, 0.36005, 0.35945, 0.35917, 0.35876, 0.35727, 0.35465, 0.35004, 0.34200, 0.30553, 0.21308, 0.07077, 0.00273]
               ] # ca = 0.5, cs = 0.5

# Create figure and axis for combined plot
plt.figure(figsize=(8, 5))

# Analytical results (dashed lines)
# plt.plot(x_values, radiation_at_100, 'r--', label=r'Radiation at $\tau$ = 100 (Analytical)')
# plt.plot(x_values, material_at_100, 'b--', label=r'Material at $\tau$ = 100 (Analytical)')


# Open the output file
fname1 = open("SuOlson1997-RN-run1.out", "rb")
fname2 = open("SuOlson1997-run1.out", "rb")

# Times corresponding to data in the output file
times = [r"$\tau$ = 0.1", r"$\tau$ = 1.0", r"$\tau$ = 10"]

# Loop to read and plot data from the output file
colors = ['red', 'blue', 'green']  # List of colors for each iteration

for i in range(3):
    time_line = fname1.readline().decode().strip()  # Skip the time line and decode it properly
    time_line2 = fname2.readline().decode().strip()

    xdata1 = pickle.load(fname1)      # cellpos
    xdata2 = pickle.load(fname2)
    # temp = pickle.load(fname)       # material temp
    # radtemp = pickle.load(fname)    # radiation temp
    matnrgdens1 = pickle.load(fname1) # material energy density
    radnrgdens1 = pickle.load(fname1) # radiation energy density

    matnrgdens2 = pickle.load(fname2)
    radnrgdens2 = pickle.load(fname2)

    # Plot the file data (solid lines)
    plt.plot(xdata1, matnrgdens1, color=colors[i], marker='o', label=f"[RN] Material at {times[i]}", linewidth=2, alpha=.5)
    plt.plot(xdata2, matnrgdens2, color=colors[i], marker='x', label=f"[NRN] Material at {times[i]}", linewidth=2, alpha=.5)
    # plt.plot(xdata1, radnrgdens1, color=colors[i], marker='o', label=f"[RN] Radiation at {times[i]}", linewidth=2, alpha=.5)
    # plt.plot(xdata2, radnrgdens2, color=colors[i], marker='x', label=f"[NRN] Radiation at {times[i]}", linewidth=2, alpha=.5)


# Now plot benchmark data 
# plt.plot(x_bench, rad_benchone[0], 'rx--', label=r'Radiation at $\tau$ = 0.1 (Benchmark)', linewidth=2)  
# plt.plot(x_bench, rad_benchone[2], 'bx--', label=r'Radiation at $\tau$ = 1.0 (Benchmark)', linewidth=2)
# plt.plot(x_bench, rad_benchone[4], 'gx--', label=r'Radiation at $\tau$ = 10 (Benchmark)', linewidth=2)

# plt.plot(x_bench, mat_benchone[0], 'rx-', label=r'Material at $\tau$ = 0.1 (Benchmark)', linewidth=2)  
# plt.plot(x_bench, mat_benchone[2], 'bx-.', label=r'Material at $\tau$ = 1.0 (Benchmark)', linewidth=2)  
# plt.plot(x_bench, mat_benchone[4], 'gx-.', label=r'Material at $\tau$ = 10 (Benchmark)', linewidth=2)  
# Set axis labels and limits
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-8, 3.0)
plt.xlim(0.1, 4.0)
plt.xlabel("x - cm")
plt.ylabel("Energy Density")

# Add legend
plt.legend(loc='lower left', numpoints=1, frameon=False)
# plt.legend(loc='upper right', numpoints=1, frameon=False)


# Save figure
plt.savefig("SuOlson1997-comparison-run1-mat.png", bbox_inches="tight", dpi=900)
# plt.savefig("SuOlson1997-comparison-run1-rad.png", bbox_inches="tight", dpi=900)

# Optionally show the plot
plt.show()

# Close the file
fname1.close()
fname2.close()
