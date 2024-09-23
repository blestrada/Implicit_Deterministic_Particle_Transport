"""Global particle data."""

# Particle Parameters
Nx = -1
Nmu = -1
Nt = -1
Nxi = -1

n_max = -1

# Particle Lists

# Global list of particles
particle_prop = []

# Global list of particles scattered from particle_prop
scattered_particles = []

# Global list of particles scattered from scattered_particles
particles_to_be_scattered = []


# Census grid
census_grid = []

# Some parameters
# mode : random numbers or no random numbers (rn or nrn)
# scattering: using analog scattering or implicit scattering - used only in NRN mode (analog or implicit)
mode = None
scattering = None