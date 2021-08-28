include("common.jl")

# Just run one example file
file = "./examples/mp-21202.cif"
formation_energy = compute_formation_energy_per_atom(file)
mpi_master() && @show (file, formation_energy)
