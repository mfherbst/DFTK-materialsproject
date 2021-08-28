using DFTK
using Unitful
using UnitfulAtomic
using JLD2
using PyCall
using MPI

if mpi_nprocs() == 1
    setup_threading()
else
    disable_threading()
end

# This tries to replicate the computational setup used for generating
# the materials project data as closely as possible. For details on
# the parameter choices see
#
#    https://docs.materialsproject.org/methodology/total-energies/
#
#
# All functions are MPI-ready, i.e. can be used through an MPI-parallelised julia session.

#
# If convergence issues on a system, try one of them (in this order and separately)
#    (a) is_metal=false
#    (b) use_experimental=false
#    (c) use_experimental=false, lower the damping (to 0.6, 0.4 and below if needed.)
#

function compute_formation_energy_per_atom(file::String; kwargs_bulk=(; ),
                                           kwargs_compound=(; ), kwargs...)
    scfres = run_if_needed(file; kwargs..., kwargs_compound...)

    bulk_energy_per_atom = Dict(map(scfres.basis.model.atoms) do (element, _)
        scfres_bulk = run_if_needed(element.symbol; kwargs..., kwargs_bulk...)
        n_atoms_bulk = sum(el_pos -> length(el_pos[2]), scfres_bulk.basis.model.atoms)
        element => scfres_bulk.energies.total / n_atoms_bulk
    end)

    atoms   = scfres.basis.model.atoms
    n_atoms = sum(el_pos -> length(el_pos[2]), atoms)
    energy_bulk = sum(el_pos -> bulk_energy_per_atom[el_pos[1]] * length(el_pos[2]), atoms)
    formation_energy_hartree = (scfres.energies.total - energy_bulk) / n_atoms
    auconvert(u"eV", formation_energy_hartree)
end


function run_if_needed(file::String; kwargs...)
    outfile = joinpath(@__DIR__, file * ".jld2")
    if !isfile(outfile)
        mpi_master() && println("#\n# -- $file\n#")
        scfres = run_from_file(file; kwargs...)
        store_results(scfres, outfile)
        scfres
    else
        load_results(outfile)
    end
end
function run_if_needed(element::Symbol; kwargs...)
    mkpath(joinpath(@__DIR__, "reference"))
    outfile = joinpath(@__DIR__, "reference", "$element.jld2")
    if !isfile(outfile)
        mpi_master() && println("#\n# -- $element\n#")
        scfres = run_bulk(element; kwargs...)
        store_results(scfres, outfile)
        scfres
    else
        load_results(outfile)
    end
end


function run_bulk(element::Symbol; kwargs...)
    bulkstructure = pyimport("ase.build").bulk(string(element))
    lattice = load_lattice(bulkstructure)
    atoms   = load_atoms(bulkstructure)
    run_dftk(lattice, atoms; kwargs...)
end


function run_from_file(file::AbstractString; kwargs...)
    lattice = load_lattice(file)
    atoms   = load_atoms(file)
    run_dftk(lattice, atoms; kwargs...)
end


# Store the results from `run_from_file`
function store_results(scfres, file)
    basis_master = DFTK.gather_kpts(scfres.basis)
    if mpi_master()
        JLD2.jldopen(file, "w") do jld
            jld["basis"]    = basis_master
            jld["ρ"]        = scfres.ρ
            jld["energies"] = scfres.energies
        end
    end
end


function load_results(file)
    JLD2.jldopen(file, "r") do jld
        (energies=jld["energies"], basis=jld["basis"], ρ=jld["ρ"])
    end
end


function run_dftk(lattice, atoms; use_experimental=true, is_metal=true, damping=0.8, debug=false, kwargs...)
    n_atoms = sum(el_pos -> length(el_pos[2]), atoms)

    # Check there are no atoms in the atoms list that require GGA+U
    has_ggau_metal = any(atoms) do (element, positions)
        element.symbol in (:V, :Cr, :Mn, :Fe, :Co, :Ni, :W, :Mo)
    end
    has_oxygen   = any(el_pos -> el_pos[1].symbol == :O, atoms)
    has_flouride = any(el_pos -> el_pos[1].symbol == :F, atoms)
    if (has_oxygen || has_flouride) && has_ggau_metal
        error("This compound requires GGA+U which is not implemented in DFTK.")
    end

    # Pseudos: Select the ones with largest number of electrons
    atoms = map(atoms) do (element, positions)
        symbol = element.symbol
        pspid = sort(list_psp(symbol, functional="pbe"), by=t->t.n_elec_valence)[end]
        ElementPsp(symbol, psp=load_psp(pspid.identifier)) => positions
    end

    magnetic_moments = map(atoms) do (element, positions)
        element => fill(default_magnetic_moment(element.symbol), length(positions))
    end
    # https://docs.materialsproject.org/methodology/total-energies/#calculation-details

    # They used the Tetrahedron method + Blöchl corrections ... which we don't have in DFTK.
    smearing = Smearing.Gaussian()
    temperature = 0.2u"eV"
    model = model_PBE(lattice, atoms; temperature, magnetic_moments, smearing)

    if debug
        kgrid = kgrid_from_minimal_n_kpoints(lattice, 6)
        Ecut = 10
    else
        kgrid = kgrid_from_minimal_n_kpoints(lattice, min(500, ceil(Int, 1000 / n_atoms)))
        Ecut  = 520u"eV"
    end
    basis = PlaneWaveBasis(model; Ecut, kgrid)

    ρ       = guess_density(basis, magnetic_moments)
    tol     = austrip(n_atoms * 5e-5u"eV")
    n_bands = DFTK.default_n_bands(basis.model)

    if use_experimental
        mixing    = is_metal ? KerkerMixing() : SimpleMixing()
        damping   = :adaptive
        algorithm = ("adaptive damping", DFTK.scf_potential_mixing_adaptive)
        extra     = (; )
    else
        mixing  = LdosMixing()
        extra = (damping=damping, )
        algorithm = ("fixed damping", self_consistent_field)
    end

    if mpi_master()
        magmom_compact = [element.symbol => values for (element, values) in magnetic_moments]
        display(basis)
        println()
        println()
        println("Solver parameters:")
        println("    magmom     : $magmom_compact")
        println("    tol        : $tol")
        println("    n_bands    : $n_bands")
        println("    algorithm  : $(algorithm[1])")
        println("    mixing     : $mixing")
        println("    damping    : $damping")
        println("    kwargs     : $(Dict(kwargs...))")
        println()
        flush(stdout)
    end

    DFTK.reset_timer!(DFTK.timer)
    run_scf = algorithm[2]
    scfres = run_scf(basis; ρ, tol, mixing, n_bands, maxiter=200, extra..., kwargs...)
    if mpi_master()
        println()
        println(DFTK.timer)
        println()
    end
    scfres
end


function default_magnetic_moment(element::Symbol)
    # TODO Consider adding this to PeriodicTable
    #      ... or DFTK ... or both
    get(Dict(
        :H  => 1.0,
        :He => 0.0,
        :Li => 1.0,
        :Be => 0.0,
        :B  => 1.0,
        :C  => 2.0,
        :N  => 3.0,
        :O  => 2.0,
        :F  => 1.0,
        :Ne => 0.0,
        :Na => 1.0,
        :Mg => 0.0,
        :Al => 1.0,
        :Si => 2.0,
        :P  => 3.0,
        :S  => 2.0,
        :Cl => 1.0,
        :Ar => 0.0,
        :K  => 1.0,
        :Ca => 0.0,
        :Sc => 1.0,
        :Ti => 2.0,
        :V  => 3.0,
        :Cr => 6.0,
        :Mn => 5.0,
        :Fe => 4.0,
        :Co => 3.0,
        :Ni => 2.0,
        :Cu => 1.0,
        :Zn => 0.0,
        :Ga => 1.0,
        :Ge => 2.0,
        :As => 3.0,
        :Se => 2.0,
        :Br => 1.0,
        :Kr => 0.0,
        :Rb => 1.0,
        :Sr => 0.0,
        :Y  => 1.0,
        :Zr => 2.0,
        :Nb => 5.0,
        :Mo => 6.0,
        :Tc => 5.0,
        :Ru => 4.0,
        :Rh => 3.0,
        :Pd => 0.0,
        :Ag => 1.0,
        :Cd => 0.0,
        :In => 1.0,
        :Sn => 2.0,
        :Sb => 3.0,
        :Te => 2.0,
        :I  => 1.0,
        :Xe => 0.0,
        :Cs => 1.0,
        :Ba => 0.0,
        :La => 1.0,
        :Ce => 1.0,
        :Pr => 3.0,
        :Nd => 4.0,
        :Pm => 5.0,
        :Sm => 6.0,
        :Eu => 7.0,
        :Gd => 8.0,
        :Tb => 5.0,
        :Dy => 4.0,
        :Ho => 3.0,
        :Er => 2.0,
        :Tm => 1.0,
        :Yb => 0.0,
        :Lu => 1.0,
        :Hf => 2.0,
        :Ta => 3.0,
        :W  => 4.0,
        :Re => 5.0,
        :Os => 4.0,
        :Ir => 3.0,
        :Pt => 2.0,
        :Au => 1.0,
        :Hg => 0.0,
        :Tl => 1.0,
        :Pb => 2.0,
        :Bi => 3.0,
        :Po => 2.0,
        :At => 1.0,
        :Rn => 0.0,
        :Fr => 1.0,
        :Ra => 0.0,
        :Ac => 1.0,
        :Th => 2.0,
        :Pa => 3.0,
        :U  => 4.0,
        :Np => 5.0,
        :Pu => 6.0,
        :Am => 7.0,
        :Cm => 8.0,
        :Bk => 5.0,
        :Cf => 4.0,
        :Es => 4.0,
        :Fm => 2.0,
        :Md => 1.0,
        :No => 0.0,
    ), element, 0)
end
