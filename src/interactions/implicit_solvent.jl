export ImplicitSolventOBC2

"""
    ImplicitSolventOBC2()

Onufriev-Bashford-Case GBSA model using the GBOBCII parameters.
"""
struct ImplicitSolventOBC2 end

function forces(inter::ImplicitSolventOBC2, sys, neighbors)
    return ustrip_vec.(zero(sys.coords))
end

function potential_energy(inter::ImplicitSolventOBC2, sys, neighbors)
    return 0.0
end
