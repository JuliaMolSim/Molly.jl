# See https://udel.edu/~arthij/MD.pdf for information on forces
# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See Gromacs manual for other aspects of forces

export
    LennardJones,
    Coulomb,
    Bond,
    Angle,
    Dihedral,
    update_accelerations!

"Square of the non-bonded interaction distance cutoff in nm^2."
const sqdist_cutoff_nb = 1.0 ^ 2

"The constant for Coulomb interaction, 1/(4*π*ϵ0*ϵr)."
const coulomb_const = 138.935458 / 70.0 # Treat ϵr as 70 for now

"The Lennard-Jones 6-12 interaction."
struct LennardJones <: GeneralInteraction
    nl_only::Bool
end

LennardJones() = LennardJones(false)

"The Coulomb electrostatic interaction."
struct Coulomb <: GeneralInteraction
    nl_only::Bool
end

"A bond between two atoms."
struct Bond <: SpecificInteraction
    i::Int
    j::Int
    b0::Float64
    kb::Float64
end

"A bond angle between three atoms."
struct Angle <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    th0::Float64
    cth::Float64
end

"A dihedral torsion angle between four atoms."
struct Dihedral <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    l::Int
    f1::Float64
    f2::Float64
    f3::Float64
    f4::Float64
end

"Update the accelerations in response to a given interation type."
function update_accelerations! end

@fastmath @inbounds function update_accelerations!(accels,
                                            inter::LennardJones,
                                            s::Simulation,
                                            i::Integer,
                                            j::Integer)
    if s.atoms[i].σ == 0.0 || s.atoms[j].σ == 0.0
        return
    end
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    if r2 > sqdist_cutoff_nb
        return
    end
    invr2 = inv(r2)
    six_term = (σ ^ 2 * invr2) ^ 3
    # Limit this to 100 as a fudge to stop it exploding
    f = min((24ϵ * invr2) * (2 * six_term ^ 2 - six_term), 100.0)
    fdr = f * dr
    accels[i] -= fdr
    accels[j] += fdr
end

@fastmath @inbounds function update_accelerations!(accels,
                                            inter::Coulomb,
                                            s::Simulation,
                                            i::Integer,
                                            j::Integer)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    if r2 > sqdist_cutoff_nb
        return
    end
    f = (coulomb_const * s.atoms[i].charge * s.atoms[j].charge) / sqrt(r2 ^ 3)
    fdr = f * dr
    accels[i] -= fdr
    accels[j] += fdr
end

function update_accelerations!(accels,
                                b::Bond,
                                s::Simulation)
    ab = vector(s.coords[b.i], s.coords[b.j], s.box_size)
    c = b.kb * (norm(ab) - b.b0)
    f = c * normalize(ab)
    accels[b.i] += f
    accels[b.j] -= f
end

# Sometimes domain error occurs for acos if the float is > 1.0 or < 1.0
acosbound(x::Real) = acos(max(min(x, 1.0), -1.0))

function update_accelerations!(accels,
                                a::Angle,
                                s::Simulation)
    ba = vector(s.coords[a.j], s.coords[a.i], s.box_size)
    bc = vector(s.coords[a.j], s.coords[a.k], s.box_size)
    pa = normalize(ba × (ba × bc))
    pc = normalize(-bc × (ba × bc))
    angle_term = -a.cth * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    accels[a.i] += fa
    accels[a.j] += fb
    accels[a.k] += fc
end

function update_accelerations!(accels,
                                d::Dihedral,
                                s::Simulation)
    ba = vector(s.coords[d.j], s.coords[d.i], s.box_size)
    bc = vector(s.coords[d.j], s.coords[d.k], s.box_size)
    dc = vector(s.coords[d.l], s.coords[d.k], s.box_size)
    p1 = normalize(ba × bc)
    p2 = normalize(-dc × -bc)
    θ = atan(dot((-ba × bc) × (bc × -dc), normalize(bc)), dot(-ba × bc, bc × -dc))
    angle_term = 0.5*(d.f1*sin(θ) - 2*d.f2*sin(2*θ) + 3*d.f3*sin(3*θ))
    fa = (angle_term / (norm(ba) * sin(acosbound(dot(ba, bc) / (norm(ba) * norm(bc)))))) * p1
    # fd clashes with a function name
    f_d = (angle_term / (norm(dc) * sin(acosbound(dot(bc, dc) / (norm(bc) * norm(dc)))))) * p2
    oc = 0.5 * bc
    tc = -(oc × f_d + 0.5 * (-dc × f_d) + 0.5 * (ba × fa))
    fc = (1 / dot(oc, oc)) * (tc × oc)
    fb = -fa - fc -f_d
    accels[d.i] += fa
    accels[d.j] += fb
    accels[d.k] += fc
    accels[d.l] += f_d
end
