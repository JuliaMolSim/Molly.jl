# Implicit solvent models
# Based on the OpenMM source code

export
    ImplicitSolventOBC,
    ImplicitSolventGBN2

# Generalized Born (GB) implicit solvent models augmented with the
#   hydrophobic solvent accessible surface area (SA) term
# Custom GBSA methods should sub-type this abstract type
abstract type AbstractGBSA end

# Default solvent dielectric is 78.5 for consistency with AMBER
# Elsewhere it is 78.3
const gb_solvent_dielectric = 78.5
const gb_solute_dielectric = 1.0

const obc_offset = 0.009u"nm"
const gbn2_offset = 0.0195141u"nm"

const gb_probe_radius = 0.14u"nm"
const gb_sa_factor = 28.3919551u"kJ * mol^-1 * nm^-2"

const gbn2_neck_scale = 0.826836
const gbn2_neck_cut = 0.68u"nm"

const mbondi2_element_to_radius = Dict(
    "N"     => 0.155u"nm",
    "O"     => 0.15u"nm" ,
    "F"     => 0.15u"nm" ,
    "Si"    => 0.21u"nm" ,
    "P"     => 0.185u"nm",
    "S"     => 0.18u"nm" ,
    "Cl"    => 0.17u"nm" ,
    "C"     => 0.17u"nm" ,
    "H"     => 0.12u"nm" ,
    "H_N"   => 0.13u"nm" ,
    "H_ARG" => 0.117u"nm",
    "O_CAR" => 0.14u"nm" ,
    "-"     => 0.15u"nm" ,
)

const obc_element_to_screen = Dict(
    "H" => 0.85,
    "C" => 0.72,
    "N" => 0.79,
    "O" => 0.85,
    "F" => 0.88,
    "P" => 0.86,
    "S" => 0.96,
    "-" => 0.80,
)

const gbn2_element_to_screen = Dict(
    "H" => 1.425952,
    "C" => 1.058554,
    "N" => 0.733599,
    "O" => 1.061039,
    "F" => 0.5,
    "P" => 0.5,
    "S" => -0.703469,
    "-" => 0.5,
)

const gbn2_element_to_screen_nucleic = Dict(
    "H" => 1.696538,
    "C" => 1.268902,
    "N" => 1.4259728,
    "O" => 0.1840098,
    "F" => 0.5,
    "P" => 1.5450597,
    "S" => 0.05,
    "-" => 0.5,
)

const gbn2_atom_params = Dict(
    "H_α" => 0.788440, "H_β" => 0.798699, "H_γ" => 0.437334,
    "D_α" => 0.788440, "D_β" => 0.798699, "D_γ" => 0.437334,
    "C_α" => 0.733756, "C_β" => 0.506378, "C_γ" => 0.205844,
    "N_α" => 0.503364, "N_β" => 0.316828, "N_γ" => 0.192915,
    "O_α" => 0.867814, "O_β" => 0.876635, "O_γ" => 0.387882,
    "S_α" => 0.867814, "S_β" => 0.876635, "S_γ" => 0.387882,
    "-_α" => 1.0     , "-_β" => 0.8     , "-_γ" => 4.851   ,
)

const gbn2_atom_params_nucleic = Dict(
    "H_α" => 0.537050, "H_β" => 0.362861, "H_γ" => 0.116704 ,
    "D_α" => 0.537050, "D_β" => 0.362861, "D_γ" => 0.116704 ,
    "C_α" => 0.331670, "C_β" => 0.196842, "C_γ" => 0.093422 ,
    "N_α" => 0.686311, "N_β" => 0.463189, "N_γ" => 0.138722 ,
    "O_α" => 0.606344, "O_β" => 0.463006, "O_γ" => 0.142262 ,
    "S_α" => 0.606344, "S_β" => 0.463006, "S_γ" => 0.142262 ,
    "P_α" => 0.418365, "P_β" => 0.290054, "P_γ" => 0.1064245,
    "-_α" => 1.0     , "-_β" => 0.8     , "-_γ" => 4.851    ,
)

const gbn2_data_d0 = [
    2.26685, 2.32548, 2.38397, 2.44235, 2.50057, 2.55867, 2.61663, 2.67444,
    2.73212, 2.78965, 2.84705, 2.9043, 2.96141, 3.0184, 3.07524, 3.13196,
    3.18854, 3.24498, 3.30132, 3.35752, 3.4136,
    2.31191, 2.37017, 2.4283, 2.48632, 2.5442, 2.60197, 2.65961, 2.71711,
    2.77449, 2.83175, 2.88887, 2.94586, 3.00273, 3.05948, 3.1161, 3.1726,
    3.22897, 3.28522, 3.34136, 3.39738, 3.45072,
    2.35759, 2.41549, 2.47329, 2.53097, 2.58854, 2.646, 2.70333, 2.76056,
    2.81766, 2.87465, 2.93152, 2.98827, 3.0449, 3.10142, 3.15782, 3.21411,
    3.27028, 3.32634, 3.3823, 3.43813, 3.49387,
    2.4038, 2.46138, 2.51885, 2.57623, 2.63351, 2.69067, 2.74773, 2.80469,
    2.86152, 2.91826, 2.97489, 3.0314, 3.08781, 3.1441, 3.20031, 3.25638,
    3.31237, 3.36825, 3.42402, 3.4797, 3.53527,
    2.45045, 2.50773, 2.56492, 2.62201, 2.679, 2.7359, 2.7927, 2.8494, 2.90599,
    2.9625, 3.0189, 3.07518, 3.13138, 3.18748, 3.24347, 3.29937, 3.35515,
    3.41085, 3.46646, 3.52196, 3.57738,
    2.4975, 2.5545, 2.61143, 2.66825, 2.72499, 2.78163, 2.83818, 2.89464,
    2.95101, 3.00729, 3.06346, 3.11954, 3.17554, 3.23143, 3.28723, 3.34294,
    3.39856, 3.45409, 3.50952, 3.56488, 3.62014,
    2.54489, 2.60164, 2.6583, 2.71488, 2.77134, 2.8278, 2.88412, 2.94034,
    2.9965, 3.05256, 3.10853, 3.16442, 3.22021, 3.27592, 3.33154, 3.38707,
    3.44253, 3.49789, 3.55316, 3.60836, 3.66348,
    2.59259, 2.6491, 2.70553, 2.76188, 2.81815, 2.87434, 2.93044, 2.98646,
    3.04241, 3.09827, 3.15404, 3.20974, 3.26536, 3.32089, 3.37633, 3.4317,
    3.48699, 3.54219, 3.59731, 3.65237, 3.70734,
    2.64054, 2.69684, 2.75305, 2.80918, 2.86523, 2.92122, 2.97712, 3.03295,
    3.0887, 3.14437, 3.19996, 3.25548, 3.31091, 3.36627, 3.42156, 3.47677,
    3.5319, 3.58695, 3.64193, 3.69684, 3.75167,
    2.68873, 2.74482, 2.80083, 2.85676, 2.91262, 2.96841, 3.02412, 3.07976,
    3.13533, 3.19082, 3.24623, 3.30157, 3.35685, 3.41205, 3.46718, 3.52223,
    3.57721, 3.63213, 3.68696, 3.74174, 3.79644,
    2.73713, 2.79302, 2.84884, 2.90459, 2.96027, 3.01587, 3.0714, 3.12686,
    3.18225, 3.23757, 3.29282, 3.34801, 3.40313, 3.45815, 3.51315, 3.56805,
    3.6229, 3.67767, 3.73237, 3.78701, 3.84159,
    2.78572, 2.84143, 2.89707, 2.95264, 3.00813, 3.06356, 3.11892, 3.17422,
    3.22946, 3.28462, 3.33971, 3.39474, 3.44971, 3.5046, 3.55944, 3.61421,
    3.66891, 3.72356, 3.77814, 3.83264, 3.8871,
    2.83446, 2.89, 2.94547, 3.00088, 3.05621, 3.11147, 3.16669, 3.22183,
    3.27689, 3.33191, 3.38685, 3.44174, 3.49656, 3.55132, 3.60602, 3.66066,
    3.71523, 3.76975, 3.82421, 3.8786, 3.93293,
    2.88335, 2.93873, 2.99404, 3.04929, 3.10447, 3.15959, 3.21464, 3.26963,
    3.32456, 3.37943, 3.43424, 3.48898, 3.54366, 3.5983, 3.65287, 3.70737,
    3.76183, 3.81622, 3.87056, 3.92484, 3.97905,
    2.93234, 2.9876, 3.04277, 3.09786, 3.15291, 3.20787, 3.26278, 3.31764,
    3.37242, 3.42716, 3.48184, 3.53662, 3.591, 3.64551, 3.69995, 3.75435,
    3.80867, 3.86295, 3.91718, 3.97134, 4.02545,
    2.98151, 3.0366, 3.09163, 3.14659, 3.20149, 3.25632, 3.3111, 3.36581,
    3.42047, 3.47507, 3.52963, 3.58411, 3.63855, 3.69293, 3.74725, 3.80153,
    3.85575, 3.90991, 3.96403, 4.01809, 4.07211,
    3.03074, 3.08571, 3.14061, 3.19543, 3.25021, 3.30491, 3.35956, 3.41415,
    3.46869, 3.52317, 3.57759, 3.63196, 3.68628, 3.74054, 3.79476, 3.84893,
    3.90303, 3.95709, 4.01111, 4.06506, 4.11897,
    3.08008, 3.13492, 3.1897, 3.2444, 3.29905, 3.35363, 3.40815, 3.46263,
    3.51704, 3.57141, 3.62572, 3.67998, 3.73418, 3.78834, 3.84244, 3.8965,
    3.95051, 4.00447, 4.05837, 4.11224, 4.16605,
    3.12949, 3.18422, 3.23888, 3.29347, 3.348, 3.40247, 3.45688, 3.51124,
    3.56554, 3.6198, 3.674, 3.72815, 3.78225, 3.83629, 3.8903, 3.94425,
    3.99816, 4.05203, 4.10583, 4.15961, 4.21333,
    3.17899, 3.23361, 3.28815, 3.34264, 3.39706, 3.45142, 3.50571, 3.55997,
    3.61416, 3.66831, 3.72241, 3.77645, 3.83046, 3.8844, 3.93831, 3.99216,
    4.04598, 4.09974, 4.15347, 4.20715, 4.26078,
    3.22855, 3.28307, 3.33751, 3.39188, 3.4462, 3.50046, 3.55466, 3.6088,
    3.6629, 3.71694, 3.77095, 3.82489, 3.8788, 3.93265, 3.98646, 4.04022,
    4.09395, 4.14762, 4.20126, 4.25485, 4.3084,
]u"nm" ./ 10

const gbn2_data_m0 = [
    0.0381511, 0.0338587, 0.0301776, 0.027003, 0.0242506, 0.0218529,
    0.0197547, 0.0179109, 0.0162844, 0.0148442, 0.0135647, 0.0124243,
    0.0114047, 0.0104906, 0.00966876, 0.008928, 0.0082587, 0.00765255,
    0.00710237, 0.00660196, 0.00614589,
    0.0396198, 0.0351837, 0.0313767, 0.0280911, 0.0252409, 0.0227563,
    0.0205808, 0.0186681, 0.0169799, 0.0154843, 0.014155, 0.0129696,
    0.0119094, 0.0109584, 0.0101031, 0.00933189, 0.0086348, 0.00800326,
    0.00742986, 0.00690814, 0.00643255,
    0.041048, 0.0364738, 0.0325456, 0.0291532, 0.0262084, 0.0236399,
    0.0213897, 0.0194102, 0.0176622, 0.0161129, 0.0147351, 0.0135059,
    0.0124061, 0.0114192, 0.0105312, 0.00973027, 0.00900602, 0.00834965,
    0.0077535, 0.00721091, 0.00671609,
    0.0424365, 0.0377295, 0.0336846, 0.0301893, 0.0271533, 0.0245038,
    0.0221813, 0.0201371, 0.018331, 0.0167295, 0.0153047, 0.014033,
    0.0128946, 0.0118727, 0.0109529, 0.0101229, 0.00937212, 0.00869147,
    0.00807306, 0.00751003, 0.00699641,
    0.0437861, 0.0389516, 0.0347944, 0.0311998, 0.0280758, 0.0253479,
    0.0229555, 0.0208487, 0.0189864, 0.0173343, 0.0158637, 0.0145507,
    0.0133748, 0.0123188, 0.0113679, 0.0105096, 0.0097329, 0.00902853,
    0.00838835, 0.00780533, 0.0072733,
    0.0450979, 0.0401406, 0.0358753, 0.0321851, 0.0289761, 0.0261726,
    0.0237125, 0.0215451, 0.0196282, 0.017927, 0.0164121, 0.0150588,
    0.0138465, 0.0127573, 0.0117761, 0.0108902, 0.0100882, 0.00936068,
    0.00869923, 0.00809665, 0.00754661,
    0.0463729, 0.0412976, 0.0369281, 0.0331456, 0.0298547, 0.026978,
    0.0244525, 0.0222264, 0.0202567, 0.0185078, 0.0169498, 0.0155575,
    0.0143096, 0.0131881, 0.0121775, 0.0112646, 0.010438, 0.00968781,
    0.00900559, 0.00838388, 0.00781622,
    0.0476123, 0.0424233, 0.0379534, 0.034082, 0.0307118, 0.0277645,
    0.0251757, 0.0228927, 0.0208718, 0.0190767, 0.0174768, 0.0160466,
    0.0147642, 0.0136112, 0.0125719, 0.0116328, 0.0107821, 0.0100099,
    0.00930735, 0.00866695, 0.00808206,
    0.0488171, 0.0435186, 0.038952, 0.0349947, 0.0315481, 0.0285324,
    0.0258824, 0.0235443, 0.0214738, 0.0196339, 0.0179934, 0.0165262,
    0.0152103, 0.0140267, 0.0129595, 0.0119947, 0.0111206, 0.0103268,
    0.00960445, 0.00894579, 0.00834405,
    0.0499883, 0.0445845, 0.0399246, 0.0358844, 0.032364, 0.0292822,
    0.0265729, 0.0241815, 0.0220629, 0.0201794, 0.0184994, 0.0169964,
    0.0156479, 0.0144345, 0.0133401, 0.0123504, 0.0114534, 0.0106386,
    0.00989687, 0.00922037, 0.00860216,
    0.0511272, 0.0456219, 0.040872, 0.0367518, 0.0331599, 0.0300142,
    0.0272475, 0.0248045, 0.0226392, 0.0207135, 0.0189952, 0.0174574,
    0.0160771, 0.0148348, 0.0137138, 0.0126998, 0.0117805, 0.0109452,
    0.0101846, 0.00949067, 0.00885636,
    0.0522348, 0.0466315, 0.0417948, 0.0375973, 0.0339365, 0.030729,
    0.0279067, 0.0254136, 0.023203, 0.0212363, 0.0194809, 0.0179092,
    0.016498, 0.0152275, 0.0140807, 0.013043, 0.012102, 0.0112466,
    0.0104676, 0.00975668, 0.00910664,
    0.0533123, 0.0476145, 0.042694, 0.0384218, 0.0346942, 0.0314268,
    0.0285507, 0.026009, 0.0237547, 0.0217482, 0.0199566, 0.018352,
    0.0169108, 0.0156128, 0.0144408, 0.0133801, 0.0124179, 0.011543,
    0.010746, 0.0100184, 0.00935302,
    0.0543606, 0.0485716, 0.04357, 0.0392257, 0.0354335, 0.0321082,
    0.02918, 0.0265913, 0.0242943, 0.0222492, 0.0204225, 0.0187859,
    0.0173155, 0.0159908, 0.0147943, 0.0137111, 0.0127282, 0.0118343,
    0.0110197, 0.0102759, 0.00959549,
    0.0553807, 0.0495037, 0.0444239, 0.0400097, 0.0361551, 0.0327736,
    0.0297949, 0.0271605, 0.0248222, 0.0227396, 0.0208788, 0.0192111,
    0.0177122, 0.0163615, 0.0151413, 0.0140361, 0.013033, 0.0121206,
    0.0112888, 0.0105292, 0.00983409,
    0.0563738, 0.0504116, 0.0452562, 0.0407745, 0.0368593, 0.0334235,
    0.0303958, 0.0277171, 0.0253387, 0.0232197, 0.0213257, 0.0196277,
    0.0181013, 0.0167252, 0.0154817, 0.0143552, 0.0133325, 0.0124019,
    0.0115534, 0.0107783, 0.0100688,
    0.0573406, 0.0512963, 0.0460676, 0.0415206, 0.0375468, 0.0340583,
    0.030983, 0.0282614, 0.0258441, 0.0236896, 0.0217634, 0.020036,
    0.0184826, 0.017082, 0.0158158, 0.0146685, 0.0136266, 0.0126783,
    0.0118135, 0.0110232, 0.0102998,
    0.0582822, 0.0521584, 0.0468589, 0.0422486, 0.038218, 0.0346784,
    0.0315571, 0.0287938, 0.0263386, 0.0241497, 0.0221922, 0.0204362,
    0.0188566, 0.0174319, 0.0161437, 0.0149761, 0.0139154, 0.0129499,
    0.0120691, 0.0112641, 0.0105269,
    0.0591994, 0.0529987, 0.0476307, 0.042959, 0.0388734, 0.0352843,
    0.0321182, 0.0293144, 0.0268225, 0.0246002, 0.0226121, 0.0208283,
    0.0192232, 0.0177751, 0.0164654, 0.015278, 0.0141991, 0.0132167,
    0.0123204, 0.0115009, 0.0107504,
    0.0600932, 0.053818, 0.0483836, 0.0436525, 0.0395136, 0.0358764,
    0.0326669, 0.0298237, 0.0272961, 0.0250413, 0.0230236, 0.0212126,
    0.0195826, 0.0181118, 0.0167811, 0.0155744, 0.0144778, 0.0134789,
    0.0125673, 0.0117338, 0.0109702,
    0.0609642, 0.0546169, 0.0491183, 0.0443295, 0.0401388, 0.036455,
    0.0332033, 0.030322, 0.0277596, 0.0254732, 0.0234266, 0.0215892,
    0.0199351, 0.018442, 0.0170909, 0.0158654, 0.0147514, 0.0137365,
    0.0128101, 0.0119627, 0.0111863,
]u"nm^-1" .* 10

# This is force field dependent
is_carboxylate_O(at_data) = at_data.atom_type == "O2"

function atoms_bonded_to_N(atoms_data, bonds)
    bonded_to_N = falses(length(atoms_data))
    for (i, j) in zip(from_device(bonds.is), from_device(bonds.js))
        if atoms_data[i].element == "N"
            bonded_to_N[j] = true
        end
        if atoms_data[j].element == "N"
            bonded_to_N[i] = true
        end
    end
    return bonded_to_N
end

function mbondi2_radii(atoms_data, bonds; use_mbondi3=false,
                        element_to_radius=mbondi2_element_to_radius)
    bonded_to_N = atoms_bonded_to_N(atoms_data, bonds)
    return map(atoms_data, bonded_to_N) do at_data, at_bonded_to_N
        if use_mbondi3 && at_data.res_name == "ARG" &&
                (startswith(at_data.atom_name, "HH") || startswith(at_data.atom_name, "HE"))
            radius = element_to_radius["H_ARG"]
        elseif use_mbondi3 && is_carboxylate_O(at_data)
            radius = element_to_radius["O_CAR"]
        elseif at_data.element in ("H", "D")
            radius = at_bonded_to_N ? element_to_radius["H_N"] : element_to_radius["H"]
        else
            radius = dict_get(element_to_radius, at_data.element, element_to_radius["-"])
        end
        return radius
    end
end

function mbondi3_radii(atoms_data, bonds; element_to_radius=mbondi2_element_to_radius)
    return mbondi2_radii(atoms_data, bonds; use_mbondi3=true, element_to_radius=element_to_radius)
end

# Assign each atom a class based on its radius, returning the class of each atom and
#   the radius of each class
function radius_classes(radii)
    class_radii = eltype(radii)[]
    classes = Vector{Int32}(undef, length(radii))
    for (i, radius) in enumerate(radii)
        class_i = 0
        for (ci, class_radius) in enumerate(class_radii)
            if class_radius == radius
                class_i = ci
                break
            end
        end
        if class_i == 0
            push!(class_radii, radius)
            class_i = length(class_radii)
        end
        classes[i] = class_i
    end
    return classes, class_radii
end

# The pairwise value for atoms i and j is table[classes[i], classes[j]]
function lookup_table(full_table::AbstractArray{T}, radii) where T
    n_atoms = length(radii)
    table_positions = [(r - 0.1u"nm") * 200 for r in radii]
    # These zero-based indexes are converted to one-based when looking up the full table
    index_1, index_2 = zeros(Int, n_atoms), zeros(Int, n_atoms)
    weight_1, weight_2 = zeros(n_atoms), zeros(n_atoms)
    for (i, p) in enumerate(table_positions)
        if p <= 0.0u"nm"
            weight_1[i] = 1.0
        elseif p >= 20.0u"nm"
            index_1[i] = 20
            weight_1[i] = 1.0
        else
            ps = ustrip(u"nm", p)
            index_1[i] = Int(floor(ps))
            index_2[i] = index_1[i] + 1
            weight_1[i] = index_2[i] - ps
            weight_2[i] = 1.0 - weight_1[i]
        end
    end
    table = zeros(T, n_atoms, n_atoms)
    for i in 1:n_atoms
        for j in 1:n_atoms
            table[j, i] = weight_1[i] * weight_1[j] * full_table[index_1[i] * 21 + index_1[j] + 1] +
                          weight_1[i] * weight_2[j] * full_table[index_1[i] * 21 + index_2[j] + 1] +
                          weight_2[i] * weight_1[j] * full_table[index_2[i] * 21 + index_1[j] + 1] +
                          weight_2[i] * weight_2[j] * full_table[index_2[i] * 21 + index_2[j] + 1]
        end
    end
    return table
end

function lookup_table(full_table::AbstractArray, radii::AbstractArray{<:AbstractFloat})
    return lookup_table(full_table, radii * u"nm")
end

gbsa_n_threads(inter::AbstractGBSA) = max(length(inter.buffer_force_chunks), 1)

# On GPU the buffers do not depend on the number of threads, which is not used
check_gbsa_n_threads(::AbstractArray, n_threads::Integer) = nothing

function check_gbsa_n_threads(force_chunks::AbstractVector, n_threads::Integer)
    if n_threads != length(force_chunks)
        throw(ArgumentError("the implicit solvent interaction was set up for " *
                "$(length(force_chunks)) thread(s) but forces or the potential energy " *
                "were requested with n_threads=$n_threads, the number of threads has to " *
                "match as the buffers are allocated during setup"))
    end
    return nothing
end

# Buffers used to avoid allocating memory on each force or energy call
# On the GPU the buffers are also used to accumulate the results of the kernels
# The per-thread buffers are sized from `n_threads`, so force and energy calls have to
#   use the same number of threads
function gbsa_buffers(::Type{AT}, ::Type{T}, n_atoms, offset_radii, sa_factor_used,
                      dist_cutoff_used, offset_used, factor_solute,
                      n_threads::Integer) where {AT, T}
    if n_threads < 1
        throw(ArgumentError("n_threads must be at least 1, found $n_threads"))
    end
    born_force_type = typeof(T(sa_factor_used * oneunit(dist_cutoff_used)))
    born_force_scaled_type = typeof(T(sa_factor_used * oneunit(dist_cutoff_used) *
                                      oneunit(offset_used)^2))
    energy_type = typeof(factor_solute / oneunit(offset_used))

    Bs = to_device(zero(offset_radii), AT)
    B_grads = to_device(zeros(T, n_atoms), AT)
    atom_charges = to_device(zeros(T, n_atoms), AT)
    born_forces = to_device(zeros(born_force_type, n_atoms), AT)
    born_forces_scaled = to_device(zeros(born_force_scaled_type, n_atoms), AT)
    if AT <: AbstractGPUArray
        born_forces_mod = to_device(zeros(T, n_atoms), AT)
        Is_nounits = to_device(zeros(T, n_atoms), AT)
        pes = to_device(zeros(T, n_atoms), AT)
        fs_mat = to_device(zeros(T, 3, n_atoms), AT)
        force_chunks = zeros(T, 0, 0, 0)
    else
        born_forces_mod = zeros(T, 0)
        Is_nounits = zeros(T, 0)
        fs_mat = zeros(T, 0, 0)
        pes = zeros(energy_type, n_threads)
        # The first 3 columns are the force components and the fourth column is the Born
        #   force, which has the same units as the force
        force_chunks = [zeros(T, n_atoms, 4) for _ in 1:n_threads]
    end

    return (Bs=Bs, B_grads=B_grads, atom_charges=atom_charges, born_forces=born_forces,
            born_forces_scaled=born_forces_scaled, born_forces_mod=born_forces_mod,
            Is_nounits=Is_nounits, pes=pes, fs_mat=fs_mat, force_chunks=force_chunks)
end

"""
    ImplicitSolventOBC(atoms, atoms_data, bonds)

Onufriev-Bashford-Case GBSA model implemented as an AtomsCalculators.jl calculator.

Should be used along with a Coulomb interaction.
The keyword argument `use_OBC2` determines whether to use parameter set
I (`false`, the default) or II (`true`).

Not currently compatible with virial calculation.
"""
struct ImplicitSolventOBC{T, D, VT, VD, K, S, F, BF, BS, PT, MT, FC} <: AbstractGBSA
    offset_radii::VD
    scaled_offset_radii::VD
    solvent_dielectric::T
    solute_dielectric::T
    kappa::K
    offset::D
    dist_cutoff::D
    use_ACE::Bool
    α::T
    β::T
    γ::T
    probe_radius::D
    sa_factor::S
    factor_solute::F
    factor_solvent::F
    buffer_Bs::VD
    buffer_B_grads::VT
    buffer_atom_charges::VT
    buffer_born_forces::BF
    buffer_born_forces_scaled::BS
    buffer_born_forces_mod::VT
    buffer_Is_nounits::VT
    buffer_pes::PT
    buffer_fs_mat::MT
    buffer_force_chunks::FC
end

function ImplicitSolventOBC(atoms::AbstractArray{Atom{TY, M, T, D, E, L}},
                            atoms_data,
                            bonds;
                            solvent_dielectric=gb_solvent_dielectric,
                            solute_dielectric=gb_solute_dielectric,
                            kappa=0.0u"nm^-1",
                            offset=obc_offset,
                            dist_cutoff=0.0u"nm",
                            probe_radius=gb_probe_radius,
                            sa_factor=gb_sa_factor,
                            use_ACE=true,
                            use_OBC2=false,
                            element_to_radius=mbondi2_element_to_radius,
                            element_to_screen=obc_element_to_screen,
                            n_threads::Integer=Threads.nthreads()) where {TY, M, T, D, E, L}
    units = dimension(D) == u"𝐋"
    radii = mbondi2_radii(atoms_data, bonds; element_to_radius=element_to_radius)

    # The radii can be given with or without units, for example parameter injection
    #   gives them without units, so strip them and add them back if required
    radii_nounits = ustrip.(radii)
    if units
        offset_radii = T.(radii_nounits .* unit(offset) .- offset)
    else
        offset_radii = T.(radii_nounits .- ustrip(offset))
    end
    scaled_offset_radii = map(atoms_data, offset_radii) do at_data, offset_radius
        screen = dict_get(element_to_screen, at_data.element, element_to_screen["-"])
        return T(screen) * offset_radius
    end

    if use_OBC2
        # GBOBCII parameters
        α, β, γ = T(1.0), T(0.8), T(4.85)
    else
        # GBOBCI parameters
        α, β, γ = T(0.8), T(0.0), T(2.909125)
    end

    n_atoms = length(atoms)
    coulomb_const_units = (units ? coulomb_const : ustrip(coulomb_const))
    if !iszero_value(solute_dielectric)
        factor_solute = -T(coulomb_const_units) / T(solute_dielectric)
    else
        factor_solute = zero(T(coulomb_const_units))
    end
    if !iszero_value(solvent_dielectric)
        factor_solvent = T(coulomb_const_units) / T(solvent_dielectric)
    else
        factor_solvent = zero(T(coulomb_const_units))
    end

    AT = array_type(atoms)
    or = to_device(offset_radii, AT)
    sor = to_device(scaled_offset_radii, AT)

    if units
        bufs = gbsa_buffers(AT, T, n_atoms, offset_radii, T(sa_factor), dist_cutoff,
                            offset, factor_solute, n_threads)
        return ImplicitSolventOBC{T, D, typeof(bufs.B_grads), typeof(or), typeof(T(kappa)),
                        typeof(T(sa_factor)), typeof(factor_solute), typeof(bufs.born_forces),
                        typeof(bufs.born_forces_scaled), typeof(bufs.pes), typeof(bufs.fs_mat),
                        typeof(bufs.force_chunks)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(kappa), offset,
                    dist_cutoff, use_ACE, α, β, γ, probe_radius, T(sa_factor),
                    factor_solute, factor_solvent, bufs.Bs, bufs.B_grads, bufs.atom_charges,
                    bufs.born_forces, bufs.born_forces_scaled, bufs.born_forces_mod,
                    bufs.Is_nounits, bufs.pes, bufs.fs_mat, bufs.force_chunks)
    else
        bufs = gbsa_buffers(AT, T, n_atoms, offset_radii, ustrip(sa_factor),
                            ustrip(dist_cutoff), ustrip(offset), factor_solute, n_threads)
        return ImplicitSolventOBC{T, T, typeof(bufs.B_grads), typeof(or), T, T, T,
                        typeof(bufs.born_forces), typeof(bufs.born_forces_scaled),
                        typeof(bufs.pes), typeof(bufs.fs_mat), typeof(bufs.force_chunks)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(ustrip(kappa)),
                    ustrip(offset), ustrip(dist_cutoff), use_ACE, α, β, γ, ustrip(probe_radius),
                    ustrip(sa_factor), factor_solute, factor_solvent, bufs.Bs, bufs.B_grads,
                    bufs.atom_charges, bufs.born_forces, bufs.born_forces_scaled,
                    bufs.born_forces_mod, bufs.Is_nounits, bufs.pes, bufs.fs_mat,
                    bufs.force_chunks)
    end
end

function gb_bond_index(sys)
    return findfirst(sil -> eltype(sil.inters) <: HarmonicBond, sys.specific_inter_lists)
end

function gb_element_dicts(key_prefix, params_dic, default_radii, default_screens)
    element_to_radius = Dict{String, Float64}()
    for k in keys(default_radii)
        element_to_radius[k] = dict_get(params_dic, key_prefix * "radius_" * k,
                                        ustrip(default_radii[k]))
    end
    element_to_screen = empty(default_screens)
    for k in keys(default_screens)
        element_to_screen[k] = dict_get(params_dic, key_prefix * "screen_" * k,
                                        default_screens[k])
    end
    return element_to_radius, element_to_screen
end

function inject_interaction(inter::ImplicitSolventOBC, params_dic, sys)
    key_prefix = "inter_OBC_"
    element_to_radius, element_to_screen = gb_element_dicts(key_prefix, params_dic,
                                    mbondi2_element_to_radius, obc_element_to_screen)

    ImplicitSolventOBC(
        sys.atoms,
        sys.atoms_data,
        sys.specific_inter_lists[gb_bond_index(sys)];
        solvent_dielectric=dict_get(params_dic, key_prefix * "solvent_dielectric", inter.solvent_dielectric),
        solute_dielectric=dict_get(params_dic, key_prefix * "solute_dielectric", inter.solute_dielectric),
        kappa=dict_get(params_dic, key_prefix * "kappa", ustrip(inter.kappa))u"nm^-1",
        offset=dict_get(params_dic, key_prefix * "offset", ustrip(inter.offset))u"nm",
        dist_cutoff=inter.dist_cutoff,
        probe_radius=dict_get(params_dic, key_prefix * "probe_radius", ustrip(inter.probe_radius))u"nm",
        sa_factor=dict_get(params_dic, key_prefix * "sa_factor", ustrip(inter.sa_factor))u"kJ * mol^-1 * nm^-2",
        use_ACE=inter.use_ACE,
        # α, β and γ define the OBC1/OBC2 variants and are not treated as parameters
        use_OBC2=(inter.β != zero(inter.β)),
        element_to_radius=element_to_radius,
        element_to_screen=element_to_screen,
        n_threads=gbsa_n_threads(inter),
    )
end

function extract_parameters!(params_dic, inter::ImplicitSolventOBC, ff)
    key_prefix = "inter_OBC_"
    params_dic[key_prefix * "solvent_dielectric"] = inter.solvent_dielectric
    params_dic[key_prefix * "solute_dielectric" ] = inter.solute_dielectric
    params_dic[key_prefix * "kappa"             ] = ustrip(inter.kappa)
    params_dic[key_prefix * "offset"            ] = ustrip(inter.offset)
    params_dic[key_prefix * "probe_radius"      ] = ustrip(inter.probe_radius)
    params_dic[key_prefix * "sa_factor"         ] = ustrip(inter.sa_factor)
    for (k, v) in mbondi2_element_to_radius
        params_dic[key_prefix * "radius_" * k] = ustrip(v)
    end
    for (k, v) in obc_element_to_screen
        params_dic[key_prefix * "screen_" * k] = v
    end
    return params_dic
end

"""
    ImplicitSolventGBN2(atoms, atoms_data, bonds)

GBn2 solvation model implemented as an AtomsCalculators.jl calculator.

Should be used along with a Coulomb interaction.

Not currently compatible with virial calculation.
"""
struct ImplicitSolventGBN2{T, D, VT, VD, K, S, F, TD, TM, VI, BF, BS, PT, MT, FC} <: AbstractGBSA
    offset_radii::VD
    scaled_offset_radii::VD
    solvent_dielectric::T
    solute_dielectric::T
    kappa::K
    offset::D
    dist_cutoff::D
    use_ACE::Bool
    αs::VT
    βs::VT
    γs::VT
    probe_radius::D
    sa_factor::S
    factor_solute::F
    factor_solvent::F
    d0s::TD
    m0s::TM
    radius_classes::VI
    neck_scale::T
    neck_cut::D
    buffer_Bs::VD
    buffer_B_grads::VT
    buffer_atom_charges::VT
    buffer_born_forces::BF
    buffer_born_forces_scaled::BS
    buffer_born_forces_mod::VT
    buffer_Is_nounits::VT
    buffer_pes::PT
    buffer_fs_mat::MT
    buffer_force_chunks::FC
end

function ImplicitSolventGBN2(atoms::AbstractArray{Atom{TY, M, T, D, E, L}},
                                atoms_data,
                                bonds;
                                solvent_dielectric=gb_solvent_dielectric,
                                solute_dielectric=gb_solute_dielectric,
                                kappa=0.0u"nm^-1",
                                offset=gbn2_offset,
                                dist_cutoff=0.0u"nm",
                                probe_radius=gb_probe_radius,
                                sa_factor=gb_sa_factor,
                                use_ACE=true,
                                neck_scale=gbn2_neck_scale,
                                neck_cut=gbn2_neck_cut,
                                element_to_radius=mbondi2_element_to_radius,
                                element_to_screen=gbn2_element_to_screen,
                                element_to_screen_nucleic=gbn2_element_to_screen_nucleic,
                                atom_params=gbn2_atom_params,
                                atom_params_nucleic=gbn2_atom_params_nucleic,
                                data_d0=gbn2_data_d0,
                                data_m0=gbn2_data_m0,
                                n_threads::Integer=Threads.nthreads()) where {TY, M, T, D, E, L}
    units = dimension(D) == u"𝐋"
    radii = mbondi3_radii(atoms_data, bonds; element_to_radius=element_to_radius)
    nucleic_acid_residues = ("A", "C", "G", "U", "DA", "DC", "DG", "DT")

    # The radii can be given with or without units, for example parameter injection
    #   gives them without units, so strip them and add them back if required
    radii_nounits = ustrip.(radii)
    if units
        offset_radii = T.(radii_nounits .* unit(offset) .- offset)
    else
        offset_radii = T.(radii_nounits .- ustrip(offset))
    end
    scaled_offset_radii = map(atoms_data, offset_radii) do at_data, offset_radius
        if at_data.res_name in nucleic_acid_residues
            screen = dict_get(element_to_screen_nucleic, at_data.element, element_to_screen_nucleic["-"])
        else
            screen = dict_get(element_to_screen, at_data.element, element_to_screen["-"])
        end
        return T(screen) * offset_radius
    end

    αs_cpu = map(atoms_data) do at_data
        if at_data.res_name in nucleic_acid_residues
            α = dict_get(atom_params_nucleic, at_data.element * "_α", atom_params_nucleic["-_α"])
        else
            α = dict_get(atom_params, at_data.element * "_α", atom_params["-_α"])
        end
        return T(α)
    end
    βs_cpu = map(atoms_data) do at_data
        if at_data.res_name in nucleic_acid_residues
            β = dict_get(atom_params_nucleic, at_data.element * "_β", atom_params_nucleic["-_β"])
        else
            β = dict_get(atom_params, at_data.element * "_β", atom_params["-_β"])
        end
        return T(β)
    end
    γs_cpu = map(atoms_data) do at_data
        if at_data.res_name in nucleic_acid_residues
            γ = dict_get(atom_params_nucleic, at_data.element * "_γ", atom_params_nucleic["-_γ"])
        else
            γ = dict_get(atom_params, at_data.element * "_γ", atom_params["-_γ"])
        end
        return T(γ)
    end

    n_atoms = length(atoms)

    classes, class_radii = radius_classes(radii)
    table_d0_units = T.(lookup_table(data_d0, class_radii))
    table_m0_units = T.(lookup_table(data_m0, class_radii))
    if units
        table_d0 = table_d0_units
        table_m0 = table_m0_units
    else
        table_d0 = ustrip.(table_d0_units)
        table_m0 = ustrip.(table_m0_units)
    end

    coulomb_const_units = (units ? coulomb_const : ustrip(coulomb_const))
    if !iszero_value(solute_dielectric)
        factor_solute = -T(coulomb_const_units) / T(solute_dielectric)
    else
        factor_solute = zero(T(coulomb_const_units))
    end
    if !iszero_value(solvent_dielectric)
        factor_solvent = T(coulomb_const_units) / T(solvent_dielectric)
    else
        factor_solvent = zero(T(coulomb_const_units))
    end

    AT = array_type(atoms)
    or = to_device(offset_radii, AT)
    sor = to_device(scaled_offset_radii, AT)
    d0s, m0s = to_device(table_d0, AT), to_device(table_m0, AT)
    rcs = to_device(classes, AT)
    αs, βs, γs = to_device(αs_cpu, AT), to_device(βs_cpu, AT), to_device(γs_cpu, AT)

    if units
        bufs = gbsa_buffers(AT, T, n_atoms, offset_radii, T(sa_factor), dist_cutoff,
                            offset, factor_solute, n_threads)
        return ImplicitSolventGBN2{T, D, typeof(bufs.B_grads), typeof(or), typeof(T(kappa)),
                        typeof(T(sa_factor)), typeof(factor_solute), typeof(d0s), typeof(m0s),
                        typeof(rcs), typeof(bufs.born_forces), typeof(bufs.born_forces_scaled),
                        typeof(bufs.pes), typeof(bufs.fs_mat), typeof(bufs.force_chunks)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(kappa), offset, dist_cutoff,
                    use_ACE, αs, βs, γs, probe_radius, T(sa_factor), factor_solute, factor_solvent,
                    d0s, m0s, rcs, neck_scale, neck_cut, bufs.Bs, bufs.B_grads, bufs.atom_charges,
                    bufs.born_forces, bufs.born_forces_scaled, bufs.born_forces_mod,
                    bufs.Is_nounits, bufs.pes, bufs.fs_mat, bufs.force_chunks)
    else
        bufs = gbsa_buffers(AT, T, n_atoms, offset_radii, ustrip(sa_factor),
                            ustrip(dist_cutoff), ustrip(offset), factor_solute, n_threads)
        return ImplicitSolventGBN2{T, T, typeof(bufs.B_grads), typeof(or), T, T, T, typeof(d0s),
                        typeof(m0s), typeof(rcs), typeof(bufs.born_forces),
                        typeof(bufs.born_forces_scaled), typeof(bufs.pes), typeof(bufs.fs_mat),
                        typeof(bufs.force_chunks)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(ustrip(kappa)), ustrip(offset),
                    ustrip(dist_cutoff), use_ACE, αs, βs, γs, ustrip(probe_radius), ustrip(sa_factor),
                    factor_solute, factor_solvent, d0s, m0s, rcs, neck_scale, ustrip(neck_cut),
                    bufs.Bs, bufs.B_grads, bufs.atom_charges, bufs.born_forces,
                    bufs.born_forces_scaled, bufs.born_forces_mod, bufs.Is_nounits, bufs.pes,
                    bufs.fs_mat, bufs.force_chunks)
    end
end

function inject_interaction(inter::ImplicitSolventGBN2, params_dic, sys)
    key_prefix = "inter_GB_"
    element_to_radius, element_to_screen = gb_element_dicts(key_prefix, params_dic,
                                    mbondi2_element_to_radius, gbn2_element_to_screen)
    atom_params = empty(gbn2_atom_params)
    for k in keys(gbn2_atom_params)
        atom_params[k] = dict_get(params_dic, key_prefix * "params_" * k, gbn2_atom_params[k])
    end

    ImplicitSolventGBN2(
        sys.atoms,
        sys.atoms_data,
        sys.specific_inter_lists[gb_bond_index(sys)];
        solvent_dielectric=dict_get(params_dic, key_prefix * "solvent_dielectric", inter.solvent_dielectric),
        solute_dielectric=dict_get(params_dic, key_prefix * "solute_dielectric", inter.solute_dielectric),
        kappa=dict_get(params_dic, key_prefix * "kappa", ustrip(inter.kappa))u"nm^-1",
        offset=dict_get(params_dic, key_prefix * "offset", ustrip(inter.offset))u"nm",
        dist_cutoff=inter.dist_cutoff,
        probe_radius=dict_get(params_dic, key_prefix * "probe_radius", ustrip(inter.probe_radius))u"nm",
        sa_factor=dict_get(params_dic, key_prefix * "sa_factor", ustrip(inter.sa_factor))u"kJ * mol^-1 * nm^-2",
        use_ACE=inter.use_ACE,
        neck_scale=dict_get(params_dic, key_prefix * "neck_scale", inter.neck_scale),
        neck_cut=dict_get(params_dic, key_prefix * "neck_cut", ustrip(inter.neck_cut))u"nm",
        element_to_radius=element_to_radius,
        element_to_screen=element_to_screen,
        atom_params=atom_params,
        n_threads=gbsa_n_threads(inter),
    )
end

function extract_parameters!(params_dic, inter::ImplicitSolventGBN2, ff)
    key_prefix = "inter_GB_"
    params_dic[key_prefix * "solvent_dielectric"] = inter.solvent_dielectric
    params_dic[key_prefix * "solute_dielectric" ] = inter.solute_dielectric
    params_dic[key_prefix * "kappa"             ] = ustrip(inter.kappa)
    params_dic[key_prefix * "offset"            ] = ustrip(inter.offset)
    params_dic[key_prefix * "probe_radius"      ] = ustrip(inter.probe_radius)
    params_dic[key_prefix * "sa_factor"         ] = ustrip(inter.sa_factor)
    params_dic[key_prefix * "neck_scale"        ] = inter.neck_scale
    params_dic[key_prefix * "neck_cut"          ] = ustrip(inter.neck_cut)
    for (k, v) in mbondi2_element_to_radius
        params_dic[key_prefix * "radius_" * k] = ustrip(v)
    end
    for (k, v) in gbn2_element_to_screen
        params_dic[key_prefix * "screen_" * k] = v
    end
    for (k, v) in gbn2_atom_params
        params_dic[key_prefix * "params_" * k] = v
    end
    return params_dic
end

#=
The functions below are shared between the CPU and the GPU implementations.
The per-pair functions are used by both, and the loops that sum the contribution to
atom `i` over a range of the other atoms are used by the GPU kernels and, where the
CPU does not benefit from only looking at each pair once, by the CPU loops too.
=#

# Parameters that are the same for every atom are stored as a scalar rather than an array
@inline atom_param(x::AbstractArray, i) = @inbounds x[i]
@inline atom_param(x, i) = x

@inline gb_length_unit(inter::AbstractGBSA) = unit(eltype(inter.offset_radii))

# The GBn2 neck integral is parameterised in Å, this converts a length in the unit used
#   by the model to Å. The float type has to be given explicitly, otherwise the whole neck
#   term is promoted to Float64 and is 32x slower on a consumer GPU.
gb_neck_dist_scale(::Type{T}, u::Unitful.Units) where {T} = T(ustrip(u"Å", 1.0 * u))
# Lengths without units are taken to be nm
gb_neck_dist_scale(::Type{T}, ::typeof(NoUnits)) where {T} = T(10)

@inline function gb_sqdist_cutoff(inter::AbstractGBSA)
    dist_cutoff = ustrip(gb_length_unit(inter), inter.dist_cutoff)
    return iszero(dist_cutoff) ? typemax(dist_cutoff) : dist_cutoff^2
end

# Data required for the GBn2 neck correction, nothing for models without a neck term
# Returned as a named tuple so that it can be passed to GPU kernels
gb_neck_data(::AbstractGBSA) = nothing

function gb_neck_data(inter::ImplicitSolventGBN2)
    lu = gb_length_unit(inter)
    return (scale=inter.neck_scale, cut=ustrip(lu, inter.neck_cut),
            offset=ustrip(lu, inter.offset), d0s=inter.d0s, m0s=inter.m0s,
            classes=inter.radius_classes, lu=lu,
            dist_scale=gb_neck_dist_scale(typeof(inter.neck_scale), lu))
end

# Fast math versions of the transcendental functions are used in Float32, where they map
#   to hardware instructions on the GPU; on the CPU they are the standard functions apart
#   from `exp`, which flushes subnormal results to zero
@inline gb_exp(x::Float32) = Base.FastMath.exp_fast(x)
@inline gb_exp(x) = exp(x)
@inline gb_inv(x::Float32) = Base.FastMath.inv_fast(x)
@inline gb_inv(x) = inv(x)
@inline gb_div(x::Float32, y::Float32) = Base.FastMath.div_fast(x, y)
@inline gb_div(x, y) = x / y

const gb_sqrt2 = 1.4142135623730951
const gb_ln2 = 0.6931471805599453
# atanh(s)/s as a series in s^2, enough terms for Float64 precision when the argument of
#   the log has been reduced to [sqrt(2)/2, sqrt(2)), giving |s| <= 0.1716
const gb_atanh_coeffs = (1.0, 1/3, 1/5, 1/7, 1/9, 1/11, 1/13, 1/15, 1/17, 1/19, 1/21)

#=
`log` is the most expensive part of the Born radii integral and of its derivative.
`Base.log` has branches and a table lookup, which stops the surrounding loops from
vectorising, so a branch free version is used in Float64. It is accurate to 1.6 ulp for
any positive normal number.

The exponent of the argument, and the power of two that reduces it to a mantissa, are
step functions of the argument and so contribute nothing to the derivative. They are
split into `gb_log_scaling`, which the Enzyme extension marks as inactive since Enzyme
cannot differentiate the bit manipulation. The mantissa is then recovered by a floating
point multiplication rather than by moving bits, so the derivative is still correct.
=#
@inline function gb_log_scaling(x::Float64)
    bits = reinterpret(UInt64, x)
    exponent_bits = bits >> 52
    # The mantissa in [1, 2), used to decide whether to centre the range on sqrt(2)
    mantissa = reinterpret(Float64, (bits & 0x000fffffffffffff) | 0x3ff0000000000000)
    inc = ifelse(mantissa > gb_sqrt2, UInt64(1), UInt64(0))
    # The exponent as a float, avoiding an integer to float conversion
    e = reinterpret(Float64, (exponent_bits + inc) | 0x4330000000000000) - 0x1p52 - 1023
    return e, reinterpret(Float64, (UInt64(2046) - exponent_bits - inc) << 52)
end

@inline function gb_log(x::Float64)
    e, scale = gb_log_scaling(x)
    m = x * scale # In [sqrt(2)/2, sqrt(2))
    s = (m - 1) / (m + 1)
    return e * gb_ln2 + 2 * s * evalpoly(s * s, gb_atanh_coeffs)
end

@inline gb_log(x::Float32) = Base.FastMath.log_fast(x)
@inline gb_log(x) = log(x)

# Per-atom α, β and γ parameters, scalars for OBC and arrays for GBn2
gb_αs(inter::ImplicitSolventOBC ) = inter.α
gb_βs(inter::ImplicitSolventOBC ) = inter.β
gb_γs(inter::ImplicitSolventOBC ) = inter.γ
gb_αs(inter::ImplicitSolventGBN2) = inter.αs
gb_βs(inter::ImplicitSolventGBN2) = inter.βs
gb_γs(inter::ImplicitSolventGBN2) = inter.γs

# HCT pairwise integral, common to the OBC and GBn2 models
@inline function born_integral(r, rinv, ori, srj)
    U = r + srj
    L = max(ori, abs(r - srj))
    Linv, Uinv = gb_inv(L), gb_inv(U)
    I = (Linv - Uinv + (r - srj*srj*rinv)*(Uinv*Uinv - Linv*Linv)/4 +
         gb_log(L*Uinv)*rinv/2) / 2
    # Atom i is entirely inside atom j
    I += ifelse(ori < (srj - r), 2 * (gb_inv(ori) - Linv), zero(I))
    return ifelse(ori < U, I, zero(I))
end

@inline neck_integral(     ::Nothing, i, j, r, ori, orj) = zero(r)
@inline neck_integral_grad(::Nothing, i, j, r, ori, orj) = zero(r)

# The neck lookup only depends on the radius class of the two atoms
@inline function neck_d0_m0(neck, i, j)
    @inbounds ci, cj = neck.classes[i], neck.classes[j]
    @inbounds return ustrip(neck.lu, neck.d0s[ci, cj]), ustrip(inv(neck.lu), neck.m0s[ci, cj])
end

@inline function neck_in_range(neck, r, ori, orj)
    return r < (ori + orj + 2 * neck.offset + neck.cut)
end

@inline function neck_integral(neck, i, j, r, ori, orj)
    d0, m0 = neck_d0_m0(neck, i, j)
    rd = neck.dist_scale * (r - d0)
    rd2 = rd * rd
    denom = 1 + rd2 + 3 * rd2 * rd2 * rd2 / 10
    I = neck.scale * m0 / denom
    return ifelse(neck_in_range(neck, r, ori, orj), I, zero(I))
end

# Derivative of the neck integral with respect to the atomic distance
@inline function neck_integral_grad(neck, i, j, r, ori, orj)
    d0, m0 = neck_d0_m0(neck, i, j)
    rd = neck.dist_scale * (r - d0)
    rd2 = rd * rd
    rd4 = rd2 * rd2
    denom = 1 + rd2 + 3 * rd4 * rd2 / 10
    numer = 2 * rd + 9 * rd4 * rd / 5
    denom_inv = gb_inv(denom)
    I_grad = -neck.dist_scale * neck.scale * m0 * numer * denom_inv * denom_inv
    return ifelse(neck_in_range(neck, r, ori, orj), I_grad, zero(I_grad))
end

# Non-polar solvation force from the ACE approximation, the ratio is raised to the
#   sixth power by repeated multiplication as the power function is slow on GPUs
@inline function ace_born_force(or, offset, probe_radius, sa_factor, B)
    radius = or + offset
    rp = radius + probe_radius
    if B > zero(B)
        ratio = radius / B
        ratio3 = ratio * ratio * ratio
        return -6 * sa_factor * rp * rp * ratio3 * ratio3 / B
    else
        return zero(sa_factor * rp * rp / radius)
    end
end

# Born radius and its gradient with respect to the Born radii integral, without units
function born_radii_sum(or, offset, I, α, β, γ)
    radius = or + offset
    ψ = I * or
    ψ2 = ψ^2
    tanh_sum = tanh(α * ψ - β * ψ2 + γ * ψ2 * ψ)
    B = inv(inv(or) - tanh_sum / radius)
    grad_term = or * (α - 2 * β * ψ + 3 * γ * ψ2)
    B_grad = (1 - tanh_sum^2) * grad_term / radius
    return B, B_grad
end

# Born radii integral for atom i summed over the atoms in jrange
@inline function born_radii_partial(coords, or, sor, neck, sqdist_cutoff, boundary, i, jrange)
    lu = unit(eltype(or))
    @inbounds begin
        coord_i, ori = ustrip.(lu, coords[i]), ustrip(lu, or[i])
        I_sum = zero(ori)
        @simd for j in jrange
            r2 = sum(abs2, vector(coord_i, ustrip.(lu, coords[j]), boundary))
            in_range = (j != i) & !iszero(r2) & (r2 <= sqdist_cutoff)
            r = sqrt(ifelse(in_range, r2, one(r2)))
            rinv = gb_inv(r)
            I = born_integral(r, rinv, ori, ustrip(lu, sor[j])) +
                neck_integral(neck, i, j, r, ori, ustrip(lu, or[j]))
            I_sum += ifelse(in_range, I, zero(I))
        end
        return I_sum
    end
end

# Polarisation energy derivatives for a pair of atoms, also used for the self term
#   by passing a squared distance of zero and Bj equal to Bi
@inline function gb_pair_gpol(r2, charge_ij, Bi, Bj, factor_solute, factor_solvent, kappa)
    alpha2_ij = Bi * Bj
    D_term = gb_div(r2, 4 * alpha2_ij)
    exp_term = gb_exp(-D_term)
    denominator2 = r2 + alpha2_ij * exp_term
    denominator = sqrt(denominator2)
    denominator_inv = gb_inv(denominator)
    denominator2_inv = denominator_inv * denominator_inv
    if iszero(kappa)
        pre_factor = factor_solute + factor_solvent
    else
        exp_kappa = gb_exp(-kappa * denominator)
        pre_factor = factor_solute + exp_kappa * factor_solvent +
                        kappa * denominator * exp_kappa * factor_solvent
    end
    Gpol = pre_factor * charge_ij * denominator_inv
    dGpol_dr = -Gpol * (1 - exp_term/4) * denominator2_inv
    dGpol_dalpha2_ij = -Gpol * exp_term * (1 + D_term) * denominator2_inv / 2
    return dGpol_dr, dGpol_dalpha2_ij
end

# Born force and direct force on atom i from the atoms in jrange
# The self term is included when jrange contains i
@inline function gb_force_1_partial(coords, charges, Bs, sqdist_cutoff, factor_solute,
                                    factor_solvent, kappa, boundary, lu, i, jrange)
    @inbounds begin
        coord_i, charge_i, Bi = ustrip.(lu, coords[i]), charges[i], ustrip(lu, Bs[i])
        f_i = zero(coord_i)
        born_force_i = zero(eltype(coord_i))
        for j in jrange
            if j == i
                _, dGpol_dalpha2_ij = gb_pair_gpol(zero(born_force_i), charge_i^2, Bi, Bi,
                                                   factor_solute, factor_solvent, kappa)
                born_force_i += dGpol_dalpha2_ij * Bi
            else
                dr = vector(coord_i, ustrip.(lu, coords[j]), boundary)
                r2 = sum(abs2, dr)
                if r2 > sqdist_cutoff
                    continue
                end
                Bj = ustrip(lu, Bs[j])
                dGpol_dr, dGpol_dalpha2_ij = gb_pair_gpol(r2, charge_i * charges[j], Bi, Bj,
                                                    factor_solute, factor_solvent, kappa)
                born_force_i += dGpol_dalpha2_ij * Bj
                f_i += dr * dGpol_dr
            end
        end
        return born_force_i, f_i
    end
end

# Chain rule term for the derivative of the Born radius of atom i with respect to
#   the distance to atom j
@inline function gb_force_2_de(r, rinv, r2inv, ori, srj, bi, I_grad)
    rsrj = r + srj
    Lval = max(ori, abs(r - srj))
    L = gb_inv(Lval)
    U = gb_inv(rsrj)
    t3 = (1 + srj*srj*r2inv)*(L*L - U*U)/8 + gb_log(Lval*U)*r2inv/4
    de = bi * (t3 - I_grad) * rinv
    return ifelse(ori < rsrj, de, zero(de))
end

# Force on atom i from the change in the Born radii of atom i and of the atoms in jrange
@inline function gb_force_2_partial(coords, or, sor, born_forces_scaled, neck,
                                    sqdist_cutoff, boundary, i, jrange)
    lu = unit(eltype(or))
    @inbounds begin
        coord_i, ori = ustrip.(lu, coords[i]), ustrip(lu, or[i])
        sori, bfi = ustrip(lu, sor[i]), ustrip(born_forces_scaled[i])
        f_i = zero(coord_i)
        @simd for j in jrange
            dr = vector(coord_i, ustrip.(lu, coords[j]), boundary)
            r2 = sum(abs2, dr)
            in_range = (j != i) & !iszero(r2) & (r2 <= sqdist_cutoff)
            r = sqrt(ifelse(in_range, r2, one(r2)))
            rinv = gb_inv(r)
            r2inv = rinv * rinv
            orj = ustrip(lu, or[j])
            de = gb_force_2_de(r, rinv, r2inv, ori, ustrip(lu, sor[j]), bfi,
                               neck_integral_grad(neck, i, j, r, ori, orj)) +
                 gb_force_2_de(r, rinv, r2inv, orj, sori, ustrip(born_forces_scaled[j]),
                               neck_integral_grad(neck, j, i, r, orj, ori))
            f_i -= dr * ifelse(in_range, de, zero(de))
        end
        return f_i
    end
end

# Energy of atom i with the atoms in jrange, the self term is included when jrange
#   contains i and each pair should only appear once across the ranges used
@inline function gb_energy_partial(coords, charges, Bs, or, sqdist_cutoff, factor_solute,
                                   factor_solvent, kappa, offset, probe_radius, sa_factor,
                                   use_ACE, boundary, lu, dist_cutoff_inv, i, jrange)
    @inbounds begin
        coord_i, charge_i, Bi = ustrip.(lu, coords[i]), charges[i], ustrip(lu, Bs[i])
        E = zero(Bi)
        for j in jrange
            if j == i
                if iszero(kappa)
                    pre_factor = factor_solute + factor_solvent
                else
                    pre_factor = factor_solute + gb_exp(-kappa * Bi) * factor_solvent
                end
                E += pre_factor * (charge_i^2) / (2*Bi)
                if use_ACE && (Bi > zero(Bi))
                    radius_i = ustrip(lu, or[i]) + offset
                    E += sa_factor * (radius_i + probe_radius)^2 * (radius_i / Bi)^6
                end
            else
                r2 = sum(abs2, vector(coord_i, ustrip.(lu, coords[j]), boundary))
                if r2 > sqdist_cutoff
                    continue
                end
                alpha2_ij = Bi * ustrip(lu, Bs[j])
                f = sqrt(r2 + alpha2_ij * gb_exp(-gb_div(r2, 4 * alpha2_ij)))
                f_cutoff = gb_inv(f) - dist_cutoff_inv
                if iszero(kappa)
                    pre_factor = factor_solute + factor_solvent
                else
                    pre_factor = factor_solute + gb_exp(-kappa * f) * factor_solvent
                end
                E += pre_factor * charge_i * charges[j] * f_cutoff
            end
        end
        return E
    end
end

#=
CPU implementation. The Born radii loop splits the atoms into contiguous chunks across
the threads, since each thread only writes to its own atoms no reduction is required.
The force loops only look at each pair once and scatter the result to both atoms, so
each thread accumulates into its own force buffer which is reduced at the end. The atoms
are along the first dimension of that buffer so that the scatter is contiguous, which is
needed for the loop to vectorise.
=#

@inline function gbsa_chunk_range(n_atoms::Integer, chunk_i::Integer, n_chunks::Integer)
    return (((chunk_i - 1) * n_atoms) ÷ n_chunks + 1):((chunk_i * n_atoms) ÷ n_chunks)
end

# The components are added explicitly rather than in a loop over the dimensions, which is
#   what lets the compiler see the scatter as a contiguous store and vectorise the loop
@inline function gbsa_chunk_add!(chunk, j, f::StaticVector{2})
    @inbounds chunk[j, 1] += f[1]
    @inbounds chunk[j, 2] += f[2]
    return nothing
end

@inline function gbsa_chunk_add!(chunk, j, f::StaticVector{3})
    @inbounds chunk[j, 1] += f[1]
    @inbounds chunk[j, 2] += f[2]
    @inbounds chunk[j, 3] += f[3]
    return nothing
end

# The pair loops work without units, so check once per call that the unit the
#   interaction gives is the one the system expects
function gbsa_force_units(inter::AbstractGBSA, force_units)
    inter_force_units = unit(inter.factor_solute / oneunit(inter.offset)^2)
    if inter_force_units != force_units
        error("system force units are $force_units but the implicit solvent interaction ",
              "gives $inter_force_units")
    end
    return force_units
end

gbsa_energy_unit(inter::AbstractGBSA) = unit(inter.factor_solute / oneunit(inter.offset))

# Calculate Born radii and the gradients of the Born radii with respect to the
#   Born radii integral
# Custom GBSA methods should implement this function
function born_radii_and_grad!(inter::AbstractGBSA, coords, boundary, n_threads::Integer)
    n_atoms = length(coords)
    if n_threads > 1
        Threads.@threads for chunk_i in 1:n_threads
            born_radii_chunk!(inter, coords, boundary, n_atoms, chunk_i, n_threads)
        end
    else
        born_radii_chunk!(inter, coords, boundary, n_atoms, 1, 1)
    end
    return inter.buffer_Bs, inter.buffer_B_grads
end

@noinline function born_radii_chunk!(inter, coords, boundary, n_atoms, chunk_i, n_chunks)
    lu = gb_length_unit(inter)
    or, sor, neck = inter.offset_radii, inter.scaled_offset_radii, gb_neck_data(inter)
    offset, sqdist_cutoff = ustrip(lu, inter.offset), gb_sqdist_cutoff(inter)
    bnd = ustrip(lu, boundary)
    αs, βs, γs = gb_αs(inter), gb_βs(inter), gb_γs(inter)
    Bs, B_grads = inter.buffer_Bs, inter.buffer_B_grads
    B_unit = oneunit(eltype(Bs))
    @inbounds for i in gbsa_chunk_range(n_atoms, chunk_i, n_chunks)
        I_sum = born_radii_partial(coords, or, sor, neck, sqdist_cutoff, bnd, i, 1:n_atoms)
        B, B_grad = born_radii_sum(ustrip(lu, or[i]), offset, I_sum, atom_param(αs, i),
                                   atom_param(βs, i), atom_param(γs, i))
        Bs[i] = B * B_unit
        B_grads[i] = B_grad
    end
    return nothing
end

# Each thread writes to its own slice of the force chunk buffer, meaning that each
#   pair only has to be looked at once and the results can be scattered to both atoms
function forces_gbsa!(fs, born_forces, sys, inter, Bs, B_grads, atom_charges,
                      n_threads::Integer)
    n_atoms = length(sys)
    coords, boundary = sys.coords, sys.boundary
    force_units = gbsa_force_units(inter, sys.force_units)
    chunks = inter.buffer_force_chunks
    n_chunks = n_threads

    if n_chunks > 1
        Threads.@threads for chunk_i in 1:n_chunks
            gbsa_force_1_chunk!(chunks[chunk_i], coords, boundary, inter, Bs, atom_charges,
                                n_atoms, chunk_i, n_chunks)
        end
    else
        gbsa_force_1_chunk!(chunks[1], coords, boundary, inter, Bs, atom_charges,
                            n_atoms, 1, 1)
    end

    gbsa_reduce_born_forces!(born_forces, chunks, n_atoms, n_chunks)
    inter.buffer_born_forces_scaled .= born_forces .* Bs .^ 2 .* B_grads

    if n_chunks > 1
        Threads.@threads for chunk_i in 1:n_chunks
            gbsa_force_2_chunk!(chunks[chunk_i], coords, boundary, inter, n_atoms,
                                chunk_i, n_chunks)
        end
    else
        gbsa_force_2_chunk!(chunks[1], coords, boundary, inter, n_atoms, 1, 1)
    end

    gbsa_reduce_forces!(fs, chunks, n_atoms, n_chunks, force_units)
    return fs
end

# Only the upper triangle of pairs is looked at as the polarisation force is symmetric
@noinline function gbsa_force_1_chunk!(chunk, coords, boundary, inter, Bs, atom_charges,
                                       n_atoms, chunk_i, n_chunks)
    lu = gb_length_unit(inter)
    sqdist_cutoff, bnd = gb_sqdist_cutoff(inter), ustrip(lu, boundary)
    kappa = ustrip(inv(lu), inter.kappa)
    factor_solute, factor_solvent = ustrip(inter.factor_solute), ustrip(inter.factor_solvent)
    fill!(chunk, zero(eltype(chunk)))
    @inbounds begin
        # Interleaved as the number of pairs decreases with atom index
        for i in chunk_i:n_chunks:n_atoms
            coord_i, charge_i = ustrip.(lu, coords[i]), atom_charges[i]
            Bi = ustrip(lu, Bs[i])
            f_i = zero(coord_i)
            _, dGpol_dalpha2_ii = gb_pair_gpol(zero(Bi), charge_i^2, Bi, Bi,
                                               factor_solute, factor_solvent, kappa)
            born_force_i = dGpol_dalpha2_ii * Bi
            for j in (i + 1):n_atoms
                dr = vector(coord_i, ustrip.(lu, coords[j]), bnd)
                r2 = sum(abs2, dr)
                if r2 > sqdist_cutoff
                    continue
                end
                Bj = ustrip(lu, Bs[j])
                dGpol_dr, dGpol_dalpha2_ij = gb_pair_gpol(r2, charge_i * atom_charges[j],
                                        Bi, Bj, factor_solute, factor_solvent, kappa)
                born_force_i += dGpol_dalpha2_ij * Bj
                chunk[j, 4] += dGpol_dalpha2_ij * Bi
                fdr = dr * dGpol_dr
                f_i += fdr
                gbsa_chunk_add!(chunk, j, -fdr)
            end
            gbsa_chunk_add!(chunk, i, f_i)
            chunk[i, 4] += born_force_i
        end
    end
    return nothing
end

# The Born radii chain rule term is not symmetric, but the sum of the two directions of
#   a pair is, so only the upper triangle of pairs has to be looked at and the distance
#   can be shared between the two directions
@noinline function gbsa_force_2_chunk!(chunk, coords, boundary, inter, n_atoms,
                                       chunk_i, n_chunks)
    lu = gb_length_unit(inter)
    or, sor, neck = inter.offset_radii, inter.scaled_offset_radii, gb_neck_data(inter)
    sqdist_cutoff, bnd = gb_sqdist_cutoff(inter), ustrip(lu, boundary)
    born_forces_scaled = inter.buffer_born_forces_scaled
    @inbounds for i in chunk_i:n_chunks:n_atoms
        coord_i, ori, sori = ustrip.(lu, coords[i]), ustrip(lu, or[i]), ustrip(lu, sor[i])
        bfi = ustrip(born_forces_scaled[i])
        f_i = zero(coord_i)
        @simd for j in (i + 1):n_atoms
            dr = vector(coord_i, ustrip.(lu, coords[j]), bnd)
            r2 = sum(abs2, dr)
            in_range = !iszero(r2) & (r2 <= sqdist_cutoff)
            r = sqrt(ifelse(in_range, r2, one(r2)))
            rinv = gb_inv(r)
            r2inv = rinv * rinv
            orj = ustrip(lu, or[j])
            de_pair = gb_force_2_de(r, rinv, r2inv, ori, ustrip(lu, sor[j]), bfi,
                                    neck_integral_grad(neck, i, j, r, ori, orj)) +
                      gb_force_2_de(r, rinv, r2inv, orj, sori, ustrip(born_forces_scaled[j]),
                                    neck_integral_grad(neck, j, i, r, orj, ori))
            fdr = dr * ifelse(in_range, de_pair, zero(de_pair))
            f_i -= fdr
            gbsa_chunk_add!(chunk, j, fdr)
        end
        gbsa_chunk_add!(chunk, i, f_i)
    end
    return nothing
end

@noinline function gbsa_reduce_born_forces!(born_forces, chunks, n_atoms, n_chunks)
    born_force_unit = unit(eltype(born_forces))
    @inbounds for i in 1:n_atoms
        bf = zero(eltype(eltype(chunks)))
        for chunk_i in 1:n_chunks
            bf += chunks[chunk_i][i, 4]
        end
        born_forces[i] += bf * born_force_unit
    end
    return born_forces
end

@noinline function gbsa_reduce_forces!(fs, chunks, n_atoms, n_chunks, force_units)
    D = length(eltype(fs))
    FT = eltype(eltype(chunks))
    @inbounds for i in 1:n_atoms
        f = zero(SVector{D, FT})
        for chunk_i in 1:n_chunks
            chunk = chunks[chunk_i]
            f += SVector{D, FT}(ntuple(dim -> chunk[i, dim], Val(D)))
        end
        fs[i] = fs[i] .+ f .* force_units
    end
    return fs
end

function gbsa_energy(sys, inter, Bs, atom_charges, n_threads::Integer)
    n_atoms = length(sys)
    energy_unit = gbsa_energy_unit(inter)
    if n_threads > 1
        pes = inter.buffer_pes
        n_chunks = n_threads
        # The atoms are interleaved as the number of pairs decreases with atom index
        Threads.@threads for chunk_i in 1:n_chunks
            pes[chunk_i] = gbsa_energy_chunk(sys.coords, sys.boundary, inter, Bs, atom_charges,
                                             n_atoms, chunk_i, n_chunks) * energy_unit
        end
        E = zero(eltype(pes))
        @inbounds for chunk_i in 1:n_chunks
            E += pes[chunk_i]
        end
        return E
    else
        return gbsa_energy_chunk(sys.coords, sys.boundary, inter, Bs, atom_charges,
                                 n_atoms, 1, 1) * energy_unit
    end
end

@noinline function gbsa_energy_chunk(coords, boundary, inter, Bs, atom_charges, n_atoms,
                                     chunk_i, n_chunks)
    lu = gb_length_unit(inter)
    or, offset = inter.offset_radii, ustrip(lu, inter.offset)
    dist_cutoff = ustrip(lu, inter.dist_cutoff)
    dist_cutoff_inv = iszero(dist_cutoff) ? zero(inv(dist_cutoff)) : inv(dist_cutoff)
    sqdist_cutoff, bnd = gb_sqdist_cutoff(inter), ustrip(lu, boundary)
    kappa = ustrip(inv(lu), inter.kappa)
    factor_solute, factor_solvent = ustrip(inter.factor_solute), ustrip(inter.factor_solvent)
    probe_radius, sa_factor = ustrip(lu, inter.probe_radius), ustrip(inter.sa_factor)
    use_ACE = inter.use_ACE
    E = zero(factor_solute / offset)
    @inbounds for i in chunk_i:n_chunks:n_atoms
        E += gb_energy_partial(coords, atom_charges, Bs, or, sqdist_cutoff, factor_solute,
                               factor_solvent, kappa, offset, probe_radius, sa_factor,
                               use_ACE, bnd, lu, dist_cutoff_inv, i, i:n_atoms)
    end
    return E
end

#=
GPU implementation, one thread deals with one atom and a strided slice of the other
atoms, meaning that the only atomic operations are one per thread at the end.
=#

# Number of slices of the other atoms, chosen to give enough threads to fill the GPU
#   while leaving each thread with a reasonable number of pairs to work on
function gbsa_n_chunks(n_atoms::Integer)
    n_threads_min = gpu_threads_env("MOLLY_GPUMINTHREADS_IMPLICIT", 128000)
    max_chunks = max(cld(n_atoms, 4), 1)
    return clamp(cld(n_threads_min, max(n_atoms, 1)), 1, max_chunks)
end

gpu_threads_gbsa() = gpu_threads_env("MOLLY_GPUNTHREADS_IMPLICIT", 128)

@inline function gbsa_atom_chunk(idx, n_atoms)
    # Consecutive threads deal with consecutive atoms to keep the writes coalesced
    return (idx - 1) % n_atoms + 1, (idx - 1) ÷ n_atoms + 1
end

function born_radii_and_grad!(inter::AbstractGBSA, coords::AbstractGPUArray, boundary,
                              n_threads::Integer)
    n_atoms = length(coords)
    n_chunks = gbsa_n_chunks(n_atoms)
    backend = get_backend(coords)
    lu = gb_length_unit(inter)
    Is_nounits = inter.buffer_Is_nounits

    kernel_1! = gbsa_born_kernel!(backend, gpu_threads_gbsa())
    kernel_1!(Is_nounits, coords, inter.offset_radii, inter.scaled_offset_radii,
              gb_neck_data(inter), gb_sqdist_cutoff(inter), ustrip(lu, boundary), n_chunks;
              ndrange=(n_atoms * n_chunks))

    kernel_2! = gbsa_born_sum_kernel!(backend, gpu_threads_gbsa())
    kernel_2!(inter.buffer_Bs, inter.buffer_B_grads, Is_nounits, inter.offset_radii,
              ustrip(lu, inter.offset), gb_αs(inter), gb_βs(inter), gb_γs(inter),
              oneunit(eltype(inter.buffer_Bs)), lu; ndrange=n_atoms)

    return inter.buffer_Bs, inter.buffer_B_grads
end

@kernel inbounds=true function gbsa_born_kernel!(Is_nounits, @Const(coords), @Const(or),
                            @Const(sor), neck, sqdist_cutoff, boundary, n_chunks)
    idx = @index(Global, Linear)
    n_atoms = length(coords)

    if idx <= n_atoms * n_chunks
        i, chunk_i = gbsa_atom_chunk(idx, n_atoms)
        I_sum = born_radii_partial(coords, or, sor, neck, sqdist_cutoff, boundary, i,
                                   chunk_i:n_chunks:n_atoms)
        Atomix.@atomic Is_nounits[i] += convert(eltype(Is_nounits), I_sum)
    end
end

# The integral buffer is left zeroed for the next call rather than being filled at the
#   start, which saves a device memory operation per call
@kernel inbounds=true function gbsa_born_sum_kernel!(Bs, B_grads, Is_nounits, @Const(or),
                            offset, αs, βs, γs, B_unit, lu)
    i = @index(Global, Linear)

    if i <= length(Bs)
        B, B_grad = born_radii_sum(ustrip(lu, or[i]), offset, Is_nounits[i],
                                   atom_param(αs, i), atom_param(βs, i), atom_param(γs, i))
        Bs[i] = B * B_unit
        B_grads[i] = B_grad
        Is_nounits[i] = zero(eltype(Is_nounits))
    end
end

function forces_gbsa!(fs, born_forces, sys::System{D, <:AbstractGPUArray, T}, inter, Bs,
                      B_grads, atom_charges, n_threads::Integer) where {D, T}
    n_atoms = length(sys)
    n_chunks = gbsa_n_chunks(n_atoms)
    backend = get_backend(sys.coords)
    lu = gb_length_unit(inter)
    force_units = gbsa_force_units(inter, sys.force_units)
    sqdist_cutoff, bnd = gb_sqdist_cutoff(inter), ustrip(lu, sys.boundary)
    # The accumulators are zeroed by gbsa_setup!
    fs_mat = inter.buffer_fs_mat
    born_forces_mod = inter.buffer_born_forces_mod

    kernel_1! = gbsa_force_1_kernel!(backend, gpu_threads_gbsa())
    kernel_1!(fs_mat, born_forces_mod, sys.coords, atom_charges, Bs, bnd, sqdist_cutoff,
              ustrip(inter.factor_solute), ustrip(inter.factor_solvent),
              ustrip(inv(lu), inter.kappa), lu, n_chunks, Val(D);
              ndrange=(n_atoms * n_chunks))

    kernel_s! = gbsa_born_scale_kernel!(backend, gpu_threads_gbsa())
    kernel_s!(born_forces, inter.buffer_born_forces_scaled, born_forces_mod, Bs, B_grads,
              unit(eltype(born_forces)); ndrange=n_atoms)

    kernel_2! = gbsa_force_2_kernel!(backend, gpu_threads_gbsa())
    kernel_2!(fs_mat, sys.coords, inter.offset_radii, inter.scaled_offset_radii,
              inter.buffer_born_forces_scaled, gb_neck_data(inter), bnd, sqdist_cutoff,
              n_chunks, Val(D); ndrange=(n_atoms * n_chunks))

    kernel_a! = gbsa_add_forces_kernel!(backend, gpu_threads_gbsa())
    kernel_a!(fs, fs_mat, Val(D), Val(T), Val(force_units); ndrange=n_atoms)
    return fs
end

@kernel inbounds=true function gbsa_force_1_kernel!(fs_mat, born_forces_mod, @Const(coords),
                            @Const(charges), @Const(Bs), boundary, sqdist_cutoff,
                            factor_solute, factor_solvent, kappa, lu, n_chunks,
                            ::Val{D}) where D
    idx = @index(Global, Linear)
    n_atoms = length(coords)

    if idx <= n_atoms * n_chunks
        i, chunk_i = gbsa_atom_chunk(idx, n_atoms)
        born_force_i, f_i = gb_force_1_partial(coords, charges, Bs, sqdist_cutoff,
                                factor_solute, factor_solvent, kappa, boundary, lu, i,
                                chunk_i:n_chunks:n_atoms)
        Atomix.@atomic born_forces_mod[i] += convert(eltype(born_forces_mod), born_force_i)
        for dim in 1:D
            Atomix.@atomic fs_mat[dim, i] += convert(eltype(fs_mat), f_i[dim])
        end
    end
end

# Add the Born forces from the polarisation energy to those from the ACE term and apply
#   the chain rule factor for the derivative of the Born radius
@kernel inbounds=true function gbsa_born_scale_kernel!(born_forces, born_forces_scaled,
                    @Const(born_forces_mod), @Const(Bs), @Const(B_grads), born_force_unit)
    i = @index(Global, Linear)

    if i <= length(born_forces)
        bf = born_forces[i] + born_forces_mod[i] * born_force_unit
        born_forces[i] = bf
        born_forces_scaled[i] = bf * Bs[i]^2 * B_grads[i]
    end
end

@kernel inbounds=true function gbsa_force_2_kernel!(fs_mat, @Const(coords), @Const(or),
                            @Const(sor), @Const(born_forces_scaled), neck, boundary,
                            sqdist_cutoff, n_chunks, ::Val{D}) where D
    idx = @index(Global, Linear)
    n_atoms = length(coords)

    if idx <= n_atoms * n_chunks
        i, chunk_i = gbsa_atom_chunk(idx, n_atoms)
        f_i = gb_force_2_partial(coords, or, sor, born_forces_scaled, neck, sqdist_cutoff,
                                 boundary, i, chunk_i:n_chunks:n_atoms)
        for dim in 1:D
            Atomix.@atomic fs_mat[dim, i] += convert(eltype(fs_mat), f_i[dim])
        end
    end
end

@kernel inbounds=true function gbsa_add_forces_kernel!(fs, @Const(fs_mat), ::Val{D},
                            ::Val{T}, ::Val{force_units}) where {D, T, force_units}
    i = @index(Global, Linear)

    if i <= length(fs)
        f = SVector{D, T}(ntuple(dim -> fs_mat[dim, i], Val(D)))
        fs[i] = fs[i] .+ apply_force_units_gpu(f, Val(force_units))
    end
end

function gbsa_energy(sys::System{<:Any, <:AbstractGPUArray}, inter, Bs, atom_charges,
                     n_threads::Integer)
    n_atoms = length(sys)
    n_chunks = gbsa_n_chunks(n_atoms)
    backend = get_backend(sys.coords)
    lu = gb_length_unit(inter)
    dist_cutoff = ustrip(lu, inter.dist_cutoff)
    dist_cutoff_inv = iszero(dist_cutoff) ? zero(inv(dist_cutoff)) : inv(dist_cutoff)
    # The energy accumulator is zeroed by gbsa_setup!
    pes = inter.buffer_pes

    kernel! = gbsa_energy_kernel!(backend, gpu_threads_gbsa())
    kernel!(pes, sys.coords, atom_charges, Bs, inter.offset_radii, ustrip(lu, sys.boundary),
            gb_sqdist_cutoff(inter), ustrip(inter.factor_solute), ustrip(inter.factor_solvent),
            ustrip(inv(lu), inter.kappa), ustrip(lu, inter.offset),
            ustrip(lu, inter.probe_radius), ustrip(inter.sa_factor), inter.use_ACE, lu,
            dist_cutoff_inv, n_chunks; ndrange=(n_atoms * n_chunks))

    return sum(pes) * gbsa_energy_unit(inter)
end

@kernel inbounds=true function gbsa_energy_kernel!(pes, @Const(coords), @Const(charges),
                            @Const(Bs), @Const(or), boundary, sqdist_cutoff, factor_solute,
                            factor_solvent, kappa, offset, probe_radius, sa_factor,
                            use_ACE, lu, dist_cutoff_inv, n_chunks)
    idx = @index(Global, Linear)
    n_atoms = length(coords)

    if idx <= n_atoms * n_chunks
        i, chunk_i = gbsa_atom_chunk(idx, n_atoms)
        E = gb_energy_partial(coords, charges, Bs, or, sqdist_cutoff, factor_solute,
                              factor_solvent, kappa, offset, probe_radius, sa_factor,
                              use_ACE, boundary, lu, dist_cutoff_inv, i,
                              (i + chunk_i - 1):n_chunks:n_atoms)
        Atomix.@atomic pes[i] += convert(eltype(pes), E)
    end
end

# Per-atom quantities that depend on the Born radii, along with zeroing the accumulators
#   that the GPU kernels add to
function gbsa_setup!(inter::AbstractGBSA, sys, Bs)
    if inter.use_ACE
        inter.buffer_born_forces .= ace_born_force.(inter.offset_radii, inter.offset,
                                            inter.probe_radius, inter.sa_factor, Bs)
    else
        fill!(inter.buffer_born_forces, zero(eltype(inter.buffer_born_forces)))
    end
    inter.buffer_atom_charges .= charge.(sys.atoms)
    return nothing
end

function gbsa_setup!(inter::AbstractGBSA, sys::System{<:Any, <:AbstractGPUArray}, Bs)
    n_atoms = length(sys)
    kernel! = gbsa_setup_kernel!(get_backend(sys.coords), gpu_threads_gbsa())
    kernel!(inter.buffer_atom_charges, inter.buffer_born_forces, inter.buffer_born_forces_mod,
            inter.buffer_fs_mat, inter.buffer_pes, sys.atoms, Bs, inter.offset_radii,
            inter.offset, inter.probe_radius, inter.sa_factor, inter.use_ACE; ndrange=n_atoms)
    return nothing
end

@kernel inbounds=true function gbsa_setup_kernel!(atom_charges, born_forces, born_forces_mod,
                            fs_mat, pes, @Const(atoms), @Const(Bs), @Const(or), offset,
                            probe_radius, sa_factor, use_ACE)
    i = @index(Global, Linear)

    if i <= length(atom_charges)
        atom_charges[i] = charge(atoms[i])
        if use_ACE
            born_forces[i] = ace_born_force(or[i], offset, probe_radius, sa_factor, Bs[i])
        else
            born_forces[i] = zero(eltype(born_forces))
        end
        born_forces_mod[i] = zero(eltype(born_forces_mod))
        pes[i] = zero(eltype(pes))
        for dim in axes(fs_mat, 1)
            fs_mat[dim, i] = zero(eltype(fs_mat))
        end
    end
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs, sys,
                        inter::AbstractGBSA; n_threads::Integer=Threads.nthreads(), kwargs...)
    check_gbsa_n_threads(inter.buffer_force_chunks, n_threads)
    Bs, B_grads = born_radii_and_grad!(inter, sys.coords, sys.boundary, n_threads)
    gbsa_setup!(inter, sys, Bs)
    forces_gbsa!(fs, inter.buffer_born_forces, sys, inter, Bs, B_grads,
                 inter.buffer_atom_charges, n_threads)
    return fs
end

function AtomsCalculators.potential_energy(sys, inter::AbstractGBSA;
                                           n_threads::Integer=Threads.nthreads(), kwargs...)
    check_gbsa_n_threads(inter.buffer_force_chunks, n_threads)
    Bs, B_grads = born_radii_and_grad!(inter, sys.coords, sys.boundary, n_threads)
    gbsa_setup!(inter, sys, Bs)
    return gbsa_energy(sys, inter, Bs, inter.buffer_atom_charges, n_threads)
end
