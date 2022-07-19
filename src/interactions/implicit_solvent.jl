# Implicit solvent models
# Based on the OpenMM source code

export
    AbstractGBSA,
    ImplicitSolventOBC,
    ImplicitSolventGBN2,
    born_radii_and_grad

"""
Generalized Born (GB) implicit solvent models augmented with the
hydrophobic solvent accessible surface area (SA) term.
Custom GBSA methods should sub-type this type.
"""
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
    for (i, j) in zip(bonds.is, bonds.js)
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

# We use a full atom pairwise table rather than looking up a value with the atom radius
# This works better with broadcasting
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

function lookup_table(full_table::AbstractArray{T}, radii::AbstractArray{<:AbstractFloat}) where T
    return lookup_table(full_table, radii * u"nm")
end

"""
    ImplicitSolventOBC(atoms, atoms_data, bonds)

Onufriev-Bashford-Case GBSA model.
Should be used along with a [`Coulomb`](@ref) or [`CoulombReactionField`](@ref) interaction.
The keyword argument `use_OBC2` determines whether to use parameter set
I (`false`, the default) or II (`true`).
"""
struct ImplicitSolventOBC{T, D, V, K, S, F, I} <: AbstractGBSA
    offset_radii::V
    scaled_offset_radii::V
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
    is::I
    js::I
end

function ImplicitSolventOBC(atoms::AbstractArray{Atom{T, M, D, E}},
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
                            element_to_screen=obc_element_to_screen) where {T, M, D, E}
    units = dimension(D) == u"𝐋"
    radii = mbondi2_radii(atoms_data, bonds; element_to_radius=element_to_radius)

    if units
        offset_radii = T.(radii .- offset)
    else
        offset_radii = ustrip.(T.(radii .- offset))
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
    inds_i = hcat([collect(1:n_atoms) for i in 1:n_atoms]...)
    inds_j = permutedims(inds_i, (2, 1))

    coulomb_const = units ? coulombconst : ustrip(coulombconst)
    if !iszero(solute_dielectric)
        factor_solute = -T(coulomb_const) / T(solute_dielectric)
    else
        factor_solute = zero(T(coulomb_const))
    end
    if !iszero(solvent_dielectric)
        factor_solvent = T(coulomb_const) / T(solvent_dielectric)
    else
        factor_solvent = zero(T(coulomb_const))
    end

    if isa(atoms, CuArray)
        or = CuArray(offset_radii) # cu converts to Float32
        sor = CuArray(scaled_offset_radii)
        is, js = cu(inds_i), cu(inds_j)
    else
        or = offset_radii
        sor = scaled_offset_radii
        is, js = inds_i, inds_j
    end

    if units
        return ImplicitSolventOBC{T, D, typeof(or), typeof(T(kappa)), typeof(T(sa_factor)),
                        typeof(factor_solute), typeof(is)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(kappa), offset,
                    dist_cutoff, use_ACE, α, β, γ, probe_radius, T(sa_factor),
                    factor_solute, factor_solvent, is, js)
    else
        return ImplicitSolventOBC{T, T, typeof(or), typeof(T(ustrip(kappa))), T, T, typeof(is)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(ustrip(kappa)),
                    ustrip(offset), ustrip(dist_cutoff), use_ACE, α, β, γ, ustrip(probe_radius),
                    ustrip(sa_factor), factor_solute, factor_solvent, is, js)
    end
end

"""
    ImplicitSolventGBN2(atoms, atoms_data, bonds)

GBn2 solvation model.
Should be used along with a [`Coulomb`](@ref) or [`CoulombReactionField`](@ref) interaction.
"""
struct ImplicitSolventGBN2{T, D, VT, VD, K, S, F, I, TD, TM} <: AbstractGBSA
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
    is::I
    js::I
    d0s::TD
    m0s::TM
    neck_scale::T
    neck_cut::D
end

function ImplicitSolventGBN2(atoms::AbstractArray{Atom{T, M, D, E}},
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
                                data_m0=gbn2_data_m0) where {T, M, D, E}
    units = dimension(D) == u"𝐋"
    radii = mbondi3_radii(atoms_data, bonds; element_to_radius=element_to_radius)
    nucleic_acid_residues = ("A", "C", "G", "U", "DA", "DC", "DG", "DT")

    if units
        offset_radii = T.(radii .- offset)
    else
        offset_radii = T.(ustrip.(radii) .- ustrip(offset))
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
    inds_i = hcat([collect(1:n_atoms) for i in 1:n_atoms]...)
    inds_j = permutedims(inds_i, (2, 1))

    table_d0_units = T.(lookup_table(data_d0, radii))
    table_m0_units = T.(lookup_table(data_m0, radii))
    if units
        table_d0 = table_d0_units
        table_m0 = table_m0_units
    else
        table_d0 = ustrip.(table_d0_units)
        table_m0 = ustrip.(table_m0_units)
    end

    coulomb_const = units ? coulombconst : ustrip(coulombconst)
    if !iszero(solute_dielectric)
        factor_solute = -T(coulomb_const) / T(solute_dielectric)
    else
        factor_solute = zero(T(coulomb_const))
    end
    if !iszero(solvent_dielectric)
        factor_solvent = T(coulomb_const) / T(solvent_dielectric)
    else
        factor_solvent = zero(T(coulomb_const))
    end

    if isa(atoms, CuArray)
        or = CuArray(offset_radii) # cu converts to Float32
        sor = CuArray(scaled_offset_radii)
        is, js = cu(inds_i), cu(inds_j)
        d0s, m0s = CuArray(table_d0), CuArray(table_m0)
        αs, βs, γs = CuArray(αs_cpu), CuArray(βs_cpu), CuArray(γs_cpu)
    else
        or = offset_radii
        sor = scaled_offset_radii
        is, js = inds_i, inds_j
        d0s, m0s = table_d0, table_m0
        αs, βs, γs = αs_cpu, βs_cpu, γs_cpu
    end

    if units
        return ImplicitSolventGBN2{T, D, typeof(αs), typeof(or), typeof(T(kappa)), typeof(T(sa_factor)),
                        typeof(factor_solute), typeof(is), typeof(d0s), typeof(m0s)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(kappa), offset, dist_cutoff,
                    use_ACE, αs, βs, γs, probe_radius, T(sa_factor), factor_solute,
                    factor_solvent, is, js, d0s, m0s, neck_scale, neck_cut)
    else
        return ImplicitSolventGBN2{T, T, typeof(αs), typeof(or), typeof(T(ustrip(kappa))), T, T,
                        typeof(is), typeof(d0s), typeof(m0s)}(
                    or, sor, solvent_dielectric, solute_dielectric, T(ustrip(kappa)), ustrip(offset),
                    ustrip(dist_cutoff), use_ACE, αs, βs, γs, ustrip(probe_radius), ustrip(sa_factor),
                    factor_solute, factor_solvent, is, js, d0s, m0s, neck_scale, ustrip(neck_cut))
    end
end

function born_radii_loop_OBC(coord_i, coord_j, ori, srj, dist_cutoff, boundary)
    I = zero(coord_i[1] / unit(dist_cutoff)^2)
    r = norm(vector(coord_i, coord_j, boundary))
    if iszero(r) || (!iszero(dist_cutoff) && r > dist_cutoff)
        return I
    end
    U = r + srj
    if ori < U
        D_ij = abs(r - srj)
        L = max(ori, D_ij)
        I += (1/L - 1/U + (r - (srj^2)/r)*(1/(U^2) - 1/(L^2))/4 + log(L/U)/(2*r)) / 2
        if ori < (srj - r)
            I += 2 * (1/ori - 1/L)
        end
    end
    return I
end

"""
    born_radii_and_grad(inter, coords, boundary)

Calculate Born radii and gradients of Born radii and surface area overlap
with respect to atomic distance.
Custom GBSA methods should implement this function.
"""
function born_radii_and_grad(inter::ImplicitSolventOBC, coords, boundary)
    coords_i = @view coords[inter.is]
    coords_j = @view coords[inter.js]
    oris = @view inter.offset_radii[inter.is]
    srjs = @view inter.scaled_offset_radii[inter.js]
    loop_res = born_radii_loop_OBC.(coords_i, coords_j, oris, srjs,
                                    inter.dist_cutoff, (boundary,))
    Is = dropdims(sum(loop_res; dims=2); dims=2)
    I_grads = zero(loop_res) ./ unit(inter.dist_cutoff)

    ori = inter.offset_radii
    radii = ori .+ inter.offset
    ψs = Is .* ori
    ψs2 = ψs .^ 2
    α, β, γ = inter.α, inter.β, inter.γ
    tanh_sums = tanh.(α .* ψs .- β .* ψs2 .+ γ .* ψs2 .* ψs)
    Bs = 1 ./ (1 ./ ori .- tanh_sums ./ radii)
    grad_terms = ori .* (α .- 2 .* β .* ψs .+ 3 .* γ .* ψs2)
    B_grads = (1 .- tanh_sums .^ 2) .* grad_terms ./ radii

    return Bs, B_grads, I_grads
end

struct BornRadiiGBN2LoopResult{I, G}
    I::I
    I_grad::G
end

get_I(r::BornRadiiGBN2LoopResult) = r.I
get_I_grad(r::BornRadiiGBN2LoopResult) = r.I_grad

function born_radii_loop_GBN2(coord_i::SVector{D, T}, coord_j, ori, orj, srj, dist_cutoff,
                                offset, neck_scale, neck_cut, d0, m0, boundary) where {D, T}
    I = zero(coord_i[1] / unit(dist_cutoff)^2)
    I_grad = zero(coord_i[1] / unit(dist_cutoff)^3)
    r = norm(vector(coord_i, coord_j, boundary))
    if iszero(r) || (!iszero(dist_cutoff) && r > dist_cutoff)
        return BornRadiiGBN2LoopResult(I, I_grad)
    end
    U = r + srj
    if ori < U
        D_ij = abs(r - srj)
        L = max(ori, D_ij)
        I += (1/L - 1/U + (r - (srj^2)/r)*(1/(U^2) - 1/(L^2))/4 + log(L/U)/(2*r)) / 2
        if ori < (srj - r)
            I += 2 * (1/ori - 1/L)
        end
    end
    radius_i = ori + offset
    radius_j = orj + offset
    if r < (radius_i + radius_j + neck_cut)
        if dimension(T) == u"𝐋"
            r_d0_strip = 10 * ustrip(u"nm", r - d0) # The integral uses Å
        else
            r_d0_strip = 10 * (r - d0)
        end
        denom = 1 + r_d0_strip^2 + 3 * r_d0_strip^6 / 10
        I += neck_scale * m0 / denom
        numer = 2 * r_d0_strip + 9 * r_d0_strip^5 / 5
        I_grad -= 10 * neck_scale * m0 * numer / (denom^2 * unit(dist_cutoff))
    end
    return BornRadiiGBN2LoopResult(I, I_grad)
end

function born_radii_and_grad(inter::ImplicitSolventGBN2, coords, boundary)
    coords_i = @view coords[inter.is]
    coords_j = @view coords[inter.js]
    oris = @view inter.offset_radii[inter.is]
    orjs = @view inter.offset_radii[inter.js]
    srjs = @view inter.scaled_offset_radii[inter.js]
    loop_res = born_radii_loop_GBN2.(coords_i, coords_j, oris, orjs, srjs,
                    inter.dist_cutoff, inter.offset, inter.neck_scale,
                    inter.neck_cut, inter.d0s, inter.m0s, (boundary,))
    Is = dropdims(sum(get_I.(loop_res); dims=2); dims=2)
    I_grads = get_I_grad.(loop_res)

    ori = inter.offset_radii
    radii = ori .+ inter.offset
    ψs = Is .* ori
    ψs2 = ψs .^ 2
    αs, βs, γs = inter.αs, inter.βs, inter.γs
    tanh_sums = tanh.(αs .* ψs .- βs .* ψs2 .+ γs .* ψs2 .* ψs)
    Bs = 1 ./ (1 ./ ori .- tanh_sums ./ radii)
    grad_terms = ori .* (αs .- 2 .* βs .* ψs .+ 3 .* γs .* ψs2)
    B_grads = (1 .- tanh_sums .^ 2) .* grad_terms ./ radii

    return Bs, B_grads, I_grads
end

# Store the results of the ij broadcasts during force calculation
struct ForceLoopResult1{T, V}
    bi::T
    bj::T
    fi::V
    fj::V
end

get_bi(r::ForceLoopResult1) = r.bi
get_bj(r::ForceLoopResult1) = r.bj

struct ForceLoopResult2{V}
    fi::V
    fj::V
end

get_fi(r::Union{ForceLoopResult1, ForceLoopResult2}) = r.fi
get_fj(r::Union{ForceLoopResult1, ForceLoopResult2}) = r.fj

function gb_force_loop_1(coord_i, coord_j, i, j, charge_i, charge_j, Bi, Bj, dist_cutoff,
                            factor_solute, factor_solvent, kappa, boundary)
    if j < i
        zero_force = zero(factor_solute ./ coord_i .^ 2)
        return ForceLoopResult1(zero_force[1], zero_force[1], zero_force, zero_force)
    end
    dr = vector(coord_i, coord_j, boundary)
    r2 = sum(abs2, dr)
    if !iszero(dist_cutoff) && r2 > dist_cutoff^2
        zero_force = zero(factor_solute ./ coord_i .^ 2)
        return ForceLoopResult1(zero_force[1], zero_force[1], zero_force, zero_force)
    end
    alpha2_ij = Bi * Bj
    D = r2 / (4 * alpha2_ij)
    exp_term = exp(-D)
    denominator2 = r2 + alpha2_ij * exp_term
    denominator = sqrt(denominator2)
    if iszero(kappa)
        pre_factor = factor_solute + factor_solvent
    else
        pre_factor = factor_solute + exp(-kappa * denominator) * factor_solvent +
                        kappa * denominator * exp(-kappa * denominator) * factor_solvent
    end
    Gpol = (pre_factor * charge_i * charge_j) / denominator
    dGpol_dr = -Gpol * (1 - exp_term/4) / denominator2
    dGpol_dalpha2_ij = -Gpol * exp_term * (1 + D) / (2 * denominator2)
    change_born_force_i = dGpol_dalpha2_ij * Bj
    if i != j
        change_born_force_j = dGpol_dalpha2_ij * Bi
        fdr = dr * dGpol_dr
        change_fs_i =  fdr
        change_fs_j = -fdr
        return ForceLoopResult1(change_born_force_i, change_born_force_j,
                                change_fs_i, change_fs_j)
    else
        zero_force = zero(factor_solute ./ coord_i .^ 2)
        return ForceLoopResult1(change_born_force_i, zero_force[1], zero_force, zero_force)
    end
end

function gb_force_loop_2(coord_i, coord_j, bi, ig, ori, srj, dist_cutoff, boundary)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    if iszero(r) || (!iszero(dist_cutoff) && r > dist_cutoff)
        zero_force = zero(bi ./ coord_i .^ 2)
        return ForceLoopResult2(zero_force, zero_force)
    end
    rsrj = r + srj
    if ori < rsrj
        D = abs(r - srj)
        L = inv(max(ori, D))
        U = inv(rsrj)
        rinv = inv(r)
        r2inv = rinv^2
        t3 = (1 + (srj^2)*r2inv)*(L^2 - U^2)/8 + log(U/L)*r2inv/4
        de = bi * (t3 - ig) * rinv
        fdr = dr * de
        return ForceLoopResult2(-fdr, fdr)
    else
        zero_force = zero(bi ./ coord_i .^ 2)
        return ForceLoopResult2(zero_force, zero_force)
    end
end

function forces(inter::AbstractGBSA, sys, neighbors=nothing)
    coords, atoms, boundary = sys.coords, sys.atoms, sys.boundary
    Bs, B_grads, I_grads = born_radii_and_grad(inter, coords, boundary)

    if inter.use_ACE
        radii = inter.offset_radii .+ inter.offset
        sa_terms = inter.sa_factor .* (radii .+ inter.probe_radius) .^ 2 .* (radii ./ Bs) .^ 6
        born_forces = (-6 .* sa_terms ./ Bs) .* (Bs .> zero(inter.offset))
    else
        born_forces = zeros(typeof(inter.sa_factor * inter.dist_cutoff), length(sys))
    end

    coords_i = @view coords[inter.is]
    coords_j = @view coords[inter.js]
    charges = charge.(atoms)
    charges_i = @view charges[inter.is]
    charges_j = @view charges[inter.js]
    Bsi = @view Bs[inter.is]
    Bsj = @view Bs[inter.js]
    loop_res_1 = gb_force_loop_1.(coords_i, coords_j, inter.is, inter.js, charges_i, charges_j,
                                    Bsi, Bsj, inter.dist_cutoff, inter.factor_solute,
                                    inter.factor_solvent, inter.kappa, (boundary,))
    born_forces = born_forces .+ dropdims(sum(get_bi.(loop_res_1); dims=2); dims=2)
    born_forces = born_forces .+ dropdims(sum(get_bj.(loop_res_1); dims=1); dims=1)
    fs = dropdims(sum(get_fi.(loop_res_1); dims=2); dims=2) .+ dropdims(sum(get_fj.(loop_res_1); dims=1); dims=1)

    born_forces_2 = born_forces .* (Bs .^ 2) .* B_grads

    bis = @view born_forces_2[inter.is]
    oris = @view inter.offset_radii[inter.is]
    srjs = @view inter.scaled_offset_radii[inter.js]
    loop_res_2 = gb_force_loop_2.(coords_i, coords_j, bis, I_grads, oris, srjs,
                                    inter.dist_cutoff, (boundary,))

    return fs .+ dropdims(sum(get_fi.(loop_res_2); dims=2); dims=2) .+ dropdims(sum(get_fj.(loop_res_2); dims=1); dims=1)
end

function gb_energy_loop(coord_i, coord_j, i, j, charge_i, charge_j, Bi, Bj, ori,
                        dist_cutoff, factor_solute, factor_solvent, kappa, offset,
                        probe_radius, sa_factor, use_ACE, boundary)
    if i == j
        if iszero(kappa)
            pre_factor = factor_solute + factor_solvent
        else
            pre_factor = factor_solute + exp(-kappa * Bi) * factor_solvent
        end
        E = pre_factor * (charge_i^2) / (2*Bi)
        if use_ACE && (Bi > zero(offset))
            radius_i = ori + offset
            E += sa_factor * (radius_i + probe_radius)^2 * (radius_i / Bi)^6
        end
        return E
    elseif j > i
        r2 = sum(abs2, vector(coord_i, coord_j, boundary))
        if !iszero(dist_cutoff) && r2 > dist_cutoff^2
            return zero(factor_solute / offset)
        end
        f = sqrt(r2 + Bi*Bj*exp(-r2/(4*Bi*Bj)))
        if iszero(dist_cutoff)
            f_cutoff = 1/f
        else
            f_cutoff = (1/f - 1/dist_cutoff)
        end
        if iszero(kappa)
            pre_factor = factor_solute + factor_solvent
        else
            pre_factor = factor_solute + exp(-kappa * f) * factor_solvent
        end
        return pre_factor * charge_i * charge_j * f_cutoff
    else
        return zero(factor_solute / offset)
    end
end

function potential_energy(inter::AbstractGBSA, sys, neighbors=nothing)
    coords, atoms, boundary = sys.coords, sys.atoms, sys.boundary
    Bs, B_grads, I_grads = born_radii_and_grad(inter, coords, boundary)

    coords_i = @view coords[inter.is]
    coords_j = @view coords[inter.js]
    charges = charge.(atoms)
    charges_i = @view charges[inter.is]
    charges_j = @view charges[inter.js]
    Bsi = @view Bs[inter.is]
    Bsj = @view Bs[inter.js]
    oris = @view inter.offset_radii[inter.is]
    return sum(gb_energy_loop.(coords_i, coords_j, inter.is, inter.js, charges_i, charges_j,
                                Bsi, Bsj, oris, inter.dist_cutoff, inter.factor_solute,
                                inter.factor_solvent, inter.kappa, inter.offset,
                                inter.probe_radius, inter.sa_factor, inter.use_ACE,
                                (boundary,)))
end
