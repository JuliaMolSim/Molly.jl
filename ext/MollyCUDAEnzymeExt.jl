module MollyCUDAEnzymeExt

using Molly
using CUDA
using Enzyme

ext = Base.get_extension(Molly,:MollyCUDAExt)

EnzymeRules.inactive(::typeof(ext.cuda_threads_blocks_pairwise), args...) = nothing
EnzymeRules.inactive(::typeof(ext.cuda_threads_blocks_specific), args...) = nothing


end
