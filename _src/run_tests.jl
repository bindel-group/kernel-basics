using LinearAlgebra
using StatsFuns
using SpecialFunctions
using Optim
using Test

include("testing.jl")
include("ext_la.jl")
include("sample.jl")
include("kfuns.jl")
include("kmats.jl")
include("gpp.jl")
include("hypers.jl")
include("acquisition.jl")
include("bo_step.jl")

include("test_ext_la.jl")
include("test_kfuns.jl")
include("test_kmats.jl")
include("test_gpp.jl")
include("test_hypers.jl")
include("test_acquisition.jl")
