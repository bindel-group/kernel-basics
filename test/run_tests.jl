using LinearAlgebra
using StatsFuns
using SpecialFunctions
using Optim
using Test

include("../src/testing.jl")
include("../src/ext_la.jl")
include("../src/sample.jl")
include("../src/kfuns.jl")
include("../src/kmats.jl")
include("../src/gpp.jl")
include("../src/hypers.jl")
include("../src/acquisition.jl")
include("../src/bo_step.jl")

include("test_ext_la.jl")
include("test_kfuns.jl")
include("test_kmats.jl")
include("test_gpp.jl")
include("test_hypers.jl")
include("test_acquisition.jl")
