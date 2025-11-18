using LinearAlgebra
using Base.Threads: @threads
using LinearAlgebra: eigen


# Define the Hamiltonian
function U(q1, q2)
    return U_int_value * (1+ cos(q1) + cos(q2))
end

