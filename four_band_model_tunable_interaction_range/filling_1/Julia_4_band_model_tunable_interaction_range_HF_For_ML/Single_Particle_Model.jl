using LinearAlgebra
using Base.Threads: @threads
using LinearAlgebra: eigen

# Define Gamma matrices
Gamma_1 = PM(1, 1)
Gamma_2 = PM(1, 2)
Gamma_3 = PM(1, 3)
Gamma_4 = PM(2, 0)
Gamma_5 = PM(3, 0)

# Define the Hamiltonian
function h(k1, k2)
    term5 = (1 + cos(k1) + cos(k2)) * Gamma_5
    term1 = sin(k1) * Gamma_1
    term2 = sin(k2) * Gamma_2
    term3 = sin(2*k1) * Gamma_3
    term4 = sin(2*k2) * Gamma_4
    return term1 + term2 + term3 + term4 + term5
end