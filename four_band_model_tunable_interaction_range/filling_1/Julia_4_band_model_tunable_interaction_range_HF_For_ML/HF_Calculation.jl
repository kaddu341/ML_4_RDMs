using LinearAlgebra
using Base.Threads: @threads
using LinearAlgebra: eigen

function Generate_Hartree_Term(U_int_mesh,OP_Mesh,k_numbers_in_one_direction)
    Hartree_Term = zeros(ComplexF64,k_numbers_in_one_direction-1,k_numbers_in_one_direction-1,4,4);
    OP_ave= 0 .* I(4)
    for i_k1 in 1:1:(k_numbers_in_one_direction-1)
        for i_k2 in 1:1:(k_numbers_in_one_direction-1)
            OP_ave = OP_ave .+ OP_Mesh[i_k1,i_k2,:,:] ./ (k_numbers_in_one_direction-1) ./ (k_numbers_in_one_direction-1);
        end
    end
    for i_k1 in 1:1:(k_numbers_in_one_direction-1)
        for i_k2 in 1:1:(k_numbers_in_one_direction-1)
            Hartree_Term[i_k1,i_k2,:,:] = U_int_mesh[1,1] * tr(OP_ave) .* I(4);
        end
    end
    return Hartree_Term
end

function Generate_Fock_Term(U_int_mesh,OP_Mesh,k_numbers_in_one_direction)
    Fock_Term = zeros(ComplexF64,k_numbers_in_one_direction-1,k_numbers_in_one_direction-1,4,4);
    @threads for i_k1 in 1:1:(k_numbers_in_one_direction-1)
        for i_k2 in 1:1:(k_numbers_in_one_direction-1)
            for i_q1 in 1:1:(k_numbers_in_one_direction-1)
                for i_q2 in 1:1:(k_numbers_in_one_direction-1)
                    Fock_Term[i_k1,i_k2,:,:] = Fock_Term[i_k1,i_k2,:,:] .- U_int_mesh[i_q1,i_q2].* transpose(OP_Mesh[ mod( (i_k1-1)+(i_q1-1) , k_numbers_in_one_direction-1)+1 ,mod( (i_k2-1)+(i_q2-1) , k_numbers_in_one_direction-1)+1 ,:,:]) ./ (k_numbers_in_one_direction-1) ./ (k_numbers_in_one_direction-1);
                end
            end
        end
    end
    return Fock_Term
end

function E_total_per_unit_cell(h_HF_Mesh, h_SP_Mesh ,OP_Mesh)
    L = size(h_HF_Mesh,1)
    E_tot = 0
    for i_k1 in 1:1:(k_numbers_in_one_direction-1)
        for i_k2 in 1:1:(k_numbers_in_one_direction-1)
            E_tot = E_tot + tr( (h_HF_Mesh[i_k1,i_k2,:,:] .+ h_SP_Mesh[i_k1,i_k2,:,:] )  * transpose(OP_Mesh[i_k1,i_k2,:,:]) ) / 2;
        end
    end
    return E_tot / L^2
end

function HF_iteraction(P_inital,k_numbers_in_one_direction,dk,U_int_mesh, h_SP_Mesh, filling)
    error_criterion = 1e-6

    
    OP_Mesh = deepcopy(P_inital);
    OP_Mesh_Next = deepcopy(P_inital);
    h_HF_Mesh = deepcopy(P_inital);
    E_tot_per_unit_cell_next = 1e5
    error = 1e5

    iteration = 0
    while error > error_criterion 

        iteration += 1
        
        E_tot_per_unit_cell = E_tot_per_unit_cell_next
        OP_Mesh = copy(OP_Mesh_Next)

        h_HF_Mesh = h_SP_Mesh .+ Generate_Hartree_Term(U_int_mesh,OP_Mesh,k_numbers_in_one_direction) .+ Generate_Fock_Term(U_int_mesh,OP_Mesh,k_numbers_in_one_direction);

        @threads for i_k1 in 1:1:(k_numbers_in_one_direction-1)
            for i_k2 in 1:1:(k_numbers_in_one_direction-1)
                htemp = h_HF_Mesh[i_k1,i_k2,:,:]
                eigsys = eigen(htemp)
                # Extract eigenvalues and eigenvectors
                eigvals = eigsys.values 
                eigvecs = eigsys.vectors
                # Sort eigenvalues and rearrange eigenvectors accordingly
                sorted_indices = sortperm(real(eigvals))  # Get sorting indices
                sorted_eigvals = eigvals[sorted_indices]  # Apply sorting to eigenvectors
                sorted_eigvecs = eigvecs[:, sorted_indices]  # Apply sorting to eigenvectors
                OP_Mesh_Next[i_k1,i_k2,:,:] = conj.(sorted_eigvecs[:, 1:filling] * (sorted_eigvecs[:, 1:filling]'))
            end
        end

        E_tot_per_unit_cell_next = E_total_per_unit_cell(h_HF_Mesh, h_SP_Mesh ,OP_Mesh_Next);

        error = abs(E_tot_per_unit_cell_next - E_tot_per_unit_cell);
        
        mse = sum( abs.( OP_Mesh_Next .- OP_Mesh ).^2 )/(k_numbers_in_one_direction-1)/(k_numbers_in_one_direction-1)/4/4;
        #error = mse;

        #println(maximum(abs.( OP_Mesh_Next .- OP_Mesh )))

        #println(mse)

        println("Iteration: ", iteration, " Error: ", error, " MSE: ", mse, " Energy: ", E_tot_per_unit_cell_next)

    end

    return conj.(OP_Mesh_Next)
end
