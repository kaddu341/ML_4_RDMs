import Pkg; Pkg.add("StatsBase");
using DelimitedFiles
using StatsBase

runfolder = ARGS[1]

@time include( runfolder * "/" * "Parameters.jl" )
println("Parameters.jl included.");
println(" ");
println(" ");
println(" ");
println(" ");
flush(stdout);

@time include( Basic_Codes * "/" * "Generic_Functions.jl")
println("Generic_Functions.jl included.");
println(" ");
println(" ");
println(" ");
println(" ");
flush(stdout);

@time include( Basic_Codes * "/" * "Single_Particle_Model.jl")
println("Single_Particle_Model.jl included.");
println(" ");
println(" ");
println(" ");
println(" ");
flush(stdout);

@time include( Basic_Codes * "/" * "Interaction.jl")
println("Interaction.jl included.");
println(" ");
println(" ");
println(" ");
println(" ");
flush(stdout);

@time include( Basic_Codes * "/" * "HF_Calculation.jl")
println("HF_Calculation.jl included.");
println(" ");
println(" ");
println(" ");
println(" ");
flush(stdout);


k_numbers_in_one_direction = Int(round(2π/dk))+1;

h_SP_Mesh = zeros(ComplexF64,k_numbers_in_one_direction-1,k_numbers_in_one_direction-1,4,4);
for i_k1 in 1:1:(k_numbers_in_one_direction-1)
    for i_k2 in 1:1:(k_numbers_in_one_direction-1)
        k1 = (i_k1-1)*dk
        k2 = (i_k2-1)*dk
        h_SP_Mesh[i_k1,i_k2,:,:] = h(k1, k2)
    end
end

U_int_mesh = zeros(ComplexF64,k_numbers_in_one_direction-1,k_numbers_in_one_direction-1);
for i_k1 in 1:1:(k_numbers_in_one_direction-1)
    for i_k2 in 1:1:(k_numbers_in_one_direction-1)
        k1 = (i_k1-1)*dk
        k2 = (i_k2-1)*dk
        U_int_mesh[i_k1,i_k2] = U(k1, k2)
    end
end


initial_filename = "P_HF_initial.h5"
result_filename = "P_HF_result.h5"
ML_filename = "P_HF_ML.h5"
wb = 1/(2*dk^2);

for initial_index in 1:1:number_of_random_initials

    println("Start: initial_index = ", initial_index)
    
    if check_existence_of_section_h5(ML_filename, "/P_ML_"*string(initial_index))
        P_inital_prepare = h5read(ML_filename , "P_ML_"*string(initial_index));
        P_inital = P_inital_prepare[:,:,:,:,1] .+ im .* P_inital_prepare[:,:,:,:,2];
    else
        P_inital =  zeros(ComplexF64,k_numbers_in_one_direction-1,k_numbers_in_one_direction-1,4,4);
        for i_k1 in 1:1:(k_numbers_in_one_direction-1)
            for i_k2 in 1:1:(k_numbers_in_one_direction-1)
                temp = rand(4, 4) + 1im .* rand(4, 4);
                P_inital[i_k1,i_k2,:,:] = temp + temp';
            end
        end
        h5write(initial_filename, "/P_initial_"* string(initial_index), P_inital) 
    end


    P_HF_result=HF_iteraction(P_inital,k_numbers_in_one_direction,dk,U_int_mesh,h_SP_Mesh,filling);
    h5write(result_filename, "/P_HF_result_"*string(initial_index), P_HF_result)

    println("End: initial_index = ", initial_index)
    println(" ");
    println(" ");
    println(" ");
    println(" ");
    flush(stdout);
end



