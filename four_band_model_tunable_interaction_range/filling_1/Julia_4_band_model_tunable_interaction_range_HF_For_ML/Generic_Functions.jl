import Pkg; Pkg.add("HDF5");
using HDF5

function check_existence_of_section_h5(h5file_name, section_name)
    if isfile(h5file_name)
        println("File $h5file_name exists.");
        h5file = h5open(h5file_name, "r");
        if haskey(h5file, section_name)
            println("File $h5file_name has the section $section_name.");
            close(h5file);
            return true;
        else
            println("File $h5file_name does not have the section $section_name.");
            close(h5file);
            return false;
        end
    else
        println("File $h5file_name does not exist.");
        return false;
    end
end


function rot2d(theta) 
    return [cos(theta) -sin(theta); sin(theta) cos(theta)]
end

PauliMatrix = [ [1 0 ; 0 1] , [0 1 ; 1 0] , [0 -im ; im 0] , [1 0 ; 0 -1] ];

# Define Kronecker product for matrices
function PM(X...)
    reduce(kron, [PauliMatrix[x+1] for x in X])
end

function h5read_complex_array(h5file_name, data_section)
    data_prepare = h5read(h5file_name,data_section);
    result = zeros(ComplexF64,size(data_prepare));
    for i in 1:size(data_prepare,1)
        for j in 1:size(data_prepare,2)
            result[i,j] = data_prepare[i,j][1] + im * data_prepare[i,j][2];
        end
    end
    return result;
end

function read_text_file(file_path::AbstractString)
    # Check if file exists
    if isfile(file_path)
        # Open the file in read mode
        open(file_path, "r") do file
            # Read all lines from the file
            lines = readlines(file)
            return lines
        end
    else
        println("File not found.")
        return nothing
    end
end