#include "compute/compute_dump.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

dump::dump(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Compute(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string dump::getName() 
{
    return "dump";
}

/* ----------------------------------------------------------------------------------------------------------- */

void dump::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal compute_dump command";
    std::string format   = "compute ID   groupID dump frequency filename";
    std::string example0 = "compute cdump liquid  dump 1 dump.txt";

    if (params.size() != 2) {
        print_error_and_exit("Invalid compute_dump parameters", error, format, {example0});
    }

    frequency = parse_int<unsigned int>(params[0], "Invalid compute_dump parameters", "frequency", {format,example0});
    filename  = params[1];
}

/* ----------------------------------------------------------------------------------------------------------- */

void dump::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file dump 

    file = fopen(filename.c_str(), "w");

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void dump::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file dump 

    fclose(file);
    file = nullptr; 

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void dump::compute(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    if (step % frequency != 0) {
        return;
    }
    
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    Box& box     = system.box;
    int N        = system.n_atoms;
    int n_gatoms = group.n_atoms;
    int n_types  = group.h_types.size();

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemcpy(atoms.h_id.data(),     atoms.d_id,    sizeof(int)    * N,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(atoms.h_type.data(),   atoms.d_type,  sizeof(int)    * N,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(atoms.h_pos.data(),    atoms.d_pos,   sizeof(numtyp) * N * 3, cudaMemcpyDeviceToHost));

    /* ------------------------------------------------------- */

    fprintf(file, "ITEM: TIMESTEP\n%d\n", step);
    fprintf(file, "ITEM: NUMBER OF ATOMS\n%d\n", n_gatoms);
    fprintf(file, "ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(file, "%.9f %.9f\n", box.xlo, box.xhi);
    fprintf(file, "%.9f %.9f\n", box.ylo, box.yhi);
    fprintf(file, "%.9f %.9f\n", box.zlo, box.zhi);

    if (system.atom_style == 0) 
    {
        fprintf(file, "ITEM: ATOMS id type x y z\n");
        for (int i = 0; i < system.n_atoms; ++i) 
        {
            for (int t = 0; t < n_types; ++t)  
            {
                int g_type = group.h_types[t];
                if (atoms.h_type[i] == g_type)
                {
                    fprintf(file, "%d %d %.9f %.9f %.9f\n", 
                            atoms.h_id[i], 
                            atoms.h_type[i],
                            atoms.h_pos[i*3+0], atoms.h_pos[i*3+1], atoms.h_pos[i*3+2]);
                }
            }
        }
    }
    else 
    {
        fprintf(file, "ITEM: ATOMS id mol type x y z\n");
        for (int i = 0; i < system.n_atoms; ++i) 
        {
            for (int t = 0; t < n_types; ++t)  
            {
                int g_type = group.h_types[t];
                if (atoms.h_type[i] == g_type)
                {
                    fprintf(file, "%d %d %d %.9f %.9f %.9f\n", 
                            atoms.h_id[i], 
                            atoms.h_mol_id[i], 
                            atoms.h_type[i],
                            atoms.h_pos[i*3+0], atoms.h_pos[i*3+1], atoms.h_pos[i*3+2]);
                }
            }
        }
    }

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_COMPUTE(dump)
///////////////////////////////
