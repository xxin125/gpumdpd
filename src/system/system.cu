#include "system.cuh"

/* -------------------------------------------------------------------------------------- */

void atoms_mem_alloc(System& system)
{
    /* ------------------------------------------------------- */

    int N          = system.n_atoms;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    atoms.h_id.resize(N, 0);
    atoms.h_type.resize(N, 0);
    atoms.h_pos.resize(N * 3, 0.0);
    atoms.h_vel.resize(N * 3, 0.0);
    atoms.h_uwpos.resize(N * 3, 0.0);
    atoms.h_force.resize(N * 3, 0.0);
    atoms.h_pe.resize(N, 0.0);

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMalloc(&atoms.d_id,      N   *   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&atoms.d_type,    N   *   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&atoms.d_pos,     N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&atoms.d_vel,     N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&atoms.d_uwpos,   N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&atoms.d_force,   N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&atoms.d_pe,      N   *   sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&atoms.d_viral,   N * 6 * sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&atoms.d_rho,     N   *   sizeof(numtyp)));

    CUDA_CHECK(cudaMemset(atoms.d_id,    0, N   *   sizeof(int)));
    CUDA_CHECK(cudaMemset(atoms.d_type,  0, N   *   sizeof(int)));
    CUDA_CHECK(cudaMemset(atoms.d_pos,   0, N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(atoms.d_vel,   0, N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(atoms.d_uwpos, 0, N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(atoms.d_force, 0, N * 3 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(atoms.d_pe,    0, N   *   sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(atoms.d_viral, 0, N * 6 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(atoms.d_rho,   0, N   *   sizeof(numtyp)));

    /* ------------------------------------------------------- */

    std::vector<numtyp> h_masses(256, 0.0); 
    std::vector<numtyp> h_pair_coeff(256, 0.0);
    std::vector<numtyp> h_bond_coeff(256, 0.0);
    std::vector<numtyp> h_angle_coeff(256, 0.0);

    CUDA_CHECK(cudaMemcpyToSymbol(masses,          h_masses.data(),      256 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_pair_coeff,  h_pair_coeff.data(),  256 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_bond_coeff,  h_bond_coeff.data(),  256 * sizeof(numtyp)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_angle_coeff, h_angle_coeff.data(), 256 * sizeof(numtyp)));

    h_masses.clear();
    h_pair_coeff.clear();
    h_bond_coeff.clear();
    h_angle_coeff.clear();

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void atoms_mem_copy_2_gpu(System& system, std::vector<numtyp> data_masses)
{
    /* ------------------------------------------------------- */

    int N          = system.n_atoms;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemcpy(atoms.d_id,    atoms.h_id.data(),    sizeof(int)    * N,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(atoms.d_type,  atoms.h_type.data(),  sizeof(int)    * N,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(atoms.d_pos,   atoms.h_pos.data(),   sizeof(numtyp) * N * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(atoms.d_vel,   atoms.h_vel.data(),   sizeof(numtyp) * N * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(atoms.d_uwpos, atoms.h_uwpos.data(), sizeof(numtyp) * N * 3, cudaMemcpyHostToDevice));

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemcpyToSymbol(masses, data_masses.data(), data_masses.size() * sizeof(numtyp)));

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void atoms_mem_free(System& system)
{
    /* ------------------------------------------------------- */

    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    atoms.h_id.clear();
    atoms.h_type.clear();
    atoms.h_pos.clear();
    atoms.h_vel.clear();
    atoms.h_uwpos.clear();
    atoms.h_force.clear();
    atoms.h_pe.clear();

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaFree(atoms.d_id));
    CUDA_CHECK(cudaFree(atoms.d_type));
    CUDA_CHECK(cudaFree(atoms.d_pos));
    CUDA_CHECK(cudaFree(atoms.d_vel));
    CUDA_CHECK(cudaFree(atoms.d_uwpos));
    CUDA_CHECK(cudaFree(atoms.d_force));
    CUDA_CHECK(cudaFree(atoms.d_pe));
    CUDA_CHECK(cudaFree(atoms.d_viral));
    CUDA_CHECK(cudaFree(atoms.d_rho));

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void bonds_mem_alloc(System& system)
{
    /* ------------------------------------------------------- */

    int N          = system.n_bonds;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMalloc(&atoms.d_bondlist,  N * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&atoms.d_bond_pe,   N   *   sizeof(numtyp))); 

    CUDA_CHECK(cudaMemset(atoms.d_bondlist,  -1,  N * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemset(atoms.d_bond_pe,    0,  N   *   sizeof(numtyp)));
    
    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void bonds_mem_copy_2_gpu(System& system)
{
    /* ------------------------------------------------------- */

    int N          = system.n_bonds;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemcpy(atoms.d_bondlist,   atoms.h_bondlist.data(),  sizeof(int) * N * 3, cudaMemcpyHostToDevice));

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void bonds_mem_free(System& system)
{
    /* ------------------------------------------------------- */

    Atoms& atoms = system.atoms;

    /* ------------------------------------------------------- */

    atoms.h_mol_id.clear();
    atoms.h_bondlist.clear();

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaFree(atoms.d_mol_id));
    CUDA_CHECK(cudaFree(atoms.d_bondlist));
    CUDA_CHECK(cudaFree(atoms.d_bond_pe));

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void angles_mem_alloc(System& system)
{
    /* ------------------------------------------------------- */

    int N          = system.n_angles;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMalloc(&atoms.d_anglelist,  N * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&atoms.d_angle_pe,   N   *   sizeof(numtyp))); 

    CUDA_CHECK(cudaMemset(atoms.d_anglelist,  -1,  N * 4 * sizeof(int)));
    CUDA_CHECK(cudaMemset(atoms.d_angle_pe,    0,  N   *   sizeof(numtyp)));
    
    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void angles_mem_copy_2_gpu(System& system)
{
    /* ------------------------------------------------------- */

    int N          = system.n_angles;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemcpy(atoms.d_anglelist,   atoms.h_anglelist.data(),  sizeof(int) * N * 4, cudaMemcpyHostToDevice));

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void angles_mem_free(System& system)
{
    /* ------------------------------------------------------- */

    Atoms& atoms = system.atoms;

    /* ------------------------------------------------------- */

    atoms.h_anglelist.clear();

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaFree(atoms.d_anglelist));
    CUDA_CHECK(cudaFree(atoms.d_angle_pe));

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void create_groups(System& system)
{
    std::stringstream ss(system.input);
    std::string line;
    std::vector<std::string> args;
    std::string arg;

    while (std::getline(ss, line)) 
    {
        std::istringstream iss(line);
        args.clear();
        std::string key;
        iss >> key;

        /* ------------------------------------------------------- */

        if (key != "group") continue;

        while (iss >> arg) {
            args.push_back(arg);
        }

        std::string error    = "illegal group command";
        std::string format   = "group group_name type type1 type2 ...";
        std::string example0 = "group liquid type 1";
        std::string example1 = "group solid type 1 2";

        if (args.size() < 3) {
            print_error_and_exit(line, error, format, {example0, example1});
        }

        std::string group_name = args[0];

        if (system.groups.find(group_name) != system.groups.end()) 
        {
            print_error({"Group " + group_name + " already exists"});
            continue;
        }

        if (args[1] != "type") {
            print_error_and_exit(line, error, format, {example0, example1});
        }

        /* ------------------------------------------------------- */

        Group new_group;
        new_group.name = group_name;

        std::set<int> type_set;
        for (size_t arg_i = 2; arg_i < args.size(); arg_i++) 
        {
            int type_i = parse_int<int>(args[arg_i], line, "type id", {format, example0, example1});
            if (type_i > system.n_atomtypes) {
                print_error({"Type in group is not included in the data file"});
            } else if (type_set.find(type_i) != type_set.end()) {
                std::string error1 = "Duplicate type found: " + std::to_string(type_i);
                print_error_and_exit(line, error1, format, {example0, example1});
            } else {
                new_group.h_types.push_back(type_i);
                type_set.insert(type_i);
            }
        }

        system.groups[group_name] = new_group;
        group_mem_alloc(system.groups[group_name], system.n_atoms);
        group_select(system, system.groups[group_name]);

        /* ------------------------------------------------------- */
    }
}

/* -------------------------------------------------------------------------------------- */

void group_mem_alloc(Group& group, int n_atoms)
{
    CUDA_CHECK(cudaMalloc(&group.d_types,   group.h_types.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&group.d_n_atoms, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&group.d_atoms,   n_atoms * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(group.d_types,    group.h_types.data(), group.h_types.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(group.d_n_atoms,   0,  sizeof(int)));
    CUDA_CHECK(cudaMemset(group.d_atoms,    -1,  n_atoms * sizeof(int)));
}

/* -------------------------------------------------------------------------------------- */

void group_mem_free(System& system)
{
    for (auto& pair : system.groups) 
    {
        Group& group = pair.second;  
        CUDA_CHECK(cudaFree(group.d_types));
        CUDA_CHECK(cudaFree(group.d_n_atoms));
        CUDA_CHECK(cudaFree(group.d_atoms));
    }

    system.groups.clear();  
}

/* -------------------------------------------------------------------------------------- */

Group& find_group(System& system, const std::string name) 
{
    auto it = system.groups.find(name);
    if (it != system.groups.end()) {
        return it->second;  
    } else {
        print_error({"Group " + name + " not found"});
        exit(EXIT_FAILURE);
    }
}

/* -------------------------------------------------------------------------------------- */

static __global__ void kernel_update_group_by_types
(
    const int  N, 
    const int *atom_type,  
    const int  group_n_types, 
    const int *group_types,
    int *group_n_atoms, 
    int *group_atoms
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    for (int t = 0; t < group_n_types; t++)
    {
        int typet = group_types[t];
        if (atom_type[i] == typet) 
        {
            int index = atomicAdd(group_n_atoms, 1);
            group_atoms[index] = i;
        }    
    }
}

/* -------------------------------------------------------------------------------------- */

void group_select(System& system, Group& group)
{
    /* ------------------------------------------------------- */

    int N          = system.n_atoms;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;

    kernel_update_group_by_types<<<numBlocks, blockSize>>>
    (
        N,
        atoms.d_type,  
        group.h_types.size(),  
        group.d_types,
        group.d_n_atoms,
        group.d_atoms
    );   

    /* ------------------------------------------------------- */

    int n_atoms = 0;  
    CUDA_CHECK(cudaMemcpy(&n_atoms, group.d_n_atoms, sizeof(int), cudaMemcpyDeviceToHost));
    group.n_atoms = n_atoms;

    /* ------------------------------------------------------- */
}

/* -------------------------------------------------------------------------------------- */

void print_groups(const System& system) 
{    
    for (const auto& pair : system.groups) 
    {
        const Group& group = pair.second;
        std::cout << "   group " << group.name << " type ";
        for (const int& type : group.h_types) {
            std::cout << type << " ";
        }
        std::cout << "(" << group.n_atoms << " atoms)" << std::endl;
    }
}

/* -------------------------------------------------------------------------------------- */
