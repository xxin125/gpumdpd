#include "force/bond_harmonic.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

Bond_harmonic::Bond_harmonic() 
{
    bond_style_name = "harmonic";
}

/* ----------------------------------------------------------------------------------------------------------- */

std::string Bond_harmonic::getName() 
{
    return bond_style_name;
}

/* ----------------------------------------------------------------------------------------------------------- */\

bool Bond_harmonic::isEnabled(System& system) 
{
    /* ------------------------------------------------------- */

    std::string& input = system.input; 
    int n_bondtypes    = system.n_bondtypes;
    bool enabled       = false;

    /* ------------------------------------------------------- */

    // bond_coeff

    got_bond_coeff.resize(n_bondtypes, 0);
    bond_coeff.resize(got_bond_coeff.size()*2); 

    /* ------------------------------------------------------- */

    // read run_in

    std::stringstream ss(input);
    std::string line;
    std::vector<std::string> args;
    std::string arg;

    /* ------------------------------------------------------- */

    while (std::getline(ss, line)) 
    {
        std::istringstream iss(line);
        args.clear();
        std::string key;
        iss >> key;

        while (iss >> arg) {
            args.push_back(arg);
        }

        // read bond_style
        
        if (key == "bond_style") 
        {
            if (args[0] == "harmonic") {enabled = true;}

            if (enabled)
            {
                std::string error    = "illegal bond_style harmonic command";
                std::string format   = "bond_style bond_style";
                std::string example0 = "bond_style harmonic";
                if (args.size() != 1) {
                    print_error_and_exit(line, error, format, {example0});
                }
            }
        }
    }

    /* ------------------------------------------------------- */

    // read bond_coeff

    if (enabled) 
    {
        std::stringstream ss2(input);
        while (std::getline(ss2, line)) 
        {
            std::istringstream iss(line);
            args.clear();
            std::string key;
            iss >> key;
    
            while (iss >> arg) {
                args.push_back(arg);
            }
    
            if (key == "bond_coeff") 
            {
                std::string error    = "illegal bond_coeff command for bond_style harmonic";
                std::string format   = "bond_coeff type type k r0";
                std::string example0 = "bond_coeff 1 100.0 0.675";

                if (args.size() != 3) {
                    print_error_and_exit(line, error, format, {example0});
                }

                int type = parse_int<int>(args[0], line, "type", {format, example0});

                if (type > n_bondtypes) {
                    print_error({"type exceeds n_bond_types"});                     
                } 

                numtyp k  = parse_float<numtyp>(args[1], line, "k", {format, example0});
                numtyp r0 = parse_float<numtyp>(args[2], line, "r0", {format, example0});
                int index  = type - 1;
                got_bond_coeff[index] = 1;
                bond_coeff[index*2+0] = k;
                bond_coeff[index*2+1] = r0;
            }
        } 
    }

    /* ------------------------------------------------------- */

    // check all bond_coeff

    if (enabled) 
    {
        bool all_bond_coeff = true;
        for (int i=0; i < got_bond_coeff.size(); i++) 
        {
            if (got_bond_coeff[i] == 0) 
            {
                all_bond_coeff = false;
            }
        }
        if (!all_bond_coeff) 
        {
            std::string error = "all bond_coeff are not set";
            print_error({error});      
        } else {
            CUDA_CHECK(cudaMemcpyToSymbol(gpu_bond_coeff, bond_coeff.data(), (bond_coeff.size() * sizeof(numtyp))));       
        }

    }

    /* ------------------------------------------------------- */

    return enabled;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Bond_harmonic::print_bond_info(System& system) 
{
    std::cout << "   bond_style           harmonic" << std::endl;
    std::cout << "   bond_coeff           type k r0" << std::endl;
    for (int i=0; i<system.n_bondtypes; i++)
    {
        int type = i+1;
        int _id  = type-1;
        std::cout << "   bond_coeff           " << type << " " << bond_coeff[_id*2+0] << " " << bond_coeff[_id*2+1] << std::endl;
    }
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl; 
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_bond_harmonic_log
(
    const int N,                  
    const int *d_id,              const int *d_bondlist, 
    const numtyp *d_pos,          numtyp *d_force,               numtyp *d_bond_pe,            numtyp *d_viral,
    const numtyp lx,              const numtyp ly,               const numtyp lz,
    const numtyp hlx,             const numtyp hly,              const numtyp hlz
)
{
    /* ------------------------------------------------------- */

    const int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    /* ------------------------------------------------------- */

    const int bond_type     = d_bondlist[i*3+0];
    const int bond_atomi_id = d_bondlist[i*3+1];
    const int bond_atomj_id = d_bondlist[i*3+2];

    /* ------------------------------------------------------- */

    const int bond_type_index = bond_type - 1; 
    const numtyp spring_k     = gpu_bond_coeff[bond_type_index*2+0]; 
    const numtyp equi_dis     = gpu_bond_coeff[bond_type_index*2+1]; 

    /* ------------------------------------------------------- */

    const int bond_atomi_index = bond_atomi_id - 1;
    const numtyp pos_x_i       = d_pos[bond_atomi_index*3+0];
    const numtyp pos_y_i       = d_pos[bond_atomi_index*3+1];
    const numtyp pos_z_i       = d_pos[bond_atomi_index*3+2];

    const int bond_atomj_index = bond_atomj_id - 1;
    const numtyp pos_x_j       = d_pos[bond_atomj_index*3+0];
    const numtyp pos_y_j       = d_pos[bond_atomj_index*3+1];
    const numtyp pos_z_j       = d_pos[bond_atomj_index*3+2];

    /* ------------------------------------------------------- */

    numtyp dx = pos_x_i - pos_x_j;
    numtyp dy = pos_y_i - pos_y_j;
    numtyp dz = pos_z_i - pos_z_j;    

    dx = dx - lx * ((dx >= hlx) - (dx < -hlx));
    dy = dy - ly * ((dy >= hly) - (dy < -hly));
    dz = dz - lz * ((dz >= hlz) - (dz < -hlz));
    const numtyp r2 = dx * dx + dy * dy + dz * dz;
    const numtyp r = sqrt(r2);
        
    /* --------------------------------------------------- */

    numtyp fbond = 0.0;

    if (r > (numtyp)(0.0)) {
        fbond = (numtyp)(-2.0) * spring_k * (r - equi_dis) / r;
    } else {
        fbond = (numtyp)0.0;
    }

    /* --------------------------------------------------- */

    const numtyp fbondx = (dx * fbond);
    const numtyp fbondy = (dy * fbond);
    const numtyp fbondz = (dz * fbond);

    atomicAdd(&d_force[bond_atomi_index*3+0], fbondx);
    atomicAdd(&d_force[bond_atomi_index*3+1], fbondy);
    atomicAdd(&d_force[bond_atomi_index*3+2], fbondz);

    atomicAdd(&d_force[bond_atomj_index*3+0], -fbondx);
    atomicAdd(&d_force[bond_atomj_index*3+1], -fbondy);
    atomicAdd(&d_force[bond_atomj_index*3+2], -fbondz);

    /* --------------------------------------------------- */

    const numtyp bond_pe  = spring_k * (r - equi_dis) * (r - equi_dis);
    atomicAdd(&d_bond_pe[i], bond_pe);

    /* --------------------------------------------------- */

    const numtyp viral0 = (numtyp)0.5 * dx * fbondx;
    const numtyp viral1 = (numtyp)0.5 * dx * fbondy;
    const numtyp viral2 = (numtyp)0.5 * dx * fbondz;
    const numtyp viral3 = (numtyp)0.5 * dy * fbondy;
    const numtyp viral4 = (numtyp)0.5 * dy * fbondz;
    const numtyp viral5 = (numtyp)0.5 * dz * fbondz;

    atomicAdd(&d_viral[bond_atomi_index*6+0], viral0);
    atomicAdd(&d_viral[bond_atomi_index*6+1], viral1);
    atomicAdd(&d_viral[bond_atomi_index*6+2], viral2);
    atomicAdd(&d_viral[bond_atomi_index*6+3], viral3);
    atomicAdd(&d_viral[bond_atomi_index*6+4], viral4);
    atomicAdd(&d_viral[bond_atomi_index*6+5], viral5);  

    atomicAdd(&d_viral[bond_atomj_index*6+0], viral0);
    atomicAdd(&d_viral[bond_atomj_index*6+1], viral1);
    atomicAdd(&d_viral[bond_atomj_index*6+2], viral2);
    atomicAdd(&d_viral[bond_atomj_index*6+3], viral3);
    atomicAdd(&d_viral[bond_atomj_index*6+4], viral4);
    atomicAdd(&d_viral[bond_atomj_index*6+5], viral5);  

    /* --------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_bond_harmonic
(
    const int N,                  
    const int *d_id,              const int *d_bondlist, 
    const numtyp *d_pos,          numtyp *d_force,  
    const numtyp lx,              const numtyp ly,               const numtyp lz,
    const numtyp hlx,             const numtyp hly,              const numtyp hlz
)
{
    /* ------------------------------------------------------- */

    const int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    /* ------------------------------------------------------- */

    const int bond_type     = d_bondlist[i*3+0];
    const int bond_atomi_id = d_bondlist[i*3+1];
    const int bond_atomj_id = d_bondlist[i*3+2];

    /* ------------------------------------------------------- */

    const int bond_type_index = bond_type - 1; 
    const numtyp spring_k     = gpu_bond_coeff[bond_type_index*2+0]; 
    const numtyp equi_dis     = gpu_bond_coeff[bond_type_index*2+1]; 

    /* ------------------------------------------------------- */

    const int bond_atomi_index = bond_atomi_id - 1;
    const numtyp pos_x_i       = d_pos[bond_atomi_index*3+0];
    const numtyp pos_y_i       = d_pos[bond_atomi_index*3+1];
    const numtyp pos_z_i       = d_pos[bond_atomi_index*3+2];

    const int bond_atomj_index = bond_atomj_id - 1;
    const numtyp pos_x_j       = d_pos[bond_atomj_index*3+0];
    const numtyp pos_y_j       = d_pos[bond_atomj_index*3+1];
    const numtyp pos_z_j       = d_pos[bond_atomj_index*3+2];

    /* ------------------------------------------------------- */

    numtyp dx = pos_x_i - pos_x_j;
    numtyp dy = pos_y_i - pos_y_j;
    numtyp dz = pos_z_i - pos_z_j;    

    dx = dx - lx * ((dx >= hlx) - (dx < -hlx));
    dy = dy - ly * ((dy >= hly) - (dy < -hly));
    dz = dz - lz * ((dz >= hlz) - (dz < -hlz));
    const numtyp r2 = dx * dx + dy * dy + dz * dz;
    const numtyp r = sqrt(r2);
        
    /* --------------------------------------------------- */

    numtyp fbond = 0.0;

    if (r > (numtyp)(0.0)) {
        fbond = (numtyp)(-2.0) * spring_k * (r - equi_dis) / r;
    } else {
        fbond = (numtyp)0.0;
    }

    /* --------------------------------------------------- */

    const numtyp fbondx = (dx * fbond);
    const numtyp fbondy = (dy * fbond);
    const numtyp fbondz = (dz * fbond);

    atomicAdd(&d_force[bond_atomi_index*3+0], fbondx);
    atomicAdd(&d_force[bond_atomi_index*3+1], fbondy);
    atomicAdd(&d_force[bond_atomi_index*3+2], fbondz);

    atomicAdd(&d_force[bond_atomj_index*3+0], -fbondx);
    atomicAdd(&d_force[bond_atomj_index*3+1], -fbondy);
    atomicAdd(&d_force[bond_atomj_index*3+2], -fbondz);

    /* --------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Bond_harmonic::compute_force(System& system, unsigned int step)
{
    /* ------------------------------------------------------- */

    Atoms& atoms          = system.atoms;
    Box& box              = system.box;
    Run_p& run_p          = system.run_p;

    /* ------------------------------------------------------- */
    
    bool log = false;

    if (step % run_p.log_f == 0) {
        log = true;
    }

    /* ------------------------------------------------------- */

    int N           = system.n_bonds;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    if (log)
    {
        CUDA_CHECK(cudaMemset(atoms.d_bond_pe, 0, sizeof(numtyp)*N));
        kernel_bond_harmonic_log<<<numBlocks, blockSize>>>
        (
            N,                                  
            atoms.d_id,                   atoms.d_bondlist, 
            atoms.d_pos,                  atoms.d_force,                atoms.d_bond_pe,    atoms.d_viral,
            box.lx,                       box.ly,                       box.lz,
            box.hlx,                      box.hly,                      box.hlz
        ); 
    }
    else 
    {
        kernel_bond_harmonic<<<numBlocks, blockSize>>>
        (
            N,                                  
            atoms.d_id,                   atoms.d_bondlist, 
            atoms.d_pos,                  atoms.d_force,                
            box.lx,                       box.ly,                       box.lz,
            box.hlx,                      box.hly,                      box.hlz
        );            
    }

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////
REGISTER_BOND(Bond_harmonic)
///////////////////////