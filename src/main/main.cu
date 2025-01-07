#include "main/common.cuh"
#include "main/utils.cuh"

/* ---------------------------------------------------------------------- */

// gpu constant memory

__constant__ numtyp masses[256]; 
__constant__ numtyp gpu_pair_coeff[256];
__constant__ numtyp gpu_bond_coeff[256];
__constant__ numtyp gpu_angle_coeff[256];

/* ---------------------------------------------------------------------- */

int main() 
{
    /* ------------------------------------------------------------------ */

    auto startTime = std::chrono::steady_clock::now();

    /* ------------------------------------------------------------------ */

    // simualtion structs and classes

    System        system;
    Read_in       read_in;
    Read_data     read_data;
    Neigh_list    neigh_list;
    Thermo        thermo;
    Dump          dump;

    /* ------------------------------------------------------------------ */    

    // print header

    print_header(system);

    // read info 

    read_info(system, read_in, read_data);

    // preprocess 

    preprocess(system, neigh_list, thermo, dump);

    /* ------------------------------------------------------------------ */  
    /* ------------------------------------------------------------------ */ 

    // zero step 

    zero_step(system, neigh_list, read_data, thermo, dump);
    
    // simulation

    simulation(system, neigh_list, thermo, read_data, dump);

    /* ------------------------------------------------------------------ */  
    /* ------------------------------------------------------------------ */ 

    // postprocess 

    postprocess(system, neigh_list, thermo, dump);

    // print performance

    auto endTime = std::chrono::steady_clock::now();
    print_performance(system, startTime, endTime);

    /* ------------------------------------------------------------------ */  

    return 0;
}

/* ---------------------------------------------------------------------- */    