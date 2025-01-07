#pragma once

#include "main/common.cuh"
#include "main/read_in.cuh"

#include "system/system.cuh"
#include "system/read_data.cuh"
#include "system/dump.cuh"

#include "force/neigh_list.cuh"
#include "force/pair.cuh" 
#include "force/bond.cuh" 
#include "force/angle.cuh" 

#include "compute/thermo.cuh" 
#include "compute/compute.cuh" 

#include "fix/fix.cuh" 

/* ----------------------------------------------------------------------------------------------------------- */

void print_program_info() 
{
    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                GPUDPD                                      " << std::endl;
    std::cout << "                       Xinxin Deng (TU Darmstadt)                           " << std::endl;
    std::cout << "                          Müller-Plathe Group                               " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */

void print_header(System& system) 
{
    print_program_info(); 

    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                          Device Information                                " << std::endl;
    std::cout << "                                                                            " << std::endl;

    int total_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&total_gpus));

    if (total_gpus == 0) {
        std::cout << "   No GPU found on this system!" << std::endl;
        return;
    }

    int dev = 0;  
    CUDA_CHECK(cudaSetDevice(dev)); 
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    std::cout << "   Device ID: "           << dev              << std::endl;
    std::cout << "   Device Name: "         << deviceProp.name  << std::endl;
    std::cout << "   Compute Capability: "  << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "   Total Global Memory: " 
              << static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024) 
              << " MB\n";

    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */

void read_input(System& system, Read_in& read_in) 
{
    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                          Basic input information                           " << std::endl;
    std::cout << "                                                                            " << std::endl;

    read_in.read_txt("run.in", system.input);
    read_in.read(system);

    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */

void read_system(System& system, Read_data& read_data) 
{
    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                           System Information                               " << std::endl;
    std::cout << "                                                                            " << std::endl;

    read_data.read_data(system);

    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */

void read_groups(System& system) 
{
    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                           Groups Information                               " << std::endl;
    std::cout << "                                                                            " << std::endl;

    create_groups(system);
    print_groups(system);

    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */

void read_info(System& system, Read_in& read_in, Read_data& read_data) 
{
    read_input(system, read_in);
    read_system(system, read_data);
    read_groups(system);
}

/* ----------------------------------------------------------------------------------------------------------- */

void preprocess(
    System&       system, 
    Neigh_list&   neigh_list,
    Thermo&       thermo,
    Dump&         dump
) 
{
    /* ---------------------------------------- */

    neigh_list.preprocess(system);

    /* ---------------------------------------- */

    preprocessPairs(system);
    if (system.atom_style == 1) {
        preprocessBonds(system);
    }
    if (system.atom_style == 2) {
        preprocessBonds(system);
        preprocessAngles(system);
    }

    /* ---------------------------------------- */

    preprocessFixes(system);
    preprocessCompute(system);

    /* ---------------------------------------- */

    thermo.preprocess(system);

    if (system.run_p.write_dump_flag) {
        dump.preprocess(system);
    }

    /* ---------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void zero_step(
    System&       system, 
    Neigh_list&   neigh_list,
    Read_data&    read_data,
    Thermo&       thermo,
    Dump&         dump
) 
{
    /* ---------------------------------------- */

    neigh_list.build(system);

    /* ---------------------------------------- */

    Pair_compute(system, 0);
    if (system.atom_style == 1) {
        Bond_compute(system, 0);
    }
    if (system.atom_style == 2) {
        Bond_compute(system, 0);
        Angle_compute(system, 0);
    }

    /* ---------------------------------------- */

    thermo.process(system, 0);
    Compute_compute(system, 0);

    /* ---------------------------------------- */

    if (system.run_p.write_dump_flag) {   
        if (system.run_p.dump_start == 0) {
            dump.dump(system, 0);
        }
    }

    if (system.run_p.write_data_flag) {   
        if (system.run_p.nsteps == 0) {
            read_data.write_data(system);  
        }       
    }    

    /* ---------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void neigh_update(
    System&       system,  
    Neigh_list&   neigh_list, 
    int           step
) 
{
    /* ---------------------------------------------------- */
    
    if (step % system.run_p.nl_f == 0) 
    {
        if (system.run_p.skin != static_cast<numtyp>(0.0))
        {
            neigh_list.check_update(system);
            if (system.neigh.h_update_flag == 1)
            {
                neigh_list.build(system);
            } 
        }
        else 
        {
            neigh_list.build(system);                
        }  
    }      
    
    /* ---------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void simulation(
    System&       system, 
    Neigh_list&   neigh_list, 
    Thermo&       thermo,
    Read_data&    read_data,
    Dump&         dump
) 
{
    for (unsigned int step = 1; step <= system.run_p.nsteps; ++step) 
    {

        /* ---------------------------------------- */

        Fix_initial_integrate(system, step);

        /* ---------------------------------------- */

        neigh_update(system, neigh_list, step);

        /* ---------------------------------------- */

        Pair_compute(system, step);
        if (system.atom_style == 1) {
            Bond_compute(system, step);
        }
        if (system.atom_style == 2) {
            Bond_compute(system, step);
            Angle_compute(system, step);
        }

        /* ---------------------------------------- */

        Fix_post_force(system, step);

        /* ---------------------------------------- */

        Fix_final_integrate(system, step);

        /* ---------------------------------------- */

        Fix_end_of_step(system, step);

        /* ---------------------------------------- */

        thermo.process(system, step);
        Compute_compute(system, step);

        /* ---------------------------------------- */

        if (system.run_p.write_dump_flag) {
            if (( step >= system.run_p.dump_start) && ( step - system.run_p.dump_start) % system.run_p.dump_f == 0) {
                dump.dump(system, step);
            }
        }
    
        /* ---------------------------------------- */
    }

    /* -------------------------------------------- */

    if (system.run_p.write_data_flag) {   
        if (system.run_p.nsteps > 0) {
            read_data.write_data(system);  
        }       
    }    

    /* -------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void postprocess(
    System&       system,
    Neigh_list&   neigh_list,
    Thermo&       thermo,
    Dump&         dump
) 
{
    /* ---------------------------------------- */
    
    atoms_mem_free(system);

    if (system.atom_style == 1) {
        bonds_mem_free(system);
    }

    if (system.atom_style == 2) {
        bonds_mem_free(system);
        angles_mem_free(system);
    }

    /* ---------------------------------------- */

    neigh_list.postprocess(system);

    /* ---------------------------------------- */

    postprocessPairs();
    if (system.atom_style == 1) {
        postprocessBonds();
    }
    if (system.atom_style == 2) {
        postprocessBonds();
        postprocessAngles();
    }

    /* ---------------------------------------- */

    postprocessFixes(system);
    postprocessCompute(system);

    /* ---------------------------------------- */

    thermo.postprocess(system);

    if (system.run_p.write_dump_flag) {
        dump.postprocess(system);
    }

    /* ---------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void print_performance(System& system, const std::chrono::steady_clock::time_point& startTime, const std::chrono::steady_clock::time_point& endTime) 
{
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
    double totalSeconds = duration.count();
    int hours = static_cast<int>(totalSeconds) / 3600;
    int minutes = (static_cast<int>(totalSeconds) % 3600) / 60;
    double seconds = totalSeconds - (hours * 3600 + minutes * 60);
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;
    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "   Time Taken:     " << hours << " h: " << minutes << " m: " << seconds << " s" << std::endl;
    double speed_time_per_s = (system.run_p.nsteps + 1) / totalSeconds;
    double speed_Matoms_time_per_s = speed_time_per_s*system.n_atoms/1000000;
    std::cout << "   Performance:    " << speed_time_per_s << " steps/s    " << speed_Matoms_time_per_s << " Matom*steps/s" << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */

