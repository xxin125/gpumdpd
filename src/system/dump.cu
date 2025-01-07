#include "dump.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

void Dump::preprocess(System& system)
{
    dumpfile = fopen("all.trj", "w");
}

/* ----------------------------------------------------------------------------------------------------------- */

void Dump::postprocess(System& system)
{
    if (dumpfile != nullptr)
    {
        fclose(dumpfile);
        dumpfile = nullptr; 
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

void Dump::dump(System& system, unsigned int step)
{
    /* ------------------------------------------------------- */

    Atoms& atoms = system.atoms;
    Box& box     = system.box;
    Run_p& run_p = system.run_p;
    int N        = system.n_atoms;

    /* ---------------------------------------------------------------------------*/

    // wrapped position

    if (run_p.wrapped_flag) 
    {
        CUDA_CHECK(cudaMemcpy(atoms.h_id.data(),     atoms.d_id,    sizeof(int)    * N,     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(atoms.h_type.data(),   atoms.d_type,  sizeof(int)    * N,     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(atoms.h_pos.data(),    atoms.d_pos,   sizeof(numtyp) * N * 3, cudaMemcpyDeviceToHost));
    } 

    /* ---------------------------------------------------------------------------*/

    // unwrapped position

    if (!run_p.wrapped_flag) 
    {
        CUDA_CHECK(cudaMemcpy(atoms.h_id.data(),     atoms.d_id,     sizeof(int)    * N,     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(atoms.h_type.data(),   atoms.d_type,   sizeof(int)    * N,     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(atoms.h_uwpos.data(),  atoms.d_uwpos,  sizeof(numtyp) * N * 3, cudaMemcpyDeviceToHost));
    }

    /* ---------------------------------------------------------------------------*/

    // force

    if (run_p.dumpforce_flag) 
    {
        CUDA_CHECK(cudaMemcpy(atoms.h_force.data(),  atoms.d_force,  sizeof(numtyp) * N * 3, cudaMemcpyDeviceToHost));    
    }

    /* ---------------------------------------------------------------------------*/

    // velocity

    if (run_p.dumpvel_flag) 
    {
        if (run_p.dumpforce_flag) 
        {
            CUDA_CHECK(cudaMemcpy(atoms.h_vel.data(),  atoms.d_vel,  sizeof(numtyp) * N * 3, cudaMemcpyDeviceToHost));    
        }   
    }

    /* ---------------------------------------------------------------------------*/

    // dump file header

    fprintf(dumpfile, "ITEM: TIMESTEP\n%d\n", step);
    fprintf(dumpfile, "ITEM: NUMBER OF ATOMS\n%d\n", system.n_atoms);
    fprintf(dumpfile, "ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(dumpfile, "%.9f %.9f\n", box.xlo, box.xhi);
    fprintf(dumpfile, "%.9f %.9f\n", box.ylo, box.yhi);
    fprintf(dumpfile, "%.9f %.9f\n", box.zlo, box.zhi);

    /* ---------------------------------------------------------------------------*/

    // dump wrapped position

    if (run_p.wrapped_flag) 
    {
        if (!run_p.dumpforce_flag && !run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_pos[i*3+0], atoms.h_pos[i*3+1], atoms.h_pos[i*3+2]);
            }
        } 
        else if (run_p.dumpforce_flag && !run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z fx fy fz\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_pos[i*3+0],   atoms.h_pos[i*3+1],   atoms.h_pos[i*3+2],
                        atoms.h_force[i*3+0], atoms.h_force[i*3+1], atoms.h_force[i*3+2]);
            }
        } 
        else if (!run_p.dumpforce_flag && run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z vx vy vz\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_pos[i*3+0], atoms.h_pos[i*3+1], atoms.h_pos[i*3+2],
                        atoms.h_vel[i*3+0], atoms.h_vel[i*3+1], atoms.h_vel[i*3+2]);
            }
        } 
        else if (run_p.dumpforce_flag && run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z fx fy fz vx vy vz\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_pos[i*3+0],   atoms.h_pos[i*3+1],   atoms.h_pos[i*3+2],
                        atoms.h_force[i*3+0], atoms.h_force[i*3+1], atoms.h_force[i*3+2],
                        atoms.h_vel[i*3+0],   atoms.h_vel[i*3+1],   atoms.h_vel[i*3+2]);
            }
        }
    }

    /* ---------------------------------------------------------------------------*/

    // dump unwrapped position

    if (!run_p.wrapped_flag) 
    {
        if (!run_p.dumpforce_flag && !run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_uwpos[i*3+0], atoms.h_uwpos[i*3+1], atoms.h_uwpos[i*3+2]);
            }
        } 
        else if (run_p.dumpforce_flag && !run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z fx fy fz\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_uwpos[i*3+0], atoms.h_uwpos[i*3+1], atoms.h_uwpos[i*3+2],
                        atoms.h_force[i*3+0], atoms.h_force[i*3+1], atoms.h_force[i*3+2]);
            }
        } 
        else if (!run_p.dumpforce_flag && run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z vx vy vz\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_uwpos[i*3+0], atoms.h_uwpos[i*3+1], atoms.h_uwpos[i*3+2],
                        atoms.h_vel[i*3+0],   atoms.h_vel[i*3+1],   atoms.h_vel[i*3+2]);
            }
        } 
        else if (run_p.dumpforce_flag && run_p.dumpvel_flag) 
        {
            fprintf(dumpfile, "ITEM: ATOMS id type x y z fx fy fz vx vy vz\n");
            for (int i = 0; i < system.n_atoms; ++i) 
            {
                fprintf(dumpfile, "%d %d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", 
                        atoms.h_id[i], 
                        atoms.h_type[i],
                        atoms.h_uwpos[i*3+0], atoms.h_uwpos[i*3+1], atoms.h_uwpos[i*3+2],
                        atoms.h_force[i*3+0], atoms.h_force[i*3+1], atoms.h_force[i*3+2],
                        atoms.h_vel[i*3+0],   atoms.h_vel[i*3+1],   atoms.h_vel[i*3+2]);
            }
        }
    }

    /* ---------------------------------------------------------------------------*/

    fflush(dumpfile);    
}

/* ----------------------------------------------------------------------------------------------------------- */
