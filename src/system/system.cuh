#pragma once
#include "main/common.cuh"

/* -------------------------------------------------------------------------------------- */

struct Atoms
{
    /* ------------------------------------------------- */

    std::vector<int>     h_id;
    std::vector<int>     h_type;
    std::vector<numtyp>  h_pos;
    std::vector<numtyp>  h_vel;
    std::vector<numtyp>  h_uwpos;
    std::vector<numtyp>  h_force;
    std::vector<numtyp>  h_pe;

    /* ------------------------- */

    std::vector<int>     h_mol_id;
    std::vector<int>     h_bondlist;
    std::vector<int>     h_anglelist;

    /* ------------------------------------------------- */

    int                 *d_id;
    int                 *d_type;
    numtyp              *d_pos;
    numtyp              *d_vel;
    numtyp              *d_uwpos;
    numtyp              *d_force;
    numtyp              *d_pe;
    numtyp              *d_viral;
    numtyp              *d_rho;

    /* ------------------------- */

    int                 *d_n_neigh;
    int                 *d_neigh;

    /* ------------------------- */

    int                 *d_bondlist;
    int                 *d_anglelist;
    numtyp              *d_bond_pe;
    numtyp              *d_angle_pe;

    /* ------------------------------------------------- */
};

/* -------------------------------------------------------------------------------------- */

struct Box
{
    numtyp               xlo, ylo, zlo;
    numtyp               xhi, yhi, zhi;
    numtyp               lx,  ly,  lz;
    numtyp               hlx, hly, hlz;
};

/* -------------------------------------------------------------------------------------- */

struct Group
{
    /* ------------------------- */

    std::string          name;
    int                  n_atoms;
    std::vector<int>     h_types;

    /* ------------------------- */

    int                 *d_types;
    int                 *d_n_atoms; 
    int                 *d_atoms; 

    /* ------------------------- */
};

/* -------------------------------------------------------------------------------------- */

struct Neigh
{
    /* ------------------------------------ */

    numtyp               bin_size;
    std::vector<int>     n_totalbinsxyz;
    int                  n_totalbins;

    /* ------------------------------------ */

    std::vector<int>     h_neighborbinsID;
    int                 *d_neighborbinsID;

    /* ------------------------------------ */

    int                 *binContent;
    int                 *binCounts;
    int                 *binCounts_prefix_sum;

    /* ------------------------------------ */
    
    int                  h_update_flag;
    int                 *d_update_flag;
    numtyp              *last_pos;
    
    /* ------------------------------------ */
};

/* -------------------------------------------------------------------------------------- */

struct System
{
    /* ------------------------- */

    std::string          input;
     
    /* ------------------------- */
                    
    int                  n_max_neigh;

    /* ------------------------- */

    int                  atom_style; 

    /* ------------------------- */

    int                  n_atoms;            
    int                  n_atomtypes;  

    /* ------------------------- */

    int                  n_bonds;
    int                  n_bondtypes;

    /* ------------------------- */

    int                  n_angles;
    int                  n_angletypes;

    /* ------------------------- */

    Atoms                atoms;
    Box                  box;
    Run_p                run_p;
    Thermo_p             thermo_p;
    Neigh                neigh;

    /* ------------------------- */

    std::map<std::string, Group> groups;

    /* ------------------------- */
};

/* -------------------------------------------------------------------------------------- */

void atoms_mem_alloc(System& system);
void atoms_mem_copy_2_gpu(System& system, std::vector<numtyp> data_masses);
void atoms_mem_free(System& system);

void bonds_mem_alloc(System& system);
void bonds_mem_copy_2_gpu(System& system);
void bonds_mem_free(System& system);

void angles_mem_alloc(System& system);
void angles_mem_copy_2_gpu(System& system);
void angles_mem_free(System& system);

/* -------------------------------------------------------------------------------------- */

void create_groups(System& system);
void group_mem_alloc(Group& group, int n_atoms);
void group_mem_free(System& system);

Group& find_group(System& system, const std::string name);
void group_select(System& system, Group& group); 
void print_groups(const System& system);

/* -------------------------------------------------------------------------------------- */