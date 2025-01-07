#pragma once
#include "main/common.cuh"
#include "system/system.cuh"

class Read_in 
{
public:

    void read_txt(const std::string filepath, std::string& input);
    void read(System& system);

private:

    bool is_blank_line(const std::string line);

    bool got_atom_style = false;
    bool got_neighbor = false;
    bool got_timestep = false;
    bool got_thermo = false;
    bool got_run = false;
    bool got_read_data = false;
    bool got_pair_style = false;
    bool got_bond_style = false;
    bool got_angle_style = false;

    void check(const System& system);
};
