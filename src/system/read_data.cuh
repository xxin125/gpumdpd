#pragma once
#include "main/common.cuh"
#include "system/system.cuh"

class Read_data 
{
public:

    void read_data(System& system);
    void write_data(System& system);
    
private:

    std::string input_data;
    std::string output_data;

    std::vector<numtyp> masses;
    int id, type;
    numtyp x, y, z;
    numtyp vx, vy, vz;

    bool has_ini_v;  

    int mol;
    int bondid, bondtype, bond_atomi_id, bond_atomj_id;
    int angleid, angletype, angle_atomi_id, angle_atomj_id, angle_atomk_id;
    
    void check_atom_style(System& system);
    void read_system_data(System& system);
    void read_box_info(System& system, std::ifstream& inputFile);
    void read_masses(System& system, std::ifstream& inputFile);
    void read_header_info(System& system, std::ifstream& inputFile); 
    void adjust_coordinate(numtyp& coord, numtyp lo, numtyp hi); 
    void check_velocities(System& system, std::ifstream& inputFile, std::string& section_name);
    void process_bonds(System& system, std::ifstream& inputFile);
    void process_angles(System& system, std::ifstream& inputFile);
};
