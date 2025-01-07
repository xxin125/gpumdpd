#pragma once
#include "main/common.cuh"
#include "system/system.cuh"

class Thermo
{
public:
    void preprocess(System& system);
    void process(System& system, unsigned int step);
    void postprocess(System& system);

    void compute_temp(System& system);
    void compute_pe(System& system);
    void compute_bond_pe(System& system);
    void compute_angle_pe(System& system);
    void compute_pressure(System& system);
    
    FILE* thermofile;
};

