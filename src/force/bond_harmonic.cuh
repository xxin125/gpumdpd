#pragma once
#include "force/bond.cuh"

class Bond_harmonic : public Bond 
{
public:

    Bond_harmonic();

    bool isEnabled(System& system) override;
    void print_bond_info(System& system) override;
    std::string getName() override;
    void compute_force(System& system, unsigned int step) override;

private:

    // bond_style harmonic

    std::string bond_style_name;

    // bond_coeff type k r0

    std::vector<numtyp> bond_coeff;

    // got_bond_coeff

    std::vector<int> got_bond_coeff;
};

