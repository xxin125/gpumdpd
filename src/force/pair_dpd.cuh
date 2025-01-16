#pragma once
#include "force/pair.cuh"

class Pair_dpd : public Pair 
{
public:

    Pair_dpd();

    bool isEnabled(System& system) override;
    void print_pair_info(System& system) override;
    std::string getName() override;
    void compute_force(System& system, unsigned int step) override;

private:

    // pair_style dpd temp seed

    std::string pair_style_name;
    numtyp temp;
    numtyp rc;
    unsigned int seed;

    // pair_coeff typei typej A gamma rc

    std::vector<numtyp> pair_coeff;

    // got_pair_coeff

    std::vector<int> got_pair_coeff;
};

