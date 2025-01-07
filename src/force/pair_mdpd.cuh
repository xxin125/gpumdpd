#pragma once
#include "force/pair.cuh"

class Pair_mdpd : public Pair 
{
public:

    Pair_mdpd();

    bool isEnabled(System& system) override;
    void print_pair_info(System& system) override;
    std::string getName() override;
    void compute_force(System& system, unsigned int step) override;

private:

    // pair_style mdpd temp seed

    std::string pair_style_name;
    numtyp temp;
    numtyp rc;
    numtyp rd;
    unsigned int seed;

    // pair_coeff typei typej A B gamma

    std::vector<numtyp> pair_coeff;

    // got_pair_coeff

    std::vector<int> got_pair_coeff;
};

