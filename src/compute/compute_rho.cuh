#pragma once
#include "compute/compute.cuh"

class rho : public Compute {
public:
    rho(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void preprocess(System& system) override;
    void postprocess(System& system) override;

    void compute(System& system, unsigned int step) override;

private: 
    std::string filename;
    std::ofstream file;
    int frequency;

    numtyp xlo, xhi;
    numtyp ylo, yhi;
    numtyp zlo, zhi;
    numtyp vol;

    int *d_count;
};  