#pragma once
#include "compute/compute.cuh"

class wall_contact : public Compute {
public:
    wall_contact(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void preprocess(System& system) override;
    void postprocess(System& system) override;

    void compute(System& system, unsigned int step) override;

private: 
    std::string filename;
    std::ofstream file;
    
    int frequency;
    std::string wall_groupname;
    int wall_direction;
    int wall_side; 
    numtyp cutoff;

    numtyp *wall_pos;
    
    int *d_contact;
};  