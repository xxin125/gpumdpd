#pragma once
#include "fix/fix.cuh"

class recenter : public Fix {
public:
    recenter(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void preprocess(System& system) override;
    void postprocess(System& system) override;

    void end_of_step(System& system, unsigned int step) override;

private: 
    int frequency;
    numtyp recenter_x;
    numtyp recenter_y;
    numtyp recenter_z;

    numtyp *t_mx;
    numtyp *t_my;
    numtyp *t_mz;
    numtyp *t_m;
    numtyp h_t_m;
};