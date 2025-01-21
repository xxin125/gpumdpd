#pragma once
#include "fix/fix.cuh"

class rvcm : public Fix {
public:
    rvcm(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void preprocess(System& system) override;
    void postprocess(System& system) override;

    void end_of_step(System& system, unsigned int step) override;

private: 
    int frequency;
    int rvcm_x;
    int rvcm_y;
    int rvcm_z;

    numtyp *t_mvx;
    numtyp *t_mvy;
    numtyp *t_mvz;
    numtyp *t_m;
    numtyp h_t_m;
};