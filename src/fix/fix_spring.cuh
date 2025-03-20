#pragma once
#include "fix/fix.cuh"

class spring : public Fix {
public:
    spring(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void preprocess(System& system) override;
    void postprocess(System& system) override;

    void post_force(System& system, unsigned int step) override;

private: 

    numtyp spring_k;
    numtyp spring_x;
    numtyp spring_y;
    numtyp spring_z;
    numtyp spring_r0;

    int    file_flag;
    std::string filename;
    unsigned int  frequency;
    std::ofstream file;

    numtyp *t_mx;
    numtyp *t_my;
    numtyp *t_mz;
    numtyp *t_m;
    numtyp h_t_m;
};