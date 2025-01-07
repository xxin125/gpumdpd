#pragma once
#include "fix/fix.cuh"

class move : public Fix {
public:
    move(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void initial_integrate(System& system, unsigned int step) override;

private: 
    numtyp v_x;
    numtyp v_y;
    numtyp v_z;
};