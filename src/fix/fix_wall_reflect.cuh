#pragma once
#include "fix/fix.cuh"

class wall_reflect : public Fix {
public:
    wall_reflect(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void preprocess(System& system) override;

    void post_integrate(System& system, unsigned int step) override;

private: 

    int wall_direction;
    numtyp  lo_wall_pos;
    numtyp  hi_wall_pos;
};