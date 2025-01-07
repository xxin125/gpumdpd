#pragma once
#include "fix/fix.cuh"

class nve : public Fix {
public:
    nve(std::string id, std::string gid, const std::vector<std::string>& params);

    void validateParams(const std::vector<std::string>& params) override;
    std::string getName() override;

    void initial_integrate(System& system, unsigned int step) override;
    void final_integrate(System& system, unsigned int step) override;
};