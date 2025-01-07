#pragma once
#include "force/angle.cuh"

class Angle_harmonic : public Angle 
{
public:

    Angle_harmonic();

    bool isEnabled(System& system) override;
    void print_angle_info(System& system) override;
    std::string getName() override;
    void compute_force(System& system, unsigned int step) override;

private:

    // angle_style harmonic

    std::string angle_style_name;

    // angle_coeff type k degree

    std::vector<numtyp> angle_coeff;

    // got_angle_coeff

    std::vector<int> got_angle_coeff;
};

