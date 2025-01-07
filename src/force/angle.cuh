#pragma once

#include "main/common.cuh"
#include "system/system.cuh"

/* ----------------------------------------------------------------------------------------------------------- */ 

class Angle 
{
public:
    virtual bool isEnabled(System& system) = 0;
    virtual void print_angle_info(System& system) = 0;
    virtual std::string getName() = 0;
    virtual void compute_force(System& system, unsigned int step) = 0;
};

/* ----------------------------------------------------------------------------------------------------------- */ 

std::vector<std::unique_ptr<Angle>>& getAngleRegistry();
Angle*& getEnabledAngle();

/* ----------------------------------------------------------------------------------------------------------- */ 

void registerAngle(std::unique_ptr<Angle> instance);
void preprocessAngles(System& system);
void postprocessAngles();
void Angle_compute(System& system, unsigned int step);

/* ----------------------------------------------------------------------------------------------------------- */ 

#define REGISTER_ANGLE(type) \
    static bool _##type##_registered = [](){ \
        registerAngle(std::make_unique<type>()); \
        return true; \
    }();

/* ----------------------------------------------------------------------------------------------------------- */ 
