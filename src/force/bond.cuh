#pragma once

#include "main/common.cuh"
#include "system/system.cuh"

/* ----------------------------------------------------------------------------------------------------------- */ 

class Bond 
{
public:
    virtual bool isEnabled(System& system) = 0;
    virtual void print_bond_info(System& system) = 0;
    virtual std::string getName() = 0;
    virtual void compute_force(System& system, unsigned int step) = 0;
};

/* ----------------------------------------------------------------------------------------------------------- */ 

std::vector<std::unique_ptr<Bond>>& getBondRegistry();
Bond*& getEnabledBond();

/* ----------------------------------------------------------------------------------------------------------- */ 

void registerBond(std::unique_ptr<Bond> instance);
void preprocessBonds(System& system);
void postprocessBonds();
void Bond_compute(System& system, unsigned int step);

/* ----------------------------------------------------------------------------------------------------------- */ 

#define REGISTER_BOND(type) \
    static bool _##type##_registered = [](){ \
        registerBond(std::make_unique<type>()); \
        return true; \
    }();

/* ----------------------------------------------------------------------------------------------------------- */ 
