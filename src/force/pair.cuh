#pragma once

#include "main/common.cuh"
#include "system/system.cuh"

/* ----------------------------------------------------------------------------------------------------------- */ 

class Pair 
{
public:
    virtual bool isEnabled(System& system) = 0;
    virtual void print_pair_info(System& system) = 0;
    virtual std::string getName() = 0;
    virtual void compute_force(System& system, unsigned int step) = 0;
};

/* ----------------------------------------------------------------------------------------------------------- */ 

std::vector<std::unique_ptr<Pair>>& getPairRegistry();
Pair*& getEnabledPair();

/* ----------------------------------------------------------------------------------------------------------- */ 

void registerPair(std::unique_ptr<Pair> instance);
void preprocessPairs(System& system);
void postprocessPairs();
void Pair_compute(System& system, unsigned int step);

/* ----------------------------------------------------------------------------------------------------------- */ 

#define REGISTER_PAIR(type) \
    static bool _##type##_registered = [](){ \
        registerPair(std::make_unique<type>()); \
        return true; \
    }();

/* ----------------------------------------------------------------------------------------------------------- */ 
