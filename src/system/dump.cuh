#pragma once
#include "main/common.cuh"
#include "system/system.cuh"

class Dump 
{
public:
    void preprocess(System& system);
    void dump(System& system, unsigned int step);
    void postprocess(System& system);

    FILE* dumpfile;
};