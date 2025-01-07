#pragma once
#include "main/common.cuh"
#include "system/system.cuh"

class Neigh_list 
{
public:

    void preprocess(System& system);
    void postprocess(System& system);

    void build(System& system);
    void check_update(System& system);
    
private:    

    void neigh_mem_alloc1(System& system);
    void initialize_boxes(System& system);
    void print_neigh_info(System& system);
    void neigh_mem_alloc2(System& system);
    void get_neighborbins(System& system);

    void check_update_pre(System& system);
    void check_update_post(System& system);
};
