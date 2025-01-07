#include "force/pair.cuh"

/* ----------------------------------------------------------------------------------------------------------- */ 

std::vector<std::unique_ptr<Pair>>& getPairRegistry() 
{
    static std::vector<std::unique_ptr<Pair>> pairRegistry;
    return pairRegistry;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

Pair*& getEnabledPair() 
{
    static Pair* enabledPair = nullptr;
    return enabledPair;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void registerPair(std::unique_ptr<Pair> instance) 
{
    auto& pairRegistry = getPairRegistry();  
    pairRegistry.push_back(std::move(instance));  
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void preprocessPairs(System& system) 
{
    auto& pairRegistry = getPairRegistry();  
    Pair*& enabledPair = getEnabledPair();   

    enabledPair = nullptr;  

    for (auto& pairInstance : pairRegistry)
    {
        if (pairInstance->isEnabled(system)) 
        {
            if (enabledPair != nullptr) 
            {
                std::cout << std::endl;
                std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
                std::cout << "   Error: more than one enabled pair_style found." << std::endl;
                std::cout << "   Please ensure only one pair_style is enabled." << std::endl;
                std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
                std::cout << std::endl;
                exit(1);
            }
            enabledPair = pairInstance.get();  
        }
    }

    if (enabledPair == nullptr) 
    {
        std::cout << std::endl;
        std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
        std::cout << "   Error: no enabled pair_style found." << std::endl;
        std::cout << "   Please check the pair_style command format." << std::endl;
        std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
        std::cout << std::endl;
        exit(1);
    }

    std::cout << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                       Pair Interaction Information                         " << std::endl;
    std::cout << std::endl;   
    std::cout << "   Enabled pair_style:  " << enabledPair->getName() << std::endl;
    enabledPair->print_pair_info(system);
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void Pair_compute(System& system, unsigned int step)
{
    Pair* enabledPair = getEnabledPair();  
    if (enabledPair) {
        enabledPair->compute_force(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void postprocessPairs() 
{
    Pair*& enabledPair = getEnabledPair();
    enabledPair = nullptr; 
}

/* ----------------------------------------------------------------------------------------------------------- */ 
