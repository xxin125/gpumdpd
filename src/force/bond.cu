#include "force/bond.cuh"

/* ----------------------------------------------------------------------------------------------------------- */ 

std::vector<std::unique_ptr<Bond>>& getBondRegistry() 
{
    static std::vector<std::unique_ptr<Bond>> bondRegistry;
    return bondRegistry;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

Bond*& getEnabledBond() 
{
    static Bond* enabledBond = nullptr;
    return enabledBond;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void registerBond(std::unique_ptr<Bond> instance) 
{
    auto& bondRegistry = getBondRegistry();  
    bondRegistry.push_back(std::move(instance));
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void preprocessBonds(System& system) 
{
    auto& bondRegistry = getBondRegistry();  
    Bond*& enabledBond = getEnabledBond();   

    for (auto& bondInstance : bondRegistry)
    {
        if (bondInstance->isEnabled(system)) 
        {
            if (enabledBond != nullptr) 
            {
                std::cout << std::endl;
                std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
                std::cout << "   Error: more than one enabled bond_style found." << std::endl;
                std::cout << "   Please ensure only one bond_style is enabled." << std::endl;
                std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
                std::cout << std::endl;
                exit(1);
            }
            enabledBond = bondInstance.get();  
        }
    }

    if (enabledBond == nullptr) 
    {
        std::cout << std::endl;
        std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
        std::cout << "   Error: no enabled bond_style found." << std::endl;
        std::cout << "   Please check the bond_style command format." << std::endl;
        std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
        std::cout << std::endl;
        exit(1);
    }

    std::cout << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                       Bond Interaction Information                         " << std::endl;
    std::cout << std::endl;   
    std::cout << "   Enabled bond_style:  " << enabledBond->getName() << std::endl;
    enabledBond->print_bond_info(system);
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void Bond_compute(System& system, unsigned int step)
{
    Bond* enabledBond = getEnabledBond(); 
    if (enabledBond) {
        enabledBond->compute_force(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void postprocessBonds() 
{
    Bond* enabledBond = getEnabledBond();
    enabledBond = nullptr; 
}

/* ----------------------------------------------------------------------------------------------------------- */ 
