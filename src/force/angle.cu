#include "force/angle.cuh"

/* ----------------------------------------------------------------------------------------------------------- */ 

std::vector<std::unique_ptr<Angle>>& getAngleRegistry() 
{
    static std::vector<std::unique_ptr<Angle>> angleRegistry;
    return angleRegistry;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

Angle*& getEnabledAngle() 
{
    static Angle* enabledAngle = nullptr;
    return enabledAngle;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void registerAngle(std::unique_ptr<Angle> instance) 
{
    auto& angleRegistry = getAngleRegistry();  
    angleRegistry.push_back(std::move(instance));
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void preprocessAngles(System& system) 
{
    auto& angleRegistry = getAngleRegistry();  
    Angle*& enabledAngle = getEnabledAngle();   

    for (auto& angleInstance : angleRegistry)
    {
        if (angleInstance->isEnabled(system)) 
        {
            if (enabledAngle != nullptr) 
            {
                std::cout << std::endl;
                std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
                std::cout << "   Error: more than one enabled angle_style found." << std::endl;
                std::cout << "   Please ensure only one angle_style is enabled." << std::endl;
                std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
                std::cout << std::endl;
                exit(1);
            }
            enabledAngle = angleInstance.get();  
        }
    }

    if (enabledAngle == nullptr) 
    {
        std::cout << std::endl;
        std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
        std::cout << "   Error: no enabled angle_style found." << std::endl;
        std::cout << "   Please check the angle_style command format." << std::endl;
        std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
        std::cout << std::endl;
        exit(1);
    }

    std::cout << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                       Angle Interaction Information                         " << std::endl;
    std::cout << std::endl;   
    std::cout << "   Enabled angle_style:  " << enabledAngle->getName() << std::endl;
    enabledAngle->print_angle_info(system);
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << std::endl;
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void Angle_compute(System& system, unsigned int step)
{
    Angle* enabledAngle = getEnabledAngle(); 
    if (enabledAngle) {
        enabledAngle->compute_force(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */ 

void postprocessAngles() 
{
    Angle* enabledAngle = getEnabledAngle();
    enabledAngle = nullptr; 
}

/* ----------------------------------------------------------------------------------------------------------- */ 
