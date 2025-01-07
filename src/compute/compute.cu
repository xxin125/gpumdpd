#include "compute/compute.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

ComputeRegistry& getComputeRegistry() {
    static ComputeRegistry computeRegistry;
    return computeRegistry;
}

/* ----------------------------------------------------------------------------------------------------------- */

EnabledComputeRegistry& getEnabledComputeRegistry() {
    static EnabledComputeRegistry enabledComputeRegistry;
    return enabledComputeRegistry;
}

/* ----------------------------------------------------------------------------------------------------------- */

void registerComputeType(const std::string& typeName, ComputeConstructor constructor) 
{
    auto& computeRegistry     = getComputeRegistry();
    computeRegistry[typeName] = constructor; 
}

/* ----------------------------------------------------------------------------------------------------------- */

void enableCompute(const std::string& id, const std::string& groupid, const std::string& type, const std::vector<std::string>& params, System& system) 
{
    auto& computeRegistry        = getComputeRegistry();
    auto& enabledComputeRegistry = getEnabledComputeRegistry();

    if (enabledComputeRegistry.find(id) != enabledComputeRegistry.end()) 
    {
        std::string error = "Error: Compute ID '" + id + "' is already registered!";
        print_error({error});
    }

    auto it = computeRegistry.find(type);
    if (it == computeRegistry.end()) 
    {
        std::string error = "Error: Compute type '" + type + "' not found!";
        print_error({error});
    }

    auto groupIt = system.groups.find(groupid);
    if (groupIt == system.groups.end()) 
    {
        std::string error = "Error: Group '" + groupid + "' not found!";
        print_error({error});
    }

    std::unique_ptr<Compute> computeInstance = it->second(id, groupid, params);
    computeInstance->validateParams(params);

    enabledComputeRegistry[id] = std::move(computeInstance);
}

/* ----------------------------------------------------------------------------------------------------------- */

void preprocessCompute(System& system) 
{   
    /* ----------------------------------------------------------------------------------------------------------- */

    auto& enabledComputeRegistry = getEnabledComputeRegistry();  
    enabledComputeRegistry.clear();  

    std::istringstream input(system.input);
    std::string line;
    while (std::getline(input, line)) 
    {
        std::istringstream iss(line);
        std::string command, id, groupid, type;
        std::vector<std::string> params;

        iss >> command; 
        if (command == "compute") 
        {
            iss >> id >> groupid >> type; 

            std::string param;
            while (iss >> param) {
                params.push_back(param); 
            }

            enableCompute(id, groupid, type, params, system);
        }
    }

    /* ----------------------------------------------------------------------------------------------------------- */

    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                       Compute Information                                      " << std::endl;
    std::cout << "                                                                            " << std::endl;   

    if (enabledComputeRegistry.empty()) 
    {
        std::cout << "   No compute is enabled." << std::endl;
    } 
    else 
    {
        for (const auto& it : enabledComputeRegistry) 
        {
            const std::string& id = it.first;
            const std::unique_ptr<Compute>& computeInstance = it.second;
            computeInstance->preprocess(system);
            std::cout << "   Enabled compute: " << computeInstance->getName() << " with ID: " << id << std::endl;
        }
    }

    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;

    /* ----------------------------------------------------------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void postprocessCompute(System& system) 
{
    auto& enabledComputeRegistry = getEnabledComputeRegistry();  

    for (const auto& it : enabledComputeRegistry) 
    {
        const std::unique_ptr<Compute>& computeInstance = it.second;
        computeInstance->postprocess(system);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

void Compute_compute(System& system, unsigned int step) 
{
    auto& enabledComputeRegistry = getEnabledComputeRegistry();  

    for (const auto& it : enabledComputeRegistry) 
    {
        const std::unique_ptr<Compute>& computeInstance = it.second;
        computeInstance->compute(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */
