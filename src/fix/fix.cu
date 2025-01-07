#include "fix/fix.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

FixRegistry& getFixRegistry() {
    static FixRegistry fixRegistry;
    return fixRegistry;
}

/* ----------------------------------------------------------------------------------------------------------- */

EnabledFixRegistry& getEnabledFixRegistry() {
    static EnabledFixRegistry enabledFixRegistry;
    return enabledFixRegistry;
}

/* ----------------------------------------------------------------------------------------------------------- */

void registerFixType(const std::string& typeName, FixConstructor constructor) 
{
    auto& fixRegistry     = getFixRegistry();
    fixRegistry[typeName] = constructor; 
}

/* ----------------------------------------------------------------------------------------------------------- */

void enableFix(const std::string& id, const std::string& groupid, const std::string& type, const std::vector<std::string>& params, System& system) 
{
    auto& fixRegistry        = getFixRegistry();
    auto& enabledFixRegistry = getEnabledFixRegistry();

    if (enabledFixRegistry.find(id) != enabledFixRegistry.end()) 
    {
        std::string error = "Error: Fix ID '" + id + "' is already registered!";
        print_error({error});
    }

    auto it = fixRegistry.find(type);
    if (it == fixRegistry.end()) 
    {
        std::string error = "Error: Fix type '" + type + "' not found!";
        print_error({error});
    }

    auto groupIt = system.groups.find(groupid);
    if (groupIt == system.groups.end()) 
    {
        std::string error = "Error: Group '" + groupid + "' not found!";
        print_error({error});
    }

    std::unique_ptr<Fix> fixInstance = it->second(id, groupid, params);
    fixInstance->validateParams(params);

    enabledFixRegistry[id] = std::move(fixInstance);
}

/* ----------------------------------------------------------------------------------------------------------- */

void preprocessFixes(System& system) 
{   
    /* ----------------------------------------------------------------------------------------------------------- */

    auto& enabledFixRegistry = getEnabledFixRegistry();  
    enabledFixRegistry.clear();  

    std::istringstream input(system.input);
    std::string line;
    while (std::getline(input, line)) 
    {
        std::istringstream iss(line);
        std::string command, id, groupid, type;
        std::vector<std::string> params;

        iss >> command; 
        if (command == "fix") 
        {
            iss >> id >> groupid >> type; 

            std::string param;
            while (iss >> param) {
                params.push_back(param); 
            }

            enableFix(id, groupid, type, params, system);
        }
    }

    /* ----------------------------------------------------------------------------------------------------------- */

    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                       Fix Information                                      " << std::endl;
    std::cout << "                                                                            " << std::endl;   

    if (enabledFixRegistry.empty()) 
    {
        std::cout << "   No fix is enabled." << std::endl;
    } 
    else 
    {
        for (const auto& it : enabledFixRegistry) 
        {
            const std::string& id = it.first;
            const std::unique_ptr<Fix>& fixInstance = it.second;
            fixInstance->preprocess(system);
            std::cout << "   Enabled fix: " << fixInstance->getName() << " with ID: " << id << std::endl;
        }
    }

    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;

    /* ----------------------------------------------------------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void postprocessFixes(System& system) 
{
    auto& enabledFixRegistry = getEnabledFixRegistry();  

    for (const auto& it : enabledFixRegistry) 
    {
        const std::unique_ptr<Fix>& fixInstance = it.second;
        fixInstance->postprocess(system);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

void Fix_initial_integrate(System& system, unsigned int step) 
{
    auto& enabledFixRegistry = getEnabledFixRegistry();  

    for (const auto& it : enabledFixRegistry) 
    {
        const std::unique_ptr<Fix>& fixInstance = it.second;
        fixInstance->initial_integrate(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

void Fix_final_integrate(System& system, unsigned int step) 
{
    auto& enabledFixRegistry = getEnabledFixRegistry();  

    for (const auto& it : enabledFixRegistry) 
    {
        const std::unique_ptr<Fix>& fixInstance = it.second;
        fixInstance->final_integrate(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

void Fix_post_force(System& system, unsigned int step) 
{
    auto& enabledFixRegistry = getEnabledFixRegistry();  

    for (const auto& it : enabledFixRegistry) 
    {
        const std::unique_ptr<Fix>& fixInstance = it.second;
        fixInstance->post_force(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

void Fix_end_of_step(System& system, unsigned int step) 
{
    auto& enabledFixRegistry = getEnabledFixRegistry();  

    for (const auto& it : enabledFixRegistry) 
    {
        const std::unique_ptr<Fix>& fixInstance = it.second;
        fixInstance->end_of_step(system, step);
    }
}

/* ----------------------------------------------------------------------------------------------------------- */
