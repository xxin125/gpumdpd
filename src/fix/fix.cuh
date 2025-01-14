#pragma once
#include "main/common.cuh"
#include "system/system.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

class Fix 
{
public:
    std::string fixID;
    std::string group_id;
    
    Fix(std::string id, std::string gid) : fixID(id), group_id(gid) {}

    virtual void validateParams(const std::vector<std::string>& params) = 0;
    virtual std::string getName() = 0;

    virtual void preprocess(System& system) {};
    virtual void postprocess(System& system) {};

    virtual void initial_integrate(System& system, unsigned int step) {};
    virtual void post_integrate(System& system, unsigned int step) {};
    virtual void final_integrate(System& system, unsigned int step) {};
    virtual void post_force(System& system, unsigned int step) {};
    virtual void end_of_step(System& system, unsigned int step) {};
};

/* ----------------------------------------------------------------------------------------------------------- */

using FixConstructor = std::unique_ptr<Fix>(*)(const std::string&, const std::string&, const std::vector<std::string>&);
using FixRegistry = std::unordered_map<std::string, FixConstructor>;
using EnabledFixRegistry = std::unordered_map<std::string, std::unique_ptr<Fix>>;

FixRegistry& getFixRegistry();
EnabledFixRegistry& getEnabledFixRegistry();

/* ----------------------------------------------------------------------------------------------------------- */

void registerFixType(const std::string& typeName, FixConstructor constructor);
void enableFix(const std::string& id, const std::string& groupid, const std::string& type, const std::vector<std::string>& params, System& system);

void preprocessFixes(System& system);
void postprocessFixes(System& system);

void Fix_initial_integrate(System& system, unsigned int step);
void Fix_post_integrate(System& system, unsigned int step);
void Fix_final_integrate(System& system, unsigned int step);
void Fix_post_force(System& system, unsigned int step);
void Fix_end_of_step(System& system, unsigned int step);

/* ----------------------------------------------------------------------------------------------------------- */

#define REGISTER_FIX(type) \
    static bool _##type##_registered = [](){ \
        registerFixType(#type, [](const std::string& id, const std::string& groupid, const std::vector<std::string>& params) -> std::unique_ptr<Fix> { \
            return std::unique_ptr<Fix>(static_cast<Fix*>(std::make_unique<type>(id, groupid, params).release())); \
        }); \
        return true; \
    }();
    
/* ----------------------------------------------------------------------------------------------------------- */

