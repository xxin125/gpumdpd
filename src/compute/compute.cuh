#pragma once
#include "main/common.cuh"
#include "system/system.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

class Compute 
{
public:
    std::string computeID;
    std::string group_id;
    
    Compute(std::string id, std::string gid) : computeID(id), group_id(gid) {}

    virtual void validateParams(const std::vector<std::string>& params) = 0;
    virtual std::string getName() = 0;

    virtual void preprocess(System& system) {};
    virtual void postprocess(System& system) {};

    virtual void compute(System& system, unsigned int step) {};
};

/* ----------------------------------------------------------------------------------------------------------- */

using ComputeConstructor = std::unique_ptr<Compute>(*)(const std::string&, const std::string&, const std::vector<std::string>&);
using ComputeRegistry = std::unordered_map<std::string, ComputeConstructor>;
using EnabledComputeRegistry = std::unordered_map<std::string, std::unique_ptr<Compute>>;

ComputeRegistry& getComputeRegistry();
EnabledComputeRegistry& getEnabledComputeRegistry();

/* ----------------------------------------------------------------------------------------------------------- */

void registerComputeType(const std::string& typeName, ComputeConstructor constructor);
void enableCompute(const std::string& id, const std::string& groupid, const std::string& type, const std::vector<std::string>& params, System& system);

void preprocessCompute(System& system);
void postprocessCompute(System& system);

void Compute_compute(System& system, unsigned int step);

/* ----------------------------------------------------------------------------------------------------------- */

#define REGISTER_COMPUTE(type) \
    static bool _##type##_registered = [](){ \
        registerComputeType(#type, [](const std::string& id, const std::string& groupid, const std::vector<std::string>& params) -> std::unique_ptr<Compute> { \
            return std::unique_ptr<Compute>(static_cast<Compute*>(std::make_unique<type>(id, groupid, params).release())); \
        }); \
        return true; \
    }();
    
/* ----------------------------------------------------------------------------------------------------------- */

