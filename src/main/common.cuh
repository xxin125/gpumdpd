#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <cmath>
#include <random>

#include <algorithm>
#include <iomanip> 
#include <cctype>
#include <functional>
#include <memory>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <set>
#include <unordered_set>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>


/* ----------------------------------------------------------------------------------------------------------- */

// precision options
// 1. single precision
// 2. double precision

#if   defined(SINGLE)
using numtyp = float;
#elif defined(DOUBLE)
using numtyp = double;
#endif

/* ----------------------------------------------------------------------------------------------------------- */

#define CUDA_CHECK(call)                              \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

/* ----------------------------------------------------------------------------------------------------------- */

// device constant memory 

extern  __constant__ numtyp masses[256];
extern  __constant__ numtyp gpu_pair_coeff[256];
extern  __constant__ numtyp gpu_bond_coeff[256];
extern  __constant__ numtyp gpu_angle_coeff[256];

/* ----------------------------------------------------------------------------------------------------------- */

// run parameters 

struct Run_p
{   
    /* ----------------------- */

    // neighbor command

    numtyp global_cut; 
    numtyp skin; 
    numtyp max_rho;
    int    nl_f; 

    /* ----------------------- */

    // timestep command

    numtyp dt;

    /* ----------------------- */

    // thermo command

    unsigned int log_f;

    /* ----------------------- */

    // run command

    unsigned int nsteps;    

    /* ----------------------- */

    // read_data command 
    
    std::string input_data_name;

    /* ----------------------- */

    // write_data command 

    bool write_data_flag = false;    
    std::string out_data_name;
    
    /* ----------------------- */

    // dump command

    bool write_dump_flag = false;    
    unsigned int dump_start;
    unsigned int dump_f;
    bool wrapped_flag  = false;
    bool dumpforce_flag = false;
    bool dumpvel_flag = false;

    /* ----------------------- */   

};

/* ----------------------------------------------------------------------------------------------------------- */

// thermo parameters 

struct Thermo_p
{
    /* ----------------------- */

    // temperature  

    numtyp thermo_temp;
    numtyp *d_total_ke;

    /* ----------------------- */

    // total pair energy  

    numtyp thermo_pair_pe;
    numtyp *d_total_pair_pe;

    // total bond energy   

    numtyp thermo_bond_pe;
    numtyp *d_total_bond_pe;

    // total bond energy   

    numtyp thermo_angle_pe;
    numtyp *d_total_angle_pe;

    /* ----------------------- */

    // pressure   

    numtyp thermo_pressure;
    std::vector<numtyp> pressure_tensor;
    numtyp *d_pressure_tensor;

  /* ----------------------- */
};

/* ----------------------------------------------------------------------------------------------------------- */

inline void print_error_and_exit(const std::string& line, const std::string& error, const std::string& format, const std::vector<std::string>& examples) 
{
    std::cout << "\n/* ---------------------------------------------------------------------- */\n";
    std::cout << "                                    ERROR                          " << std::endl;
    std::cout << "   E_Line:   " << line   << std::endl;
    std::cout << "   Error:    " << error  << std::endl;
    std::cout << "   Format:   " << format << std::endl;
    for (const auto& example : examples) {
        std::cout << "   Example:  " << example << std::endl;
    }
    std::cout << "/* ---------------------------------------------------------------------- */\n\n";
    exit(1);
}

/* ----------------------------------------------------------------------------------------------------------- */

inline void print_error(const std::vector<std::string>& errors) 
{
    std::cout << "\n/* ---------------------------------------------------------------------- */\n";
    std::cout << "                                    ERROR                          " << std::endl;
    for (const auto& error : errors) {
        std::cout << "   Error:  " << error << std::endl;
    }
    std::cout << "/* ---------------------------------------------------------------------- */\n\n";
    exit(1);
}

/* ----------------------------------------------------------------------------------------------------------- */

template <typename T>
struct FloatParser;

/* -------------------------------------------------------- */

template <>
struct FloatParser<float> {
    static float parse(const std::string& token) {
        return std::stof(token);
    }
};

/* -------------------------------------------------------- */

template <>
struct FloatParser<double> {
    static double parse(const std::string& token) {
        return std::stod(token);
    }
};

/* -------------------------------------------------------- */

template <typename T>
inline T parse_float(const std::string& token, const std::string& line, const std::string& error_context, const std::vector<std::string>& examples) 
{
    T value = static_cast<T>(0.0);
    try {
        value = FloatParser<T>::parse(token);
    } catch (const std::invalid_argument& e) {
        std::string error = "Invalid number value provided for " + error_context;
        std::string format = "Expected a floating-point number";
        print_error_and_exit(line, error, format, examples); 
    } catch (const std::out_of_range& e) {
        std::string error = "Number value out of range for " + error_context;
        std::string format = "The number is too large or too small";
        print_error_and_exit(line, error, format, examples);
    }
    return value;
}

/* ----------------------------------------------------------------------------------------------------------- */

template <typename T>
struct IntParser;

/* -------------------------------------------------------- */

template <>
struct IntParser<int> {
    static int parse(const std::string& token, size_t& pos) {
        return std::stoi(token, &pos);
    }
};

/* -------------------------------------------------------- */

template <>
struct IntParser<unsigned int> {
    static unsigned int parse(const std::string& token, size_t& pos) {
        unsigned long long temp_value = std::stoull(token, &pos);
        if (temp_value > std::numeric_limits<unsigned int>::max()) {
            throw std::out_of_range("Unsigned int value out of range");
        }
        return static_cast<unsigned int>(temp_value);
    }
};

/* -------------------------------------------------------- */

template <typename T>
inline T parse_int(const std::string& token, const std::string& line, const std::string& error_context, const std::vector<std::string>& examples) 
{
    T value = 0;
    try {
        size_t pos;
        value = IntParser<T>::parse(token, pos);
        if (pos != token.size()) { 
            std::string error  = "Expected integer but got non-integer value for " + error_context;
            std::string format = "Please provide a valid integer";
            print_error_and_exit(line, error, format, examples);
        }
    } catch (const std::invalid_argument& e) {
        std::string error  = "Invalid integer value provided for " + error_context + ". Non-numeric value found.";
        std::string format = "An integer must only contain digits";
        print_error_and_exit(line, error, format, examples);
    } catch (const std::out_of_range& e) {
        std::string error  = "Integer value out of range for " + error_context;
        std::string format = "The number is too large or too small to be stored as an int";
        print_error_and_exit(line, error, format, examples);
    }
    return value;
}

/* ----------------------------------------------------------------------------------------------------------- */