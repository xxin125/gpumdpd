#include "fix/fix_rvcm.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

rvcm::rvcm(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Fix(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string rvcm::getName() 
{
    return "rvcm";
}

/* ----------------------------------------------------------------------------------------------------------- */

void rvcm::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal fix_rvcm command";
    std::string format   = "fix ID groupID rvcm frequency x_flag(0/1) y_flag(0/1) z_flag(0/1)";
    std::string example0 = "fix md probe   rvcm 100 1 1 1";
    std::string example1 = "fix md probe   rvcm 100 0 0 1";
    
    if (params.size() != 4) {
        print_error_and_exit("Invalid fix_rvcm parameters", error, format, {example0, example1});
    }

    frequency = parse_int<unsigned int>(params[0], "Invalid fix_rvcm parameters", "frequency", {format, example0, example1});
    rvcm_x    = parse_int<int>(params[1], "Invalid fix_rvcm parameters", "x_flag", {format, example0, example1});
    rvcm_y    = parse_int<int>(params[2], "Invalid fix_rvcm parameters", "y_flag", {format, example0, example1});
    rvcm_z    = parse_int<int>(params[3], "Invalid fix_rvcm parameters", "z_flag", {format, example0, example1});
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_tm
(
    const int n_gatoms,        const int *g_atoms,   
    const int *d_type,         numtyp *t_m
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdata[128]; 

    /* ------------------------------------------------------- */

    numtyp p_m = (numtyp)0.0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];
        p_m = masses[d_type[atom_index]-1];
    }

    sdata[tid] = p_m;
    __syncthreads();

    /* ------------------------------------------------------- */

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* ------------------------------------------------------- */

    if (tid == 0) 
    {
        atomicAdd(t_m, sdata[0]);
    }  
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void rvcm::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaMalloc(&t_mvx, sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&t_mvy, sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&t_mvz, sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&t_m,   sizeof(numtyp)));
    
    /* ------------------------------------------------------- */

    // total mass 

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    int n_gatoms = group.n_atoms;

    /* ------------------------------------------------------- */

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaMemset(t_m, (numtyp)0.0, sizeof(numtyp)));

    kernel_tm<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,                        
        atoms.d_type,              t_m                 
    );

    /* ------------------------------------------------------- */

    h_t_m = (numtyp)0.0;
    CUDA_CHECK(cudaMemcpy(&h_t_m, t_m, sizeof(numtyp), cudaMemcpyDeviceToHost));

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void rvcm::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaFree(t_mvx));
    CUDA_CHECK(cudaFree(t_mvy));
    CUDA_CHECK(cudaFree(t_mvz));
    CUDA_CHECK(cudaFree(t_m));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_vcm
(
    const int n_gatoms,        const int *g_atoms,   
    const int *d_type,         const numtyp *d_vel,
    numtyp *t_mvx,             numtyp *t_mvy,             numtyp *t_mvz   
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdatamvx[128]; 
    __shared__ numtyp sdatamvy[128]; 
    __shared__ numtyp sdatamvz[128]; 

    /* ------------------------------------------------------- */

    numtyp p_mvx = (numtyp)0.0;
    numtyp p_mvy = (numtyp)0.0;
    numtyp p_mvz = (numtyp)0.0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];
        numtyp mass_i = masses[d_type[atom_index]-1];
        p_mvx = mass_i*d_vel[atom_index*3+0];
        p_mvy = mass_i*d_vel[atom_index*3+1];
        p_mvz = mass_i*d_vel[atom_index*3+2];
    }

    sdatamvx[tid] = p_mvx;
    sdatamvy[tid] = p_mvy;
    sdatamvz[tid] = p_mvz;
    __syncthreads();

    /* ------------------------------------------------------- */

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sdatamvx[tid] += sdatamvx[tid + s];
            sdatamvy[tid] += sdatamvy[tid + s];
            sdatamvz[tid] += sdatamvz[tid + s];
        }
        __syncthreads();
    }

    /* ------------------------------------------------------- */

    if (tid == 0) 
    {
        atomicAdd(t_mvx, sdatamvx[0]);
        atomicAdd(t_mvy, sdatamvy[0]);
        atomicAdd(t_mvz, sdatamvz[0]);
    }  
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_rvcm
(
    const int n_gatoms,        const int *g_atoms,   
    numtyp *d_vel,
    const numtyp t_mvx,        const numtyp t_mvy,      const numtyp t_mvz,
    const int rvcm_x,          const int rvcm_y,        const int rvcm_z
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */

    d_vel[i*3+0] -= rvcm_x * t_mvx;
    d_vel[i*3+1] -= rvcm_y * t_mvy;
    d_vel[i*3+2] -= rvcm_z * t_mvz;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void rvcm::end_of_step(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    if (step % frequency != 0) {
        return;
    }
    
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    int n_gatoms = group.n_atoms;

    /* ------------------------------------------------------- */

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaMemset(t_mvx, (numtyp)0.0, sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(t_mvy, (numtyp)0.0, sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(t_mvz, (numtyp)0.0, sizeof(numtyp)));

    kernel_vcm<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,                        
        atoms.d_type,              atoms.d_vel,
        t_mvx,                     t_mvy,                   t_mvz                  
    );

    /* ------------------------------------------------------- */

    numtyp h_t_mvx = (numtyp)0.0;
    numtyp h_t_mvy = (numtyp)0.0;
    numtyp h_t_mvz = (numtyp)0.0;

    CUDA_CHECK(cudaMemcpy(&h_t_mvx, t_mvx, sizeof(numtyp), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_t_mvy, t_mvy, sizeof(numtyp), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_t_mvz, t_mvz, sizeof(numtyp), cudaMemcpyDeviceToHost));

    h_t_mvx = h_t_mvx / h_t_m;
    h_t_mvy = h_t_mvy / h_t_m;
    h_t_mvz = h_t_mvz / h_t_m;

    /* ------------------------------------------------------- */

    kernel_rvcm<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,                        
        atoms.d_vel,
        h_t_mvx,                   h_t_mvy,                   h_t_mvz,
        rvcm_x,                    rvcm_y,                    rvcm_z                  
    );

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_FIX(rvcm)
///////////////////////////////
