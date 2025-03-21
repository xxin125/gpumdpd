#include "fix/fix_recenter.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

recenter::recenter(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Fix(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string recenter::getName() 
{
    return "recenter";
}

/* ----------------------------------------------------------------------------------------------------------- */

void recenter::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal fix_recenter command";
    std::string format   = "fix ID groupID recenter frequency recenter_x recenter_y recenter_z";
    std::string example0 = "fix md probe   recenter 100 10 10 10";
    std::string example1 = "fix md probe   recenter 100 50 50 50";
    
    if (params.size() != 4) {
        print_error_and_exit("Invalid fix_recenter parameters", error, format, {example0, example1});
    }

    frequency  = parse_int<unsigned int>(params[0], "Invalid fix_recenter parameters", "frequency",  {format, example0, example1});
    recenter_x = parse_float<numtyp>(params[1], "Invalid fix_recenter parameters",     "recenter_x", {format, example0, example1});
    recenter_y = parse_float<numtyp>(params[2], "Invalid fix_recenter parameters",     "recenter_y", {format, example0, example1});
    recenter_z = parse_float<numtyp>(params[3], "Invalid fix_recenter parameters",     "recenter_z", {format, example0, example1});
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

void recenter::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    Box& box = system.box;

    if (recenter_x < box.xlo || recenter_x > box.xhi)
    {
        print_error({"spring position is out of the box in x direction!"});
    }

    if (recenter_y < box.ylo || recenter_y > box.yhi)
    {
        print_error({"spring position is out of the box in y direction!"});
    }

    if (recenter_z < box.zlo || recenter_z > box.zhi)
    {
        print_error({"spring position is out of the box in z direction!"});
    }

    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaMalloc(&t_mx, sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&t_my, sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&t_mz, sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&t_m,  sizeof(numtyp)));
    
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

void recenter::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaFree(t_mx));
    CUDA_CHECK(cudaFree(t_my));
    CUDA_CHECK(cudaFree(t_mz));
    CUDA_CHECK(cudaFree(t_m));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_recenter_com
(
    const int n_gatoms,        const int *g_atoms,   
    const int *d_type,         const numtyp *d_uwpos,
    numtyp *t_mx,              numtyp *t_my,             numtyp *t_mz   
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdatamx[128]; 
    __shared__ numtyp sdatamy[128]; 
    __shared__ numtyp sdatamz[128]; 

    /* ------------------------------------------------------- */

    numtyp p_mx = (numtyp)0.0;
    numtyp p_my = (numtyp)0.0;
    numtyp p_mz = (numtyp)0.0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];
        numtyp mass_i = masses[d_type[atom_index]-1];
        p_mx = mass_i*d_uwpos[atom_index*3+0];
        p_my = mass_i*d_uwpos[atom_index*3+1];
        p_mz = mass_i*d_uwpos[atom_index*3+2];
    }

    sdatamx[tid] = p_mx;
    sdatamy[tid] = p_my;
    sdatamz[tid] = p_mz;
    __syncthreads();

    /* ------------------------------------------------------- */

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sdatamx[tid] += sdatamx[tid + s];
            sdatamy[tid] += sdatamy[tid + s];
            sdatamz[tid] += sdatamz[tid + s];
        }
        __syncthreads();
    }

    /* ------------------------------------------------------- */

    if (tid == 0) 
    {
        atomicAdd(t_mx, sdatamx[0]);
        atomicAdd(t_my, sdatamy[0]);
        atomicAdd(t_mz, sdatamz[0]);
    }   
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_recenter
(
    const int n_gatoms,        const int *g_atoms,   
    numtyp *d_pos,             numtyp *d_uwpos,
    const numtyp lx,           const numtyp ly,         const numtyp lz,
    const numtyp xlo,          const numtyp ylo,        const numtyp zlo,
    const numtyp xhi,          const numtyp yhi,        const numtyp zhi,
    const numtyp dx,           const numtyp dy,     const numtyp dz 
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */

    numtyp pos_x = d_pos[i*3+0] + dx;
    numtyp pos_y = d_pos[i*3+1] + dy;
    numtyp pos_z = d_pos[i*3+2] + dz;

    while (pos_x >= xhi || pos_x < xlo) {
        pos_x -= lx * ((pos_x >= xhi) - (pos_x < xlo));
    }
    while (pos_y >= yhi || pos_y < ylo) {
        pos_y -= ly * ((pos_y >= yhi) - (pos_y < ylo));
    }
    while (pos_z >= zhi || pos_z < zlo) {
        pos_z -= lz * ((pos_z >= zhi) - (pos_z < zlo));
    }

    d_pos[i*3+0] = pos_x;
    d_pos[i*3+1] = pos_y;
    d_pos[i*3+2] = pos_z;

    d_uwpos[i*3+0] += dx;
    d_uwpos[i*3+1] += dy;
    d_uwpos[i*3+2] += dz;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void recenter::end_of_step(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    if (step % frequency != 0) {
        return;
    }
    
    /* ------------------------------------------------------- */

    Box& box     = system.box;
    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    int n_gatoms = group.n_atoms;

    /* ------------------------------------------------------- */

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaMemset(t_mx, (numtyp)0.0, sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(t_my, (numtyp)0.0, sizeof(numtyp)));
    CUDA_CHECK(cudaMemset(t_mz, (numtyp)0.0, sizeof(numtyp)));

    kernel_recenter_com<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,                        
        atoms.d_type,              atoms.d_uwpos,
        t_mx,                      t_my,                    t_mz                  
    );

    /* ------------------------------------------------------- */

    numtyp h_t_mx = (numtyp)0.0;
    numtyp h_t_my = (numtyp)0.0;
    numtyp h_t_mz = (numtyp)0.0;

    CUDA_CHECK(cudaMemcpy(&h_t_mx, t_mx, sizeof(numtyp), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_t_my, t_my, sizeof(numtyp), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_t_mz, t_mz, sizeof(numtyp), cudaMemcpyDeviceToHost));

    h_t_mx = h_t_mx / h_t_m;
    h_t_my = h_t_my / h_t_m;
    h_t_mz = h_t_mz / h_t_m;

    /* ------------------------------------------------------- */

    numtyp dx = recenter_x - h_t_mx;
    numtyp dy = recenter_y - h_t_my;
    numtyp dz = recenter_z - h_t_mz;    

    /* ------------------------------------------------------- */

    kernel_recenter<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,                        
        atoms.d_pos,               atoms.d_uwpos,
        box.lx,                    box.ly,                        box.lz,
        box.xlo,                   box.ylo,                       box.zlo,
        box.xhi,                   box.yhi,                       box.zhi,
        dx,                        dy,                    dz                  
    );

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_FIX(recenter)
///////////////////////////////
