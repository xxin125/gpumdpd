#include "fix/fix_spring.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

spring::spring(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Fix(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string spring::getName() 
{
    return "spring";
}

/* ----------------------------------------------------------------------------------------------------------- */

void spring::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal fix_spring command";
    std::string format   = "fix ID groupID spring spring_k spring_x spring_y spring_z spring_r0 file_name(optional) file_frequency(optional)";
    std::string example0 = "fix md probe   spring 100 50 50 50 0";
    std::string example1 = "fix md probe   spring 100 50 50 50 10 force.txt 100";
    
    if (params.size() != 5 && params.size() != 7) {
        print_error_and_exit("Invalid fix_spring parameters", error, format, {example0, example1});
    }

    spring_k  = parse_float<numtyp>(params[0], "Invalid fix_spring parameters", "spring_k",  {format, example0, example1});
    spring_x  = parse_float<numtyp>(params[1], "Invalid fix_spring parameters", "spring_x",  {format, example0, example1});
    spring_y  = parse_float<numtyp>(params[2], "Invalid fix_spring parameters", "spring_y",  {format, example0, example1});
    spring_z  = parse_float<numtyp>(params[3], "Invalid fix_spring parameters", "spring_z",  {format, example0, example1});
    spring_r0 = parse_float<numtyp>(params[4], "Invalid fix_spring parameters", "spring_r0", {format, example0, example1});

    file_flag = 0;
    if (params.size() == 7) {
        file_flag = 1;
        filename  = params[5];
        frequency = parse_int<unsigned int>(params[6], "Invalid fix_spring parameters", "file_frequency",  {format, example0, example1});
    }
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

void spring::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    Box& box = system.box;

    if (spring_x < box.xlo || spring_x > box.xhi)
    {
        print_error({"spring position is out of the box in x direction!"});
    }

    if (spring_y < box.ylo || spring_y > box.yhi)
    {
        print_error({"spring position is out of the box in y direction!"});
    }

    if (spring_z < box.zlo || spring_z > box.zhi)
    {
        print_error({"spring position is out of the box in z direction!"});
    }

    /* ------------------------------------------------------- */

    // file spring

    if (file_flag == 1)
    {
        file.open(filename);

        file    << std::setw(15) << "step"           << "\t\t"
                << std::setw(15) << "spring_force_x" << "\t\t" 
                << std::setw(15) << "spring_force_y" << "\t\t"  
                << std::setw(15) << "spring_force_z" << "\t\t" 
                << std::setw(15) << "spring_force"   << "\t\t" 
                << std::setw(15) << "spring_erergy"  << std::endl;
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

void spring::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file spring
    if (file_flag == 1) {
        file.close();
    }

    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaFree(t_mx));
    CUDA_CHECK(cudaFree(t_my));
    CUDA_CHECK(cudaFree(t_mz));
    CUDA_CHECK(cudaFree(t_m));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_spring_com
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

    numtyp p_mvx = (numtyp)0.0;
    numtyp p_mvy = (numtyp)0.0;
    numtyp p_mvz = (numtyp)0.0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];
        numtyp mass_i = masses[d_type[atom_index]-1];
        p_mvx = mass_i*d_uwpos[atom_index*3+0];
        p_mvy = mass_i*d_uwpos[atom_index*3+1];
        p_mvz = mass_i*d_uwpos[atom_index*3+2];
    }

    sdatamx[tid] = p_mvx;
    sdatamy[tid] = p_mvy;
    sdatamz[tid] = p_mvz;
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

static __global__ void kernel_spring_force
(
    const int n_gatoms,        const int *g_atoms,   
    const int *d_type,         numtyp *d_force,
    const numtyp f_preatom_x,  const numtyp f_preatom_y,      const numtyp f_preatom_z
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */

    numtyp mass_i = masses[d_type[i]-1];
    d_force[i*3+0] += f_preatom_x * mass_i;
    d_force[i*3+1] += f_preatom_y * mass_i;
    d_force[i*3+2] += f_preatom_z * mass_i;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void spring::post_force(System& system, unsigned int step) 
{
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

    kernel_spring_com<<<numBlocks, blockSize>>>
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

    while (h_t_mx >= box.xhi || h_t_mx < box.xlo) {
        h_t_mx -= box.lx * ((h_t_mx >= box.xhi) - (h_t_mx < box.xlo));
    }
    while (h_t_my >= box.yhi || h_t_my < box.ylo) {
        h_t_my -= box.ly * ((h_t_my >= box.yhi) - (h_t_my < box.ylo));
    }
    while (h_t_mz >= box.zhi || h_t_mz < box.zlo) {
        h_t_mz -= box.lz * ((h_t_mz >= box.zhi) - (h_t_mz < box.zlo));
    }

    /* ------------------------------------------------------- */

    numtyp dx = h_t_mx - spring_x;
    numtyp dy = h_t_my - spring_y;
    numtyp dz = h_t_mz - spring_z;
    const numtyp r2 = dx * dx + dy * dy + dz * dz;
    numtyp r = sqrt(r2);

    numtyp spring_fx = 0.0;
    numtyp spring_fy = 0.0;
    numtyp spring_fz = 0.0;
    numtyp spring_f  = 0.0;
    numtyp spring_e  = 0.0;
    numtyp dr        = r - spring_r0;

    if (r > 1.0e-10) {
        spring_fx = (numtyp)(-spring_k * dr * dx / r);
        spring_fy = (numtyp)(-spring_k * dr * dy / r);
        spring_fz = (numtyp)(-spring_k * dr * dz / r);

        spring_f  = sqrt(spring_fx*spring_fx + spring_fy*spring_fy + spring_fz*spring_fz);
        if (dr < 0.0) spring_f = -spring_f;

        spring_e  = (numtyp)(0.5 * spring_k * dr * dr);
    }

    if (file_flag == 1) {
        if (step % frequency == 0) {
            file    << std::setw(15) << step             << "\t\t"
                    << std::setw(15) << spring_fx        << "\t\t" 
                    << std::setw(15) << spring_fy        << "\t\t"  
                    << std::setw(15) << spring_fz        << "\t\t" 
                    << std::setw(15) << spring_f         << "\t\t" 
                    << std::setw(15) << spring_e         << std::endl;    
        }
    }

    numtyp f_preatom_x = (numtyp)(spring_fx / h_t_m);  
    numtyp f_preatom_y = (numtyp)(spring_fy / h_t_m);
    numtyp f_preatom_z = (numtyp)(spring_fz / h_t_m);

    /* ------------------------------------------------------- */

    kernel_spring_force<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,                        
        atoms.d_type,              atoms.d_force, 
        f_preatom_x,               f_preatom_y,                  f_preatom_z                  
    );

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_FIX(spring)
///////////////////////////////
