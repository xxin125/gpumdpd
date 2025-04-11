#include "compute/compute_msd.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

msd::msd(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Compute(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string msd::getName() 
{
    return "msd";
}

/* ----------------------------------------------------------------------------------------------------------- */

void msd::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal compute_msd command";
    std::string format   = "compute ID   groupID msd frequency filename";
    std::string example0 = "compute cmsd liquid  msd 1 msd.txt";

    if (params.size() != 2) {
        print_error_and_exit("Invalid compute_msd parameters", error, format, {example0});
    }

    frequency = parse_int<unsigned int>(params[0], "Invalid compute_msd parameters", "frequency", {format,example0});
    filename  = params[1];
}

/* ----------------------------------------------------------------------------------------------------------- */

void msd::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    Atoms& atoms = system.atoms;
    int N        = system.n_atoms;

    /* ------------------------------------------------------- */

    // file msd 

    file.open(filename);

    file << std::setw(15) << "step" << "\t\t"
         << std::setw(15) << "msd"  << std::endl;

    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaMalloc(&ini_uwpos, sizeof(numtyp) * N * 3));
    CUDA_CHECK(cudaMemcpy(ini_uwpos,  atoms.h_uwpos.data(), sizeof(numtyp) * N * 3, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_msd,     sizeof(numtyp)));

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void msd::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file msd 

    file.close();

    /* ------------------------------------------------------- */

    // dealloc mem

    CUDA_CHECK(cudaFree(ini_uwpos));
    CUDA_CHECK(cudaFree(d_msd));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_msd
(
    const int n_gatoms,        const int *g_atoms,      
    const numtyp *d_uwpos,     const numtyp *ini_uwpos,
    numtyp *d_msd 
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdata[128]; 

    /* ------------------------------------------------------- */

    numtyp p_msd = (numtyp)0.0;
    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];

        numtyp dx = d_uwpos[atom_index*3+0] - ini_uwpos[atom_index*3+0];
        numtyp dy = d_uwpos[atom_index*3+1] - ini_uwpos[atom_index*3+1];
        numtyp dz = d_uwpos[atom_index*3+2] - ini_uwpos[atom_index*3+2];

        p_msd = dx * dx + dy * dy + dz * dz;
    }
    sdata[tid] = p_msd;
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
        atomicAdd(d_msd, sdata[0]);
    }  
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void msd::compute(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    if (step % frequency != 0) {
        return;
    }
    
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    int g_natoms = group.n_atoms;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemset(d_msd, 0, sizeof(numtyp)));

    int blockSize    = 128;
    int numBlocks    = (g_natoms + blockSize - 1) / blockSize;

    kernel_msd<<<numBlocks, blockSize>>>
    (
        g_natoms,        group.d_atoms,            
        atoms.d_uwpos,   ini_uwpos, 
        d_msd    
    );

    /* ------------------------------------------------------- */

    numtyp h_msd = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_msd, d_msd, sizeof(numtyp), cudaMemcpyDeviceToHost));
    h_msd = h_msd / g_natoms;

    /* ------------------------------------------------------- */

    file    << std::setw(15) << step    << "\t\t"
            << std::setw(15) << h_msd   << std::endl;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_COMPUTE(msd)
///////////////////////////////
