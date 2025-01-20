#include "compute/compute_temp.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

temp::temp(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Compute(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string temp::getName() 
{
    return "temp";
}

/* ----------------------------------------------------------------------------------------------------------- */

void temp::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal compute_temp command";
    std::string format   = "compute ID   groupID temp frequency filename";
    std::string example0 = "compute ctemp liquid temp 1  temp.txt";

    if (params.size() != 2) {
        print_error_and_exit("Invalid compute_temp parameters", error, format, {example0});
    }

    frequency = parse_int<unsigned int>(params[0], "Invalid compute_temp parameters", "frequency", {format,example0});
    filename  = params[1];
}

/* ----------------------------------------------------------------------------------------------------------- */

void temp::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file temp 

    file.open(filename);

    file << std::setw(15) << "step" << "\t\t"
         << std::setw(15) << "temp"  << std::endl;

    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaMalloc(&d_ctemp, sizeof(numtyp)));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void temp::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file temp 

    file.close();

    /* ------------------------------------------------------- */

    // dealloc mem

    CUDA_CHECK(cudaFree(d_ctemp));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_temp
(
    const int n_gatoms,        const int *g_atoms,    
    const int *d_type,         const numtyp *d_vel,       
    numtyp *d_ctemp  
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdata[128]; 

    /* ------------------------------------------------------- */

    numtyp p_ke = (numtyp)0.0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];

        const numtyp vel_x = d_vel[atom_index*3+0];
        const numtyp vel_y = d_vel[atom_index*3+1];
        const numtyp vel_z = d_vel[atom_index*3+2];
        const numtyp massi = masses[d_type[atom_index]-1];
        p_ke = (numtyp)0.5 * massi * (vel_x * vel_x
                                   +  vel_y * vel_y
                                   +  vel_z * vel_z);
    }
    sdata[tid] = p_ke;
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
        atomicAdd(d_ctemp, sdata[0]);
    }  
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void temp::compute(System& system, unsigned int step) 
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

    CUDA_CHECK(cudaMemset(d_ctemp, 0.0, sizeof(numtyp)));

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_temp<<<numBlocks, blockSize>>>
    (
        n_gatoms,         group.d_atoms,       
        atoms.d_type,     atoms.d_vel,      
        d_ctemp
    );
    
    /* ------------------------------------------------------- */

    numtyp total_ke = (numtyp)0.0;
    CUDA_CHECK(cudaMemcpy(&total_ke, d_ctemp, sizeof(numtyp), cudaMemcpyDeviceToHost));
    numtyp temp = (2 * total_ke) / (3 * n_gatoms * 1);

    /* ------------------------------------------------------- */

    file << std::setw(15) << step       << "\t\t"
         << std::setw(15) << temp        << std::endl;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_COMPUTE(temp)
///////////////////////////////
