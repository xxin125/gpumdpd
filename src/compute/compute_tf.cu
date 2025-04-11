#include "compute/compute_tf.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

tf::tf(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Compute(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string tf::getName() 
{
    return "tf";
}

/* ----------------------------------------------------------------------------------------------------------- */

void tf::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal compute_tf command";
    std::string format   = "compute ID   groupID tf frequency filename";
    std::string example0 = "compute ctf liquid  tf 1 tf.txt";

    if (params.size() != 2) {
        print_error_and_exit("Invalid compute_tf parameters", error, format, {example0});
    }

    frequency = parse_int<unsigned int>(params[0], "Invalid compute_tf parameters", "frequency", {format,example0});
    filename  = params[1];
}

/* ----------------------------------------------------------------------------------------------------------- */

void tf::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file tf 

    file.open(filename);

    file << std::setw(15) << "step" << "\t\t"
         << std::setw(15) << "tf_x" << "\t\t" 
         << std::setw(15) << "tf_y" << "\t\t"  
         << std::setw(15) << "tf_z" << std::endl;

    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaMalloc(&d_tf, sizeof(numtyp) * 3));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void tf::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file tf 

    file.close();

    /* ------------------------------------------------------- */

    // dealloc mem

    CUDA_CHECK(cudaFree(d_tf));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_tf
(
    const int n_gatoms,        const int *g_atoms,      
    const numtyp *d_force,     numtyp *d_tf 
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp tf_x[128]; 
    __shared__ numtyp tf_y[128];
    __shared__ numtyp tf_z[128];

    /* ------------------------------------------------------- */

    numtyp p_tf_x = (numtyp)0.0;
    numtyp p_tf_y = (numtyp)0.0;
    numtyp p_tf_Z = (numtyp)0.0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];

        p_tf_x = d_force[atom_index*3+0];
        p_tf_y = d_force[atom_index*3+1];
        p_tf_Z = d_force[atom_index*3+2];
    }

    tf_x[tid] = p_tf_x;
    tf_y[tid] = p_tf_y;
    tf_z[tid] = p_tf_Z;

    __syncthreads();

    /* ------------------------------------------------------- */

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            tf_x[tid] += tf_x[tid + s];
            tf_y[tid] += tf_y[tid + s];
            tf_z[tid] += tf_z[tid + s];
        }
        __syncthreads();
    }

    /* ------------------------------------------------------- */

    if (tid == 0) 
    {
        atomicAdd(&d_tf[0], tf_x[0]);
        atomicAdd(&d_tf[1], tf_y[0]);
        atomicAdd(&d_tf[2], tf_z[0]);
    }  
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void tf::compute(System& system, unsigned int step) 
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

    CUDA_CHECK(cudaMemset(d_tf, 0, sizeof(numtyp) * 3));

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_tf<<<numBlocks, blockSize>>>
    (
        n_gatoms,         group.d_atoms,            
        atoms.d_force,    d_tf    
    );
    
    /* ------------------------------------------------------- */

    numtyp total_fx= 0;
    numtyp total_fy= 0;
    numtyp total_fz= 0;

    CUDA_CHECK(cudaMemcpy(&total_fx, d_tf + 0, sizeof(numtyp), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&total_fy, d_tf + 1, sizeof(numtyp), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&total_fz, d_tf + 2, sizeof(numtyp), cudaMemcpyDeviceToHost));

    /* ------------------------------------------------------- */

    file << std::setw(15) << step       << "\t\t"
         << std::setw(15) << total_fx   << "\t\t"
         << std::setw(15) << total_fy   << "\t\t"
         << std::setw(15) << total_fz   << std::endl;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_COMPUTE(tf)
///////////////////////////////
