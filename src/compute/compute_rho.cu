#include "compute/compute_rho.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

rho::rho(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Compute(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string rho::getName() 
{
    return "rho";
}

/* ----------------------------------------------------------------------------------------------------------- */

void rho::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal compute_rho command";
    std::string format   = "compute ID   groupID rho frequency xlo xhi ylo yhi zlo zhi filename";
    std::string example0 = "compute crho liquid  rho 1 0.0 10.0 0.0 10.0 8.0 12.0 rho.txt";

    if (params.size() != 8) {
        print_error_and_exit("Invalid compute_rho parameters", error, format, {example0});
    }

    frequency = parse_int<unsigned int>(params[0], "Invalid compute_rho parameters", "frequency", {format,example0});
    xlo       = parse_float<numtyp>(params[1], "Invalid compute_rho parameters", "xlo", {format,example0});
    xhi       = parse_float<numtyp>(params[2], "Invalid compute_rho parameters", "xhi", {format,example0});
    ylo       = parse_float<numtyp>(params[3], "Invalid compute_rho parameters", "ylo", {format,example0});
    yhi       = parse_float<numtyp>(params[4], "Invalid compute_rho parameters", "yhi", {format,example0});
    zlo       = parse_float<numtyp>(params[5], "Invalid compute_rho parameters", "zlo", {format,example0});
    zhi       = parse_float<numtyp>(params[6], "Invalid compute_rho parameters", "zhi", {format,example0});
    filename  = params[7];
}

/* ----------------------------------------------------------------------------------------------------------- */

void rho::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    Box& box     = system.box;

    /* ------------------------------------------------------- */

    std::string error    = "illegal compute_rho command";
    std::string format   = "compute ID   groupID rho frequency xlo xhi ylo yhi zlo zhi filename";
    std::string example0 = "compute crho liquid  rho 1 0.0 10.0 0.0 10.0 8.0 12.0 rho.txt";

    if (xlo < box.xlo) {
        print_error_and_exit("xlo < box.xlo", error, format, {example0});
    }
    if (xhi > box.xhi) {
        print_error_and_exit("xhi > box.xhi", error, format, {example0});
    }
    if (ylo < box.ylo) {
        print_error_and_exit("ylo < box.ylo", error, format, {example0});
    }
    if (yhi > box.yhi) {
        print_error_and_exit("yhi > box.yhi", error, format, {example0});
    }
    if (zlo < box.zlo) {
        print_error_and_exit("zlo < box.zlo", error, format, {example0});
    }
    if (zhi > box.zhi) {
        print_error_and_exit("zhi > box.zhi", error, format, {example0});
    }

    vol = (xhi - xlo) * (yhi - ylo) * (zhi - zlo);
    
    /* ------------------------------------------------------- */

    // file rho 

    file.open(filename);

    file << std::setw(15) << "step" << "\t\t"
         << std::setw(15) << "rho"  << std::endl;

    /* ------------------------------------------------------- */

    // alloc mem

    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void rho::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file rho 

    file.close();

    /* ------------------------------------------------------- */

    // dealloc mem

    CUDA_CHECK(cudaFree(d_count));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_count
(
    const int n_gatoms,        const int *g_atoms,      
    const numtyp *d_pos,       int *d_count,
    const numtyp xlo,          const numtyp xhi,
    const numtyp ylo,          const numtyp yhi,
    const numtyp zlo,          const numtyp zhi    
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ int count[128]; 

    /* ------------------------------------------------------- */

    int p_count = 0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];

        const numtyp pos_x = d_pos[atom_index*3+0];
        const numtyp pos_y = d_pos[atom_index*3+1];
        const numtyp pos_z = d_pos[atom_index*3+2];

        int x_in = (pos_x >= xlo) * (pos_x < xhi);
        int y_in = (pos_y >= ylo) * (pos_y < yhi);
        int z_in = (pos_z >= zlo) * (pos_z < zhi);
        p_count  = x_in * y_in * z_in;
    }

    count[tid] = p_count;

    __syncthreads();

    /* ------------------------------------------------------- */

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            count[tid] += count[tid + s];
        }
        __syncthreads();
    }

    /* ------------------------------------------------------- */

    if (tid == 0) 
    {
        atomicAdd(&d_count[0], count[0]);
    }  
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void rho::compute(System& system, unsigned int step) 
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

    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_count<<<numBlocks, blockSize>>>
    (
        n_gatoms,         group.d_atoms,            
        atoms.d_pos,      d_count,
        xlo,              xhi,
        ylo,              yhi,
        zlo,              zhi    
    );
    
    /* ------------------------------------------------------- */

    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    numtyp rho = count / vol;

    /* ------------------------------------------------------- */

    file << std::setw(15) << step       << "\t\t"
         << std::setw(15) << rho        << std::endl;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_COMPUTE(rho)
///////////////////////////////
