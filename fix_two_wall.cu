#include "fix/fix_two_wall.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

two_wall::two_wall(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Fix(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string two_wall::getName() 
{
    return "two_wall";
}

/* ----------------------------------------------------------------------------------------------------------- */

void two_wall::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal fix_two_wall command";
    std::string format   = "fix ID groupID two_wall lo_wall_name hi_wall_name direction";
    std::string example0 = "fix md water   two_wall lo_wall hi_wall z";
    
    if (params.size() != 3) {
        print_error_and_exit("Invalid fix_two_wall parameters", error, format, {example0});
    }

    lo_wall = params[0];
    hi_wall = params[1];

    if (params[2] == "x") {
        wall_direction = 0;
    } else if (params[2] == "y") {
        wall_direction = 1;
    } else if (params[2] == "z") {
        wall_direction = 2;
    } else {
        print_error_and_exit("Invalid fix_two_wall direction", error, format, {example0});
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

void two_wall::preprocess(System& system)
{
    Group& lo_wall_group = find_group(system, this->lo_wall);
    Group& hi_wall_group = find_group(system, this->hi_wall);
    int lo_wall_n_atoms = lo_wall_group.n_atoms;
    int hi_wall_n_atoms = hi_wall_group.n_atoms;

    CUDA_CHECK(cudaMalloc(&lo_wall_pos, sizeof(numtyp) * lo_wall_n_atoms));
    CUDA_CHECK(cudaMalloc(&hi_wall_pos, sizeof(numtyp) * hi_wall_n_atoms));
}

/* ----------------------------------------------------------------------------------------------------------- */

void two_wall::postprocess(System& system)
{
    CUDA_CHECK(cudaFree(lo_wall_pos));
    CUDA_CHECK(cudaFree(hi_wall_pos));
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_select_pos
(
    const int n_gatoms,        const int *g_atoms,
    const numtyp *d_pos,       numtyp *wall_pos,
    const int wall_direction      
) 
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */

    wall_pos[idx] = d_pos[3*i+wall_direction];

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_two_wall
(
    const int n_gatoms,        const int *g_atoms,     
    numtyp *d_pos,             numtyp *d_uwpos,         numtyp *d_vel,
    const numtyp max_pos,      const numtyp min_pos,    const int wall_direction  
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */

    const numtyp gatom_pos = d_pos[i*3+wall_direction];
    const numtyp gatom_vel = d_vel[i*3+wall_direction];

    if (gatom_pos < max_pos)
    {
        numtyp disp = max_pos - gatom_pos;
        numtyp total_disp = 2 * disp; 
        d_pos[i*3+wall_direction]   += total_disp;
        d_vel[i*3+wall_direction]   = -gatom_vel; 
        d_uwpos[i*3+wall_direction] += total_disp; 
    } 
    else if (gatom_pos > min_pos)
    {
        numtyp disp = min_pos - gatom_pos;
        numtyp total_disp = 2 * disp; 
        d_pos[i*3+wall_direction]   += total_disp;
        d_vel[i*3+wall_direction]   = -gatom_vel; 
        d_uwpos[i*3+wall_direction] += total_disp;     
    }
 
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void two_wall::post_integrate(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    int n_gatoms = group.n_atoms;

    /* ------------------------------------------------------- */

    // lo_wall

    Group& lo_wall_group = find_group(system, this->lo_wall);
    int lo_wall_n_atoms = lo_wall_group.n_atoms;

    int blockSize    = 128;
    int numBlocks    = (lo_wall_n_atoms + blockSize - 1) / blockSize;

    kernel_select_pos<<<numBlocks, blockSize>>>
    (
        lo_wall_n_atoms,           lo_wall_group.d_atoms,                              
        atoms.d_pos,               lo_wall_pos,
        wall_direction        
    );

    thrust::device_ptr<numtyp> lo_wall_pos_ptr     = thrust::device_pointer_cast(lo_wall_pos);
    thrust::device_ptr<numtyp> max_lo_wall_pos_ptr = thrust::max_element(lo_wall_pos_ptr, lo_wall_pos_ptr + lo_wall_n_atoms);
    numtyp max_lo_wall_pos;
    cudaMemcpy(&max_lo_wall_pos, max_lo_wall_pos_ptr.get(), sizeof(numtyp), cudaMemcpyDeviceToHost);
    
    /* ------------------------------------------------------- */

    // hi_wall

    Group& hi_wall_group = find_group(system, this->hi_wall);
    int hi_wall_n_atoms = hi_wall_group.n_atoms;

    numBlocks    = (hi_wall_n_atoms + blockSize - 1) / blockSize;

    kernel_select_pos<<<numBlocks, blockSize>>>
    (
        hi_wall_n_atoms,           hi_wall_group.d_atoms,                              
        atoms.d_pos,               hi_wall_pos,
        wall_direction        
    );

    thrust::device_ptr<numtyp> hi_wall_pos_ptr     = thrust::device_pointer_cast(hi_wall_pos);
    thrust::device_ptr<numtyp> min_hi_wall_pos_ptr = thrust::min_element(hi_wall_pos_ptr, hi_wall_pos_ptr + hi_wall_n_atoms);
    numtyp min_hi_wall_pos;
    cudaMemcpy(&min_hi_wall_pos, min_hi_wall_pos_ptr.get(), sizeof(numtyp), cudaMemcpyDeviceToHost);
    
    /* ------------------------------------------------------- */

    if (min_hi_wall_pos <= max_lo_wall_pos)
    {
        print_error({"The high wall is positioned lower than the low wall"});
    }

    /* ------------------------------------------------------- */

    // reflect 

    numBlocks = (n_gatoms + blockSize - 1) / blockSize;

    kernel_two_wall<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,        
        atoms.d_pos,               atoms.d_uwpos,                 atoms.d_vel,
        max_lo_wall_pos,           min_hi_wall_pos,               wall_direction           
    );
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_FIX(two_wall)
///////////////////////////////
