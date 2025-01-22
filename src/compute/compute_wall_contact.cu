#include "compute/compute_wall_contact.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

wall_contact::wall_contact(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Compute(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string wall_contact::getName() 
{
    return "wall_contact";
}

/* ----------------------------------------------------------------------------------------------------------- */

void wall_contact::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal compute_wall_contact command";
    std::string format   = "compute ID   groupID wall_contact frequency wall_groupname lo/hi wall_direction cutoff filename";
    std::string example0 = "compute cwall_contact liquid  wall_contact 1 lo_wall lo z 1.0 wall_contact.txt";
    std::string example1 = "compute cwall_contact liquid  wall_contact 1 hi_wall hi z 1.0 wall_contact.txt";

    if (params.size() != 6) {
        print_error_and_exit("Invalid compute_wall_contact parameters", error, format, {example0, example1});
    }

    frequency      = parse_int<unsigned int>(params[0], "Invalid compute_wall_contact parameters", "frequency", {format,example0,example1});
    wall_groupname = params[1];

    if (params[2] == "lo") {
        wall_side = 1;
    } else if (params[2] == "hi") {
        wall_side = -1;
    } else {
        print_error_and_exit("Invalid compute_wall_contact parameters", error, format, {example0, example1});
    }

    if (params[3] == "x") {
        wall_direction = 0;
    } else if (params[3] == "y") {
        wall_direction = 1;
    } else if (params[3] == "z") {
        wall_direction = 2;
    } else {
        print_error_and_exit("Invalid compute_wall_contact parameters", error, format, {example0, example1});
    }
    
    cutoff   = parse_float<numtyp>(params[4], "Invalid compute_wall_contact parameters", "cutoff", {format,example0,example1});
    filename = params[5];
}

/* ----------------------------------------------------------------------------------------------------------- */

void wall_contact::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file wall_contact 

    file.open(filename);

    file << std::setw(15) << "step" << "\t\t"
         << std::setw(15) << "wall_contact"  << std::endl;

    /* ------------------------------------------------------- */

    // alloc mem

    Group& wall_group = find_group(system, this->wall_groupname);
    int wall_n_atoms  = wall_group.n_atoms;

    CUDA_CHECK(cudaMalloc(&wall_pos,  sizeof(numtyp) * wall_n_atoms));
    CUDA_CHECK(cudaMalloc(&d_contact, sizeof(int)));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void wall_contact::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    // file wall_contact 

    file.close();

    /* ------------------------------------------------------- */

    // dealloc mem

    CUDA_CHECK(cudaFree(wall_pos));
    CUDA_CHECK(cudaFree(d_contact));
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_count
(
    const int n_gatoms,        const int *g_atoms,      
    const numtyp *d_pos,       int *d_contact, 
    const int wall_direction,  const int wall_side,       
    const numtyp wall_bound,   const numtyp contact_bound
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ int contact[128]; 

    /* ------------------------------------------------------- */

    int p_contact = 0;

    if (i < n_gatoms)
    {
        int atom_index = g_atoms[i];

        const numtyp pos = d_pos[atom_index*3+wall_direction];

        if (wall_side == 1)
        {
            int pos_in = (pos >= wall_bound) * (pos < contact_bound); 
            p_contact  = pos_in;           
        }
        else if (wall_side == -1)
        {
            int pos_in = (pos >= contact_bound) * (pos < wall_bound); 
            p_contact  = pos_in;           
        }

    }

    contact[tid] = p_contact;

    __syncthreads();

    /* ------------------------------------------------------- */

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            contact[tid] += contact[tid + s];
        }
        __syncthreads();
    }

    /* ------------------------------------------------------- */

    if (tid == 0) 
    {
        atomicAdd(&d_contact[0], contact[0]);
    }  
    
    /* ------------------------------------------------------- */
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

void wall_contact::compute(System& system, unsigned int step) 
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

    Box& box          = system.box;
    Group& wall_group = find_group(system, this->wall_groupname);
    int wall_n_atoms  = wall_group.n_atoms;

    int blockSize    = 128;
    int numBlocks    = (wall_n_atoms + blockSize - 1) / blockSize;

    kernel_select_pos<<<numBlocks, blockSize>>>
    (
        wall_n_atoms,              wall_group.d_atoms,                              
        atoms.d_pos,               wall_pos,
        wall_direction        
    );

    /* ------------------------------------------------------- */

    thrust::device_ptr<numtyp> wall_pos_ptr = thrust::device_pointer_cast(wall_pos);

    numtyp wall_bound = 0.0;
    if (wall_side == 1) {
        thrust::device_ptr<numtyp> max_wall_pos_ptr = thrust::max_element(wall_pos_ptr, wall_pos_ptr + wall_n_atoms);
        cudaMemcpy(&wall_bound, max_wall_pos_ptr.get(), sizeof(numtyp), cudaMemcpyDeviceToHost);
    } else if (wall_side == -1) {
        thrust::device_ptr<numtyp> min_wall_pos_ptr = thrust::min_element(wall_pos_ptr, wall_pos_ptr + wall_n_atoms);
        cudaMemcpy(&wall_bound, min_wall_pos_ptr.get(), sizeof(numtyp), cudaMemcpyDeviceToHost);
    }

    /* ------------------------------------------------------- */

    numtyp box_bound     = 0.0;
    numtyp contact_bound = 0.0;

    if (wall_direction == 0) 
    {
        if (wall_side == 1) {
            box_bound     = box.xhi;
            contact_bound = wall_bound + cutoff;
            if (contact_bound > box_bound) {
                print_error({"contact region out of the box"});
            }
        } else if (wall_side == -1) {
            box_bound     = box.xlo;
            contact_bound = wall_bound - cutoff;
            if (contact_bound < box_bound) {
                print_error({"contact region out of the box"});
            }            
        }
    } 
    else if (wall_direction == 1) 
    {
        if (wall_side == 1) {
            box_bound     = box.yhi;
            contact_bound = wall_bound + cutoff;
            if (contact_bound > box_bound) {
                print_error({"contact region out of the box"});
            }
        } else if (wall_side == -1) {
            box_bound     = box.ylo;
            contact_bound = wall_bound - cutoff;
            if (contact_bound < box_bound) {
                print_error({"contact region out of the box"});
            }            
        } 
    }
    else if (wall_direction == 2) 
    {
        if (wall_side == 1) {
            box_bound     = box.zhi;
            contact_bound = wall_bound + cutoff;
            if (contact_bound > box_bound) {
                print_error({"contact region out of the box"});
            }
        } else if (wall_side == -1) {
            box_bound     = box.zlo;
            contact_bound = wall_bound - cutoff;
            if (contact_bound < box_bound) {
                print_error({"contact region out of the box"});
            }  
        }          
    }

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemset(d_contact, 0, sizeof(int)));

    numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_count<<<numBlocks, blockSize>>>
    (
        n_gatoms,         group.d_atoms,            
        atoms.d_pos,      d_contact,
        wall_direction,   wall_side,        
        wall_bound,       contact_bound
    );
    
    /* ------------------------------------------------------- */

    int wall_contact = 0;
    CUDA_CHECK(cudaMemcpy(&wall_contact, d_contact, sizeof(int), cudaMemcpyDeviceToHost));

    /* ------------------------------------------------------- */

    file << std::setw(15) << step           << "\t\t"
         << std::setw(15) << wall_contact   << std::endl;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_COMPUTE(wall_contact)
///////////////////////////////
