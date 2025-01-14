#include "fix/fix_wall_reflect.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

wall_reflect::wall_reflect(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Fix(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string wall_reflect::getName() 
{
    return "wall_reflect";
}

/* ----------------------------------------------------------------------------------------------------------- */

void wall_reflect::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal fix_wall_reflect command";
    std::string format   = "fix ID groupID wall_reflect lo/hi wall_position wall_direction";
    std::string example0 = "fix lo_wall water wall_reflect lo 2 z";
    std::string example1 = "fix hi_wall water wall_reflect hi 50 z";
    
    if (params.size() != 3) {
        print_error_and_exit("Invalid fix_wall_reflect parameters", error, format, {example0, example1});
    }

    if (params[0] == "lo") {
        wall_side = 1;
    } else if (params[0] == "hi") {
        wall_side = -1;
    } else {
        print_error_and_exit("Invalid fix_wall_reflect lo/hi", error, format, {example0, example1});
    }

    wall_pos = parse_float<numtyp>(params[1], "Invalid fix_wall_reflect parameters", "position",  {format, example0, example1});

    if (params[2] == "x") {
        wall_direction = 0;
    } else if (params[2] == "y") {
        wall_direction = 1;
    } else if (params[2] == "z") {
        wall_direction = 2;
    } else {
        print_error_and_exit("Invalid fix_wall_reflect direction", error, format, {example0, example1});
    }        
}

/* ----------------------------------------------------------------------------------------------------------- */

void wall_reflect::preprocess(System& system)
{
    Box& box = system.box;
    numtyp box_lo_bound;
    numtyp box_hi_bound;

    if (wall_direction == 0)
    {
        box_lo_bound = box.xlo;
        box_hi_bound = box.xhi;
    }
    else if (wall_direction == 1)
    {
        box_lo_bound = box.ylo;
        box_hi_bound = box.yhi;
    }
    else if (wall_direction == 2)
    {
        box_lo_bound = box.zlo;
        box_hi_bound = box.zhi;
    }

    if (wall_pos < box_lo_bound || wall_pos > box_hi_bound)
    {
        print_error({"Wall position is out of the box!"});
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_wall_reflect
(
    const int n_gatoms,        const int *g_atoms,     
    numtyp *d_pos,             numtyp *d_uwpos,         numtyp *d_vel,
    const numtyp wall_pos,     const int  wall_side,    const int wall_direction  
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */

    const numtyp gatom_pos = d_pos[i*3+wall_direction];
    const numtyp gatom_vel = d_vel[i*3+wall_direction];
    
    if ((wall_side == 1 && gatom_pos < wall_pos) || (wall_side == -1 && gatom_pos > wall_pos)) {
        numtyp disp = wall_pos - gatom_pos;
        numtyp total_disp = 2 * disp; 
        d_pos[i*3+wall_direction]   += total_disp;
        d_vel[i*3+wall_direction]   = -gatom_vel; 
        d_uwpos[i*3+wall_direction] += total_disp; 
    }
 
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void wall_reflect::post_integrate(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    int n_gatoms = group.n_atoms;

    /* ------------------------------------------------------- */

    // reflect 

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_wall_reflect<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,        
        atoms.d_pos,               atoms.d_uwpos,                 atoms.d_vel,
        wall_pos,                  wall_side,                     wall_direction           
    );
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_FIX(wall_reflect)
///////////////////////////////
