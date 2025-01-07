#include "fix/fix_move.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

move::move(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Fix(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string move::getName() 
{
    return "move";
}

/* ----------------------------------------------------------------------------------------------------------- */

void move::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal fix_move command";
    std::string format   = "fix ID groupID move v_x v_y v_z";
    std::string example0 = "fix md probe   move 0.0 0.0 -0.0001";
    
    if (params.size() != 3) {
        print_error_and_exit("Invalid fix_move parameters", error, format, {example0});
    }

    v_x = parse_float<numtyp>(params[0], "Invalid fix_move parameters", "v_x", {format, example0});
    v_y = parse_float<numtyp>(params[1], "Invalid fix_move parameters", "v_y", {format, example0});
    v_z = parse_float<numtyp>(params[2], "Invalid fix_move parameters", "v_z", {format, example0});
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_move
(
    const int n_gatoms,        const int *g_atoms,      const numtyp dt,  
    const numtyp lx,           const numtyp ly,         const numtyp lz,
    const numtyp xlo,          const numtyp ylo,        const numtyp zlo,
    const numtyp xhi,          const numtyp yhi,        const numtyp zhi,
    const numtyp v_x,          const numtyp v_y,        const numtyp v_z,
    numtyp *d_pos,             numtyp *d_uwpos,         numtyp *d_vel
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */

    d_vel[i*3+0] = v_x;
    d_vel[i*3+1] = v_y;
    d_vel[i*3+2] = v_z;

    /* ------------------------------------------------------- */

    const numtyp disp_x = v_x * dt;
    const numtyp disp_y = v_y * dt;
    const numtyp disp_z = v_z * dt;

    /* ------------------------------------------------------- */

    numtyp pos_x = d_pos[i*3+0] + disp_x;
    numtyp pos_y = d_pos[i*3+1] + disp_y;
    numtyp pos_z = d_pos[i*3+2] + disp_z;

    /* ------------------------------------------------------- */

    while (pos_x >= xhi || pos_x < xlo) {
        pos_x -= lx * ((pos_x >= xhi) - (pos_x < xlo));
    }
    while (pos_y >= yhi || pos_y < ylo) {
        pos_y -= ly * ((pos_y >= yhi) - (pos_y < ylo));
    }
    while (pos_z >= zhi || pos_z < zlo) {
        pos_z -= lz * ((pos_z >= zhi) - (pos_z < zlo));
    }

    /* ------------------------------------------------------- */

    d_pos[i*3+0] = pos_x;
    d_pos[i*3+1] = pos_y;
    d_pos[i*3+2] = pos_z;
    
    d_uwpos[i*3+0] += disp_x;
    d_uwpos[i*3+1] += disp_y;
    d_uwpos[i*3+2] += disp_z;        
        
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void move::initial_integrate(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    Box& box     = system.box;
    int n_gatoms = group.n_atoms;
    numtyp dt    = system.run_p.dt;

    /* ------------------------------------------------------- */

    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_move<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  group.d_atoms,                 dt,  
        box.lx,                    box.ly,                        box.lz,
        box.xlo,                   box.ylo,                       box.zlo,
        box.xhi,                   box.yhi,                       box.zhi,
        v_x,                       v_y,                           v_z,                
        atoms.d_pos,               atoms.d_uwpos,                 atoms.d_vel                  
    );
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_FIX(move)
///////////////////////////////
