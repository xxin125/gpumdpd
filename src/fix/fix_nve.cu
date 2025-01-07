#include "fix/fix_nve.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

nve::nve(std::string id, std::string gid, const std::vector<std::string>& params) 
    : Fix(id, gid) {}

/* ----------------------------------------------------------------------------------------------------------- */

std::string nve::getName() 
{
    return "nve";
}

/* ----------------------------------------------------------------------------------------------------------- */

void nve::validateParams(const std::vector<std::string>& params)
{
    std::string error    = "illegal fix_nve command";
    std::string format   = "fix ID groupID nve";
    std::string example0 = "fix md liquid  nve";

    if (!params.empty()) {
        print_error_and_exit("Invalid fix_nve parameters", error, format, {example0});
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_nve_initial_integrate
(
    const int n_gatoms,        const numtyp dt,
    const int *g_atoms,        const int *d_type,       const numtyp *d_force,
    const numtyp lx,           const numtyp ly,         const numtyp lz,
    const numtyp xlo,          const numtyp ylo,        const numtyp zlo,
    const numtyp xhi,          const numtyp yhi,        const numtyp zhi,
    numtyp *d_pos,             numtyp *d_uwpos,         numtyp *d_vel
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */
    
    const numtyp mass_i    = (numtyp)1.0 / masses[d_type[i]-1];
    const numtyp dt_half   = (numtyp)0.5 * dt;
    const numtyp dt_mass_i = dt_half * mass_i;

    /* ------------------------------------------------------- */

    numtyp vel_x = d_vel[i*3+0] + dt_mass_i * d_force[i*3+0];
    numtyp vel_y = d_vel[i*3+1] + dt_mass_i * d_force[i*3+1];
    numtyp vel_z = d_vel[i*3+2] + dt_mass_i * d_force[i*3+2];

    d_vel[i*3+0] = vel_x;
    d_vel[i*3+1] = vel_y;
    d_vel[i*3+2] = vel_z;

    numtyp pos_x = d_pos[i*3+0] + dt * vel_x;
    numtyp pos_y = d_pos[i*3+1] + dt * vel_y;
    numtyp pos_z = d_pos[i*3+2] + dt * vel_z;

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
    
    d_uwpos[i*3+0] += dt * vel_x;
    d_uwpos[i*3+1] += dt * vel_y;
    d_uwpos[i*3+2] += dt * vel_z;         
        
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void nve::initial_integrate(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;
    Box& box     = system.box;

    /* ------------------------------------------------------- */

    int n_gatoms     = group.n_atoms;
    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_nve_initial_integrate<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  system.run_p.dt,
        group.d_atoms,             atoms.d_type,                  atoms.d_force,
        box.lx,                    box.ly,                        box.lz,
        box.xlo,                   box.ylo,                       box.zlo,
        box.xhi,                   box.yhi,                       box.zhi,
        atoms.d_pos,               atoms.d_uwpos,                 atoms.d_vel  
    );
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_nve_final_integrate
(
    const int n_gatoms,        const numtyp dt,
    const int *g_atoms,        const int *d_type,       const numtyp *d_force,
    numtyp *d_vel
)
{
    /* ------------------------------------------------------- */

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gatoms) return;
    const int i   = g_atoms[idx];

    /* ------------------------------------------------------- */
    
    const numtyp mass_i    = (numtyp)1.0 / masses[d_type[i]-1];
    const numtyp dt_half   = (numtyp)0.5 * dt;
    const numtyp dt_mass_i = dt_half * mass_i;

    /* ------------------------------------------------------- */

    numtyp vel_x = d_vel[i*3+0] + dt_mass_i * d_force[i*3+0];
    numtyp vel_y = d_vel[i*3+1] + dt_mass_i * d_force[i*3+1];
    numtyp vel_z = d_vel[i*3+2] + dt_mass_i * d_force[i*3+2];

    d_vel[i*3+0] = vel_x;
    d_vel[i*3+1] = vel_y;
    d_vel[i*3+2] = vel_z;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void nve::final_integrate(System& system, unsigned int step) 
{
    /* ------------------------------------------------------- */

    Group& group = find_group(system, this->group_id);
    Atoms& atoms = system.atoms;

    /* ------------------------------------------------------- */

    int n_gatoms     = group.n_atoms;
    int blockSize    = 128;
    int numBlocks    = (n_gatoms + blockSize - 1) / blockSize;

    kernel_nve_final_integrate<<<numBlocks, blockSize>>>
    (
        n_gatoms,                  system.run_p.dt,
        group.d_atoms,             atoms.d_type,             atoms.d_force,
        atoms.d_vel  
    );
    
    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////////////
REGISTER_FIX(nve)
///////////////////////////////
