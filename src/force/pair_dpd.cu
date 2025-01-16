#include "force/pair_dpd.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

Pair_dpd::Pair_dpd() 
{
    pair_style_name = "dpd";
}

/* ----------------------------------------------------------------------------------------------------------- */

std::string Pair_dpd::getName() 
{
    return pair_style_name;
}

/* ----------------------------------------------------------------------------------------------------------- */

bool Pair_dpd::isEnabled(System& system) 
{
    /* ------------------------------------------------------- */

    Run_p& run_p       = system.run_p;
    std::string& input = system.input; 
    int n_atomtypes    = system.n_atomtypes;
    bool enabled       = false;

    /* ------------------------------------------------------- */

    // pair_coeff 

    got_pair_coeff.resize((n_atomtypes+1)*n_atomtypes/2, 0);
    pair_coeff.resize(got_pair_coeff.size()*3, 0.0);

    /* ------------------------------------------------------- */

    // read run_in

    std::stringstream ss(input);
    std::string line;
    std::vector<std::string> args;
    std::string arg;

    /* ------------------------------------------------------- */

    while (std::getline(ss, line)) 
    {
        std::istringstream iss(line);
        args.clear();
        std::string key;
        iss >> key;

        while (iss >> arg) {
            args.push_back(arg);
        }

        // read pair_style

        if (key == "pair_style") 
        {

            if (args[0] == "dpd") {enabled = true;}

            if (enabled)
            {
                std::string error    = "illegal pair_style dpd command";
                std::string format   = "pair_style dpd temp rc seed";
                std::string example0 = "pair_style dpd 1.0 1.0 123456";

                if (args.size() != 4) {
                    print_error_and_exit(line, error, format, {example0});
                }

                temp = parse_float<numtyp>(args[1], line, "temp", {format,example0});
                rc   = parse_float<numtyp>(args[2], line, "rc", {format,example0});
                seed = parse_int<unsigned int>(args[3], line, "seed", {format,example0});
                
                if (rc > run_p.global_cut) {
                    error = "global rc should not be bigger than the global_cut (neighbor)"; 
                    print_error_and_exit(line, error, format, {example0});
                }
            }
        }
    } 

    /* ------------------------------------------------------- */

    // read pair_coeff

    if (enabled) 
    {
        std::stringstream ss2(input);
        while (std::getline(ss2, line)) 
        {
            std::istringstream iss(line);
            args.clear();
            std::string key;
            iss >> key;
    
            while (iss >> arg) {
                args.push_back(arg);
            }
    
            if (key == "pair_coeff") 
            {
                std::string error    = "illegal pair_coeff command for pair dpd";
                std::string format   = "pair_coeff typei typej A gamma rc(optional)";
                std::string example0 = "pair_coeff 1 1 25.0 4.5";
                std::string example1 = "pair_coeff 1 1 25.0 4.5 1.0";

                if ((args.size() != 4) && (args.size() != 6)) {
                    print_error_and_exit(line, error, format, {example0, example1});
                }

                int typei = parse_int<int>(args[0], line, "typei", {format, example0, example1});
                int typej = parse_int<int>(args[1], line, "typej", {format, example0, example1});

                if (typei > typej || typej > n_atomtypes) 
                {
                    std::string example2 = "typei <= typej ";
                    print_error_and_exit(line, error, format, {example0, example1, example2});                     
                } 

                numtyp A     = parse_float<numtyp>(args[2], line, "A", {format,example0, example1});
                numtyp gamma = parse_float<numtyp>(args[3], line, "gamma", {format,example0, example1});

                numtyp rc_cut = 0.0;
                if (args.size() == 4) {
                    rc_cut = rc; 
                } else {
                    rc_cut = parse_float<numtyp>(args[4], line, "rc",     {format,example0, example1});
                }
              
                int index = ((typei - 1) * n_atomtypes - (typei - 1) * (typei - 2) / 2 + typej - typei + 1) - 1;

                got_pair_coeff[index] = 1;
                pair_coeff[index*3+0] = A;
                pair_coeff[index*3+1] = gamma;
                pair_coeff[index*3+2] = rc_cut;
                if (rc_cut > run_p.global_cut) {
                    error = "global rc should not be bigger than the global_cut (neighbor)"; 
                    print_error_and_exit(line, error, format, {example0});
                }
            }
        } 
    }

    /* ------------------------------------------------------- */

    // check all pair_coeff

    if (enabled) 
    {
        bool all_pair_coeff = true;
        for (int i=0; i < got_pair_coeff.size(); i++) 
        {
            if (got_pair_coeff[i] == 0) {
                all_pair_coeff = false;
            }
        }
        if (!all_pair_coeff) {
            print_error({"all pair_coeff are not set"});   
        } else {
            CUDA_CHECK(cudaMemcpyToSymbol(gpu_pair_coeff, pair_coeff.data(), (pair_coeff.size() * sizeof(numtyp))));
        }
    }

    /* ------------------------------------------------------- */

    return enabled;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Pair_dpd::print_pair_info(System& system) 
{
    std::cout << "   pair_style           dpd temp rc seed" << std::endl;
    std::cout << "   pair_style           dpd " << temp << " " << rc << " " << seed  << std::endl;
    std::cout << "   pair_coeff           typei typej A gamma rc" << std::endl;
    for (int i=0; i<system.n_atomtypes; i++)
    {
        for (int j=i; j<system.n_atomtypes; j++)
        {
            int typei = i+1;
            int typej = j+1;
            int _id = ((typei - 1) * system.n_atomtypes - (typei - 1) * (typei - 2) / 2 + typej - typei + 1) - 1;
            std::cout << "   pair_coeff           " << typei << " " << typej << " " << pair_coeff[_id*3+0] << " " 
                                                                                    << pair_coeff[_id*3+1] << " " 
                                                                                    << pair_coeff[_id*3+2] << std::endl;
        }
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

// specifically implemented for steps = 1; high = 1.0; low = -1.0
// returns uniformly distributed random numbers u in [-1.0;1.0] using TEA8
// then multiply u with sqrt(3) to "match" with a normal random distribution
// Afshar et al. mutlplies u in [-0.5;0.5] with sqrt(12)

#define TWO_N32 0.232830643653869628906250e-9f /* 2^-32 */
#define SQRT3 (numtyp) 1.7320508075688772935274463
#define k0 0xA341316C
#define k1 0xC8013EA4
#define k2 0xAD90777D
#define k3 0x7E95761E
#define delta 0x9e3779b9
#define rounds 8

static __device__ void saru(unsigned int seed1, unsigned int seed2, unsigned int seed, int timestep, numtyp &randnum) {
    unsigned int seed3 = seed + timestep;                                     
    seed3 ^= (seed1 << 7) ^ (seed2 >> 6);                                     
    seed2 += (seed1 >> 4) ^ (seed3 >> 15);                                    
    seed1 ^= (seed2 << 9) + (seed3 << 8);                                     
    seed3 ^= 0xA5366B4D * ((seed2 >> 11) ^ (seed1 << 1));                     
    seed2 += 0x72BE1579 * ((seed1 << 4) ^ (seed3 >> 16));                     
    seed1 ^= 0x3F38A6ED * ((seed3 >> 5) ^ (((signed int) seed2) >> 22));      
    seed2 += seed1 * seed3;                                                   
    seed1 += seed3 ^ (seed2 >> 2);                                            
    seed2 ^= ((signed int) seed2) >> 17;                                      
    unsigned int state = 0x79dedea3 * (seed1 ^ (((signed int) seed1) >> 14)); 
    unsigned int wstate = (state + seed2) ^ (((signed int) state) >> 8);      
    state = state + (wstate * (wstate ^ 0xdddf97f5));                         
    wstate = 0xABCB96F7 + (wstate >> 1);                                      
    unsigned int sum = 0;                                                     
    for (int i = 0; i < rounds; i++) {                                        
      sum += delta;                                                           
      state += ((wstate << 4) + k0) ^ (wstate + sum) ^ ((wstate >> 5) + k1);  
      wstate += ((state << 4) + k2) ^ (state + sum) ^ ((state >> 5) + k3);    
    }                                                                         
    unsigned int v = (state ^ (state >> 26)) + wstate;                        
    unsigned int s = (signed int) ((v ^ (v >> 20)) * 0x6957f5a7);             
    randnum = SQRT3 * (s * TWO_N32 * (numtyp) 2.0 - (numtyp) 1.0);              
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_dpd_force_log
(
    const int N,              const int n_atomtypes,   
    const int n_max_neigh,    const int *d_n_neigh,     const int *d_neigh,
    const int *d_id,          const int *d_type,
    const numtyp *d_pos,      const numtyp *d_vel,   
    numtyp *d_force,          numtyp *d_pe,             numtyp *d_viral,
    const unsigned int step,  const unsigned int seed,  const numtyp dt_1_2, 
    const numtyp lx,          const numtyp ly,          const numtyp lz,
    const numtyp hlx,         const numtyp hly,         const numtyp hlz,
    int threadsPerParticle
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x % threadsPerParticle; 
    int i = blockIdx.x * (blockDim.x / threadsPerParticle) + threadIdx.x / threadsPerParticle;  

    if (i >= N) return;

    /* ------------------------------------------------------- */

    const int id_i     = d_id[i];
    const int type_i   = d_type[i];
    const numtyp x_i   = d_pos[i*3+0];
    const numtyp y_i   = d_pos[i*3+1];
    const numtyp z_i   = d_pos[i*3+2];
    const numtyp vx_i  = d_vel[i*3+0];
    const numtyp vy_i  = d_vel[i*3+1];
    const numtyp vz_i  = d_vel[i*3+2];

    /* ------------------------------------------------------- */

    numtyp fx    = (numtyp)0.0;
    numtyp fy    = (numtyp)0.0;
    numtyp fz    = (numtyp)0.0;
    numtyp pe    = (numtyp)0.0;
    numtyp virxx = (numtyp)0.0;
    numtyp virxy = (numtyp)0.0;
    numtyp virxz = (numtyp)0.0;
    numtyp viryy = (numtyp)0.0;
    numtyp viryz = (numtyp)0.0;
    numtyp virzz = (numtyp)0.0;

    /* ------------------------------------------------------- */

    int neighPerThread = (d_n_neigh[i] + threadsPerParticle - 1) / threadsPerParticle;
    int start          = tid * neighPerThread;
    int end            = min(start + neighPerThread, d_n_neigh[i]);

    /* ------------------------------------------------------- */

    for (int k = start; k < end; k++)
    {
        /* --------------------------------------------------- */

        const int j = d_neigh[i * n_max_neigh + k];
        numtyp dx = x_i - d_pos[j*3+0];
        numtyp dy = y_i - d_pos[j*3+1];
        numtyp dz = z_i - d_pos[j*3+2];

        dx = dx - lx * ((dx >= hlx) - (dx < -hlx));
        dy = dy - ly * ((dy >= hly) - (dy < -hly));
        dz = dz - lz * ((dz >= hlz) - (dz < -hlz));
        const numtyp r2 = dx * dx + dy * dy + dz * dz;

        /* --------------------------------------------------- */

        const int type_j = d_type[j];
        int t_i = min(type_i, type_j);
        int t_j = max(type_i, type_j);
        const int index  = ((t_i - 1) * n_atomtypes - (t_i - 1) * (t_i - 2) / 2 + t_j - t_i + 1) - 1;
        const numtyp A_ij     = gpu_pair_coeff[index*3+0];
        const numtyp gamma_ij = gpu_pair_coeff[index*3+1];
        const numtyp rc       = gpu_pair_coeff[index*3+2];
        const numtyp sigma_ij_dt_1_2 = sqrt(gamma_ij) * dt_1_2;     

        /* --------------------------------------------------- */

        if (r2 <= rc*rc)
        { 
            /* ----------------------------------------------- */

            const numtyp r = sqrt(r2);
            if (r < 1.0e-10) continue;
            const numtyp r_inv = ((numtyp)1.0/r);

            /* ----------------------------------------------- */

            const numtyp dvx = vx_i - d_vel[j*3+0];
            const numtyp dvy = vy_i - d_vel[j*3+1];
            const numtyp dvz = vz_i - d_vel[j*3+2];
            const numtyp dot = dx * dvx + dy * dvy + dz * dvz;

            /* ----------------------------------------------- */

            const numtyp wc = (numtyp)1.0 - (r/rc);
            const numtyp wr = wc;

            /* ----------------------------------------------- */

            numtyp randnum    = (numtyp)0.0;
            unsigned int tag1 = min(id_i,  d_id[j]);
            unsigned int tag2 = max(id_i,  d_id[j]);
            saru(tag1, tag2, seed, step, randnum);  

            /* ----------------------------------------------- */

            // MDPD forces computation

            numtyp fpair = (A_ij * wc);
            fpair       -= (gamma_ij * wr * wr * dot * r_inv);
            fpair       += (sigma_ij_dt_1_2 * wr * randnum); 
            fpair       *= r_inv; 
            
            /* ----------------------------------------------- */

            const numtyp ffx = dx * fpair;
            const numtyp ffy = dy * fpair;
            const numtyp ffz = dz * fpair;
            fx += ffx;
            fy += ffy;
            fz += ffz;

            /* ----------------------------------------------- */

            const numtyp ppe = ((numtyp)0.5 * A_ij * rc * wc * wc);
            pe += ppe;

            /* ----------------------------------------------- */

            const numtyp vvirxx = (numtyp)0.5*dx*ffx;
            const numtyp vvirxy = (numtyp)0.5*dx*ffy;
            const numtyp vvirxz = (numtyp)0.5*dx*ffz;
            const numtyp vviryy = (numtyp)0.5*dy*ffy;
            const numtyp vviryz = (numtyp)0.5*dy*ffz;
            const numtyp vvirzz = (numtyp)0.5*dz*ffz;      

            virxx += vvirxx;
            virxy += vvirxy;
            virxz += vvirxz;
            viryy += vviryy;
            viryz += vviryz;
            virzz += vvirzz;

            /* ----------------------------------------------- */

            if (j < N)
            {
                atomicAdd(&d_force[j*3+0], -ffx);
                atomicAdd(&d_force[j*3+1], -ffy);
                atomicAdd(&d_force[j*3+2], -ffz);
                atomicAdd(&d_pe[i], (numtyp)0.5*ppe);
                atomicAdd(&d_viral[j*6+0], vvirxx);
                atomicAdd(&d_viral[j*6+1], vvirxy);
                atomicAdd(&d_viral[j*6+2], vvirxz);
                atomicAdd(&d_viral[j*6+3], vviryy);
                atomicAdd(&d_viral[j*6+4], vviryz);
                atomicAdd(&d_viral[j*6+5], vvirzz);  
            }

            /* ----------------------------------------------- */
        }
    }

    /* ------------------------------------------------------- */

    atomicAdd(&d_force[i*3+0], fx);
    atomicAdd(&d_force[i*3+1], fy);
    atomicAdd(&d_force[i*3+2], fz);
    atomicAdd(&d_pe[i], (numtyp)0.5*pe);
    atomicAdd(&d_viral[i*6+0], virxx);
    atomicAdd(&d_viral[i*6+1], virxy);
    atomicAdd(&d_viral[i*6+2], virxz);
    atomicAdd(&d_viral[i*6+3], viryy);
    atomicAdd(&d_viral[i*6+4], viryz);
    atomicAdd(&d_viral[i*6+5], virzz);        

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_dpd_force
(
    const int N,              const int n_atomtypes,   
    const int n_max_neigh,    const int *d_n_neigh,     const int *d_neigh,
    const int *d_id,          const int *d_type,
    const numtyp *d_pos,      const numtyp *d_vel,   
    numtyp *d_force,         
    const unsigned int step,  const unsigned int seed,  const numtyp dt_1_2, 
    const numtyp lx,          const numtyp ly,          const numtyp lz,
    const numtyp hlx,         const numtyp hly,         const numtyp hlz,
    int threadsPerParticle
)
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x % threadsPerParticle; 
    int i = blockIdx.x * (blockDim.x / threadsPerParticle) + threadIdx.x / threadsPerParticle;  

    if (i >= N) return;

    /* ------------------------------------------------------- */

    const int id_i     = d_id[i];
    const int type_i   = d_type[i];
    const numtyp x_i   = d_pos[i*3+0];
    const numtyp y_i   = d_pos[i*3+1];
    const numtyp z_i   = d_pos[i*3+2];
    const numtyp vx_i  = d_vel[i*3+0];
    const numtyp vy_i  = d_vel[i*3+1];
    const numtyp vz_i  = d_vel[i*3+2];

    /* ------------------------------------------------------- */

    numtyp fx    = (numtyp)0.0;
    numtyp fy    = (numtyp)0.0;
    numtyp fz    = (numtyp)0.0;

    /* ------------------------------------------------------- */

    int neighPerThread = (d_n_neigh[i] + threadsPerParticle - 1) / threadsPerParticle;
    int start          = tid * neighPerThread;
    int end            = min(start + neighPerThread, d_n_neigh[i]);

    /* ------------------------------------------------------- */

    for (int k = start; k < end; k++)
    {
        /* --------------------------------------------------- */

        const int j = d_neigh[i * n_max_neigh + k];
        numtyp dx = x_i - d_pos[j*3+0];
        numtyp dy = y_i - d_pos[j*3+1];
        numtyp dz = z_i - d_pos[j*3+2];

        dx = dx - lx * ((dx >= hlx) - (dx < -hlx));
        dy = dy - ly * ((dy >= hly) - (dy < -hly));
        dz = dz - lz * ((dz >= hlz) - (dz < -hlz));
        const numtyp r2 = dx * dx + dy * dy + dz * dz;

        /* --------------------------------------------------- */

        const int type_j = d_type[j];
        int t_i = min(type_i, type_j);
        int t_j = max(type_i, type_j);
        const int index  = ((t_i - 1) * n_atomtypes - (t_i - 1) * (t_i - 2) / 2 + t_j - t_i + 1) - 1;
        const numtyp A_ij     = gpu_pair_coeff[index*3+0];
        const numtyp gamma_ij = gpu_pair_coeff[index*3+1];
        const numtyp rc       = gpu_pair_coeff[index*3+2];
        const numtyp sigma_ij_dt_1_2 = sqrt(gamma_ij) * dt_1_2;     

        /* --------------------------------------------------- */

        if (r2 <= rc*rc)
        { 
            /* ----------------------------------------------- */

            const numtyp r = sqrt(r2);
            if (r < 1.0e-10) continue;
            const numtyp r_inv = ((numtyp)1.0/r);

            /* ----------------------------------------------- */

            const numtyp dvx = vx_i - d_vel[j*3+0];
            const numtyp dvy = vy_i - d_vel[j*3+1];
            const numtyp dvz = vz_i - d_vel[j*3+2];
            const numtyp dot = dx * dvx + dy * dvy + dz * dvz;

            /* ----------------------------------------------- */

            const numtyp wc = (numtyp)1.0 - (r/rc);
            const numtyp wr = wc;

            /* ----------------------------------------------- */

            numtyp randnum    = (numtyp)0.0;
            unsigned int tag1 = min(id_i,  d_id[j]);
            unsigned int tag2 = max(id_i,  d_id[j]);
            saru(tag1, tag2, seed, step, randnum);  

            /* ----------------------------------------------- */

            // MDPD forces computation

            numtyp fpair = (A_ij * wc);
            fpair       -= (gamma_ij * wr * wr * dot * r_inv);
            fpair       += (sigma_ij_dt_1_2 * wr * randnum); 
            fpair       *= r_inv; 
            
            /* ----------------------------------------------- */

            const numtyp ffx = dx * fpair;
            const numtyp ffy = dy * fpair;
            const numtyp ffz = dz * fpair;
            fx += ffx;
            fy += ffy;
            fz += ffz;

            /* ----------------------------------------------- */

            if (j < N)
            {
                atomicAdd(&d_force[j*3+0], -ffx);
                atomicAdd(&d_force[j*3+1], -ffy);
                atomicAdd(&d_force[j*3+2], -ffz); 
            }

            /* ----------------------------------------------- */
        }
    }

    /* ------------------------------------------------------- */

    atomicAdd(&d_force[i*3+0], fx);
    atomicAdd(&d_force[i*3+1], fy);
    atomicAdd(&d_force[i*3+2], fz);      

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Pair_dpd::compute_force(System& system, unsigned int step)
{
    /* ------------------------------------------------------- */

    Atoms& atoms    = system.atoms;
    Box& box        = system.box;
    Run_p& run_p    = system.run_p;
    int N           = system.n_atoms;
    int n_atomtypes = system.n_atomtypes;
    int n_max_neigh = system.n_max_neigh;

    /* -------------------------------------------------------*/

    numtyp dtinv_1_2 = (numtyp)1.0 / sqrt(run_p.dt);
    numtyp dt_1_2 = sqrt((numtyp)2.0 * temp) * dtinv_1_2;    

    /* -------------------------------------------------------*/

    if (step % run_p.log_f == 0) 
    {
        CUDA_CHECK(cudaMemsetAsync(atoms.d_force, 0.0, sizeof(numtyp)*N*3));
        CUDA_CHECK(cudaMemsetAsync(atoms.d_pe,    0.0, sizeof(numtyp)*N));
        CUDA_CHECK(cudaMemsetAsync(atoms.d_viral, 0.0, sizeof(numtyp)*N*6));

        int blockSize = 256;
        int threadsPerParticle = 8;
        int particlesPerBlock = blockSize / threadsPerParticle;
        int numBlocks = (N + particlesPerBlock - 1) / particlesPerBlock;
        
        kernel_dpd_force_log<<<numBlocks, blockSize>>>
        (
            N,                    n_atomtypes,             
            n_max_neigh,          atoms.d_n_neigh,         atoms.d_neigh,     
            atoms.d_id,           atoms.d_type,
            atoms.d_pos,          atoms.d_vel,             
            atoms.d_force,        atoms.d_pe,              atoms.d_viral,                
            step,                 seed,                    dt_1_2, 
            box.lx,               box.ly,                  box.lz,
            box.hlx,              box.hly,                 box.hlz,
            threadsPerParticle
        );
    } 

    /* -------------------------------------------------------*/

    else 
    {
        CUDA_CHECK(cudaMemsetAsync(atoms.d_force, 0.0, sizeof(numtyp)*N*3));

        int blockSize = 256;
        int threadsPerParticle = 8;
        int particlesPerBlock = blockSize / threadsPerParticle;
        int numBlocks = (N + particlesPerBlock - 1) / particlesPerBlock;
        
        kernel_dpd_force<<<numBlocks, blockSize>>>
        (
            N,                    n_atomtypes,             
            n_max_neigh,          atoms.d_n_neigh,         atoms.d_neigh,     
            atoms.d_id,           atoms.d_type,
            atoms.d_pos,          atoms.d_vel,            
            atoms.d_force,                 
            step,                 seed,                    dt_1_2, 
            box.lx,               box.ly,                  box.lz,
            box.hlx,              box.hly,                 box.hlz,
            threadsPerParticle
        );        
    }

    /* -------------------------------------------------------*/
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////
REGISTER_PAIR(Pair_dpd)
///////////////////////

