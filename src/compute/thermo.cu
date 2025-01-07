#include "thermo.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

void Thermo::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;

    /* ------------------------------------------------------- */

    thermofile = fopen("thermo.txt", "w");

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMalloc(&thermo_p.d_total_ke,        sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&thermo_p.d_total_pair_pe,   sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&thermo_p.d_total_bond_pe,   sizeof(numtyp)));  
    CUDA_CHECK(cudaMalloc(&thermo_p.d_total_angle_pe,  sizeof(numtyp)));
    CUDA_CHECK(cudaMalloc(&thermo_p.d_pressure_tensor, sizeof(numtyp)*6));  
    
    /* ------------------------------------------------------- */

    thermo_p.thermo_temp     = 0.0;
    thermo_p.thermo_pair_pe  = 0.0;
    thermo_p.thermo_bond_pe  = 0.0;
    thermo_p.thermo_angle_pe = 0.0;
    thermo_p.thermo_pressure = 0.0;
    thermo_p.pressure_tensor.resize(6,0.0);
    
    /* ------------------------------------------------------- */

    printf("                                                                            \n");
    printf("/* ---------------------------------------------------------------------- */\n");
    printf("                          Thermo Information                                \n");
    printf("                                                                            \n");

    if (system.atom_style == 0) 
    {
        printf("%15s\t%10s\t%20s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n",
               "step", "temp", "pair_pe", "pressure", "pxx", "pxy", "pxz", "pyy", "pyz", "pzz");

        fprintf(thermofile, "%15s\t%10s\t%20s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n",
                "step", "temp", "pair_pe", "pressure", "pxx", "pxy", "pxz", "pyy", "pyz", "pzz");
    } 
    else if (system.atom_style == 1)
    {
        printf("%15s\t%10s\t%20s\t%20s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n",
               "step", "temp", "pair_pe", "bond_pe", "pressure", "pxx", "pxy", "pxz", "pyy", "pyz", "pzz");

        fprintf(thermofile, "%15s\t%10s\t%20s\t%20s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n",
                "step", "temp", "pair_pe", "bond_pe", "pressure", "pxx", "pxy", "pxz", "pyy", "pyz", "pzz");
    }
    else if (system.atom_style == 2)
    {
        printf("%15s\t%10s\t%20s\t%20s\t%20s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n",
               "step", "temp", "pair_pe", "bond_pe", "angle_pe", "pressure", "pxx", "pxy", "pxz", "pyy", "pyz", "pzz");

        fprintf(thermofile, "%15s\t%10s\t%20s\t%20s\t%20s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n",
               "step", "temp", "pair_pe", "bond_pe", "angle_pe", "pressure", "pxx", "pxy", "pxz", "pyy", "pyz", "pzz");
    }

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Thermo::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaFree(thermo_p.d_total_ke));
    CUDA_CHECK(cudaFree(thermo_p.d_total_pair_pe));
    CUDA_CHECK(cudaFree(thermo_p.d_total_bond_pe));
    CUDA_CHECK(cudaFree(thermo_p.d_total_angle_pe));
    CUDA_CHECK(cudaFree(thermo_p.d_pressure_tensor));
    
    /* ------------------------------------------------------- */

    if (thermofile != nullptr)
    {
        fclose(thermofile);
        thermofile = nullptr; 
    }

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Thermo::process(System& system, unsigned int step)
{
    /* ------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;
    Run_p& run_p       = system.run_p;

    /* ------------------------------------------------------- */

    if (step % run_p.log_f == 0)
    {
        /* --------------------------------------------------- */

        compute_temp(system);
        compute_pe(system);
        compute_pressure(system);

        if (system.atom_style == 1 || system.atom_style == 2) {
            compute_bond_pe(system);
        }
        if (system.atom_style == 2) {
            compute_angle_pe(system);
        }

        /* --------------------------------------------------- */

        if (system.atom_style == 0) 
        {
            printf("%15d\t%10.6f\t%20.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n",
                    step,
                    thermo_p.thermo_temp,
                    thermo_p.thermo_pair_pe,
                    thermo_p.thermo_pressure,
                    thermo_p.pressure_tensor[0],
                    thermo_p.pressure_tensor[1],
                    thermo_p.pressure_tensor[2],
                    thermo_p.pressure_tensor[3],
                    thermo_p.pressure_tensor[4],
                    thermo_p.pressure_tensor[5]);
            fflush(stdout);

            fprintf(thermofile,"%15d\t%10.6f\t%20.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n",
                                step,
                                thermo_p.thermo_temp,
                                thermo_p.thermo_pair_pe,
                                thermo_p.thermo_pressure,
                                thermo_p.pressure_tensor[0],
                                thermo_p.pressure_tensor[1],
                                thermo_p.pressure_tensor[2],
                                thermo_p.pressure_tensor[3],
                                thermo_p.pressure_tensor[4],
                                thermo_p.pressure_tensor[5]);
        } 

        /* --------------------------------------------------- */

        else if (system.atom_style == 1)
        {
            printf("%15d\t%10.6f\t%20.6f\t%20.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n",
                    step,
                    thermo_p.thermo_temp,
                    thermo_p.thermo_pair_pe,
                    thermo_p.thermo_bond_pe,
                    thermo_p.thermo_pressure,
                    thermo_p.pressure_tensor[0],
                    thermo_p.pressure_tensor[1],
                    thermo_p.pressure_tensor[2],
                    thermo_p.pressure_tensor[3],
                    thermo_p.pressure_tensor[4],
                    thermo_p.pressure_tensor[5]);
            fflush(stdout);
 
            fprintf(thermofile,"%15d\t%10.6f\t%20.6f\t%20.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n",
                                step,
                                thermo_p.thermo_temp,
                                thermo_p.thermo_pair_pe,
                                thermo_p.thermo_bond_pe,
                                thermo_p.thermo_pressure,
                                thermo_p.pressure_tensor[0],
                                thermo_p.pressure_tensor[1],
                                thermo_p.pressure_tensor[2],
                                thermo_p.pressure_tensor[3],
                                thermo_p.pressure_tensor[4],
                                thermo_p.pressure_tensor[5]);       
        }

        /* --------------------------------------------------- */

        else if (system.atom_style == 2)
        {
            printf("%15d\t%10.6f\t%20.6f\t%20.6f\t%20.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n",
                    step,
                    thermo_p.thermo_temp,
                    thermo_p.thermo_pair_pe,
                    thermo_p.thermo_bond_pe,
                    thermo_p.thermo_angle_pe,
                    thermo_p.thermo_pressure,
                    thermo_p.pressure_tensor[0],
                    thermo_p.pressure_tensor[1],
                    thermo_p.pressure_tensor[2],
                    thermo_p.pressure_tensor[3],
                    thermo_p.pressure_tensor[4],
                    thermo_p.pressure_tensor[5]);
            fflush(stdout);
 
            fprintf(thermofile,"%15d\t%10.6f\t%20.6f\t%20.6f\t%20.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n",
                                step,
                                thermo_p.thermo_temp,
                                thermo_p.thermo_pair_pe,
                                thermo_p.thermo_bond_pe,
                                thermo_p.thermo_angle_pe,
                                thermo_p.thermo_pressure,
                                thermo_p.pressure_tensor[0],
                                thermo_p.pressure_tensor[1],
                                thermo_p.pressure_tensor[2],
                                thermo_p.pressure_tensor[3],
                                thermo_p.pressure_tensor[4],
                                thermo_p.pressure_tensor[5]);       
        }

        /* --------------------------------------------------- */
    }

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_compute_total_ke
(
    const int N,
    const int *d_type, 
    const numtyp *d_vel, 
    numtyp *d_total_ke
) 
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdata[128]; 

    /* ------------------------------------------------------- */

    numtyp p_ke = (numtyp)0.0;
    if (i < N) 
    {
        p_ke = (numtyp)0.5 * masses[d_type[i]-1] * (d_vel[i*3+0] * d_vel[i*3+0] 
                                                 +  d_vel[i*3+1] * d_vel[i*3+1] 
                                                 +  d_vel[i*3+2] * d_vel[i*3+2]);
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
        atomicAdd(d_total_ke, sdata[0]);
    }

    /* ------------------------------------------------------- */
}

/* ------------------------------------------------------------ */

void Thermo::compute_temp(System& system)
{
    /* -------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;
    Atoms& atoms       = system.atoms;

    /* -------------------------------------------------------- */

    CUDA_CHECK(cudaMemset(thermo_p.d_total_ke, (numtyp)0.0, sizeof(numtyp)));

    int N           = system.n_atoms;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    kernel_compute_total_ke<<<numBlocks, blockSize>>>
    (
        N,
        atoms.d_type, 
        atoms.d_vel,
        thermo_p.d_total_ke
    );

    /* -------------------------------------------------------- */

    numtyp total_ke = (numtyp)0.0;
    CUDA_CHECK(cudaMemcpy(&total_ke, thermo_p.d_total_ke, sizeof(numtyp), cudaMemcpyDeviceToHost));

    thermo_p.thermo_temp = (2 * total_ke) / (3 * system.n_atoms * 1);

    /* -------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_compute_total_pair_pe
(
    const int N,
    const numtyp *d_pe,
    numtyp *d_total_pair_pe
) 
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdata[128]; 

    /* ------------------------------------------------------- */

    numtyp p_pe = (numtyp)0.0;
    if (i < N) 
    {
        p_pe = d_pe[i];
    }
    sdata[tid] = p_pe;
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
        atomicAdd(d_total_pair_pe, sdata[0]);
    }

    /* ------------------------------------------------------- */
}

/* ------------------------------------------------------------ */

void Thermo::compute_pe(System& system)
{
    /* -------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;
    Atoms& atoms       = system.atoms;

    /* -------------------------------------------------------- */

    CUDA_CHECK(cudaMemset(thermo_p.d_total_pair_pe, (numtyp)0.0, sizeof(numtyp)));

    int N           = system.n_atoms;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    kernel_compute_total_pair_pe<<<numBlocks, blockSize>>>
    (
        N,
        atoms.d_pe,
        thermo_p.d_total_pair_pe
    );


    /* -------------------------------------------------------- */

    numtyp total_pair_pe = (numtyp)0.0;
    CUDA_CHECK(cudaMemcpy(&total_pair_pe, thermo_p.d_total_pair_pe, sizeof(numtyp), cudaMemcpyDeviceToHost));

    thermo_p.thermo_pair_pe = total_pair_pe;

    /* -------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_compute_pressure
(
    const int N,
    const int *d_type, 
    const numtyp *d_vel,
    const numtyp *d_viral,
    numtyp *d_pressure_tensor
) 
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    __shared__ numtyp pxx[128]; 
    __shared__ numtyp pxy[128]; 
    __shared__ numtyp pxz[128]; 
    __shared__ numtyp pyy[128]; 
    __shared__ numtyp pyz[128]; 
    __shared__ numtyp pzz[128]; 

    /* ------------------------------------------------------- */

    numtyp per_atom_pxx = (numtyp)0.0;
    numtyp per_atom_pxy = (numtyp)0.0;
    numtyp per_atom_pxz = (numtyp)0.0;
    numtyp per_atom_pyy = (numtyp)0.0;
    numtyp per_atom_pyz = (numtyp)0.0;
    numtyp per_atom_pzz = (numtyp)0.0;
    
    if (i < N) 
    {
        numtyp mass_i = masses[d_type[i]-1];

        per_atom_pxx = mass_i * d_vel[i*3+0] * d_vel[i*3+0] + d_viral[i*6+0];
        per_atom_pxy = mass_i * d_vel[i*3+0] * d_vel[i*3+1] + d_viral[i*6+1];
        per_atom_pxz = mass_i * d_vel[i*3+0] * d_vel[i*3+2] + d_viral[i*6+2];

        per_atom_pyy = mass_i * d_vel[i*3+1] * d_vel[i*3+1] + d_viral[i*6+3];
        per_atom_pyz = mass_i * d_vel[i*3+1] * d_vel[i*3+2] + d_viral[i*6+4];

        per_atom_pzz = mass_i * d_vel[i*3+2] * d_vel[i*3+2] + d_viral[i*6+5];
    }

    pxx[tid] = per_atom_pxx;
    pxy[tid] = per_atom_pxy;
    pxz[tid] = per_atom_pxz;
    pyy[tid] = per_atom_pyy;
    pyz[tid] = per_atom_pyz;
    pzz[tid] = per_atom_pzz;

    __syncthreads();

    /* ------------------------------------------------------- */

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            pxx[tid] += pxx[tid + s];
            pxy[tid] += pxy[tid + s];
            pxz[tid] += pxz[tid + s];
            pyy[tid] += pyy[tid + s];
            pyz[tid] += pyz[tid + s];
            pzz[tid] += pzz[tid + s];
        }
        __syncthreads();
    }

    /* ------------------------------------------------------- */

    if (tid == 0) 
    {
        atomicAdd(&d_pressure_tensor[0], pxx[0]);
        atomicAdd(&d_pressure_tensor[1], pxy[0]);
        atomicAdd(&d_pressure_tensor[2], pxz[0]);
        atomicAdd(&d_pressure_tensor[3], pyy[0]);
        atomicAdd(&d_pressure_tensor[4], pyz[0]);
        atomicAdd(&d_pressure_tensor[5], pzz[0]);
    }

    /* ------------------------------------------------------- */
}

/* ------------------------------------------------------------ */

void Thermo::compute_pressure(System& system)
{
    /* -------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;
    Atoms& atoms       = system.atoms;
    Box& box           = system.box;

    /* -------------------------------------------------------- */

    CUDA_CHECK(cudaMemset(thermo_p.d_pressure_tensor, (numtyp)0.0, sizeof(numtyp)*6));

    int N           = system.n_atoms;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    kernel_compute_pressure<<<numBlocks, blockSize>>>
    (
        N,
        atoms.d_type, 
        atoms.d_vel,
        atoms.d_viral,
        thermo_p.d_pressure_tensor
    );

    /* -------------------------------------------------------- */

    numtyp volume = box.lx * box.ly * box.lz;
    std::vector<numtyp> tensor(6, 0.0);

    CUDA_CHECK(cudaMemcpy(tensor.data(), thermo_p.d_pressure_tensor, sizeof(numtyp)*6, cudaMemcpyDeviceToHost));

    /* -------------------------------------------------------- */

    for (int i = 0; i < 6; i++) 
    {
        thermo_p.pressure_tensor[i] = tensor[i] / volume;
    }
    thermo_p.thermo_pressure = (thermo_p.pressure_tensor[0] +  thermo_p.pressure_tensor[3] + thermo_p.pressure_tensor[5]) / 3.0;

    /* -------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_compute_total_bond_pe
(
    const int N,
    const numtyp *d_bond_pe,  
    numtyp *d_total_bond_pe
) 
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdata[128]; 

    /* ------------------------------------------------------- */

    numtyp p_pe = (numtyp)0.0;
    if (i < N) 
    {
        p_pe = d_bond_pe[i];
    }
    sdata[tid] = p_pe;
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
        atomicAdd(d_total_bond_pe, sdata[0]);
    }

    /* ------------------------------------------------------- */
}

/* ------------------------------------------------------------ */

void Thermo::compute_bond_pe(System& system)
{
    /* -------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;
    Atoms& atoms       = system.atoms;

    /* -------------------------------------------------------- */

    CUDA_CHECK(cudaMemset(thermo_p.d_total_bond_pe, (numtyp)0.0, sizeof(numtyp)));

    int N           = system.n_bonds;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    kernel_compute_total_bond_pe<<<numBlocks, blockSize>>>
    (
        N,
        atoms.d_bond_pe, 
        thermo_p.d_total_bond_pe
    );
    
    /* -------------------------------------------------------- */

    numtyp total_bond_pe = (numtyp)0.0;
    CUDA_CHECK(cudaMemcpy(&total_bond_pe, thermo_p.d_total_bond_pe, sizeof(numtyp), cudaMemcpyDeviceToHost));

    thermo_p.thermo_bond_pe = total_bond_pe;

    /* -------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_compute_total_angle_pe
(
    const int N,
    const numtyp *d_angle_pe,  
    numtyp *d_total_angle_pe
) 
{
    /* ------------------------------------------------------- */

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    __shared__ numtyp sdata[128]; 

    /* ------------------------------------------------------- */

    numtyp p_pe = (numtyp)0.0;
    if (i < N) 
    {
        p_pe = d_angle_pe[i];
    }
    sdata[tid] = p_pe;
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
        atomicAdd(d_total_angle_pe, sdata[0]);
    }

    /* ------------------------------------------------------- */
}

/* ------------------------------------------------------------ */

void Thermo::compute_angle_pe(System& system)
{
    /* -------------------------------------------------------- */

    Thermo_p& thermo_p = system.thermo_p;
    Atoms& atoms       = system.atoms;

    /* -------------------------------------------------------- */

    CUDA_CHECK(cudaMemset(thermo_p.d_total_angle_pe, (numtyp)0.0, sizeof(numtyp)));

    int N           = system.n_angles;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    kernel_compute_total_angle_pe<<<numBlocks, blockSize>>>
    (
        N,
        atoms.d_angle_pe, 
        thermo_p.d_total_angle_pe
    );
    
    /* -------------------------------------------------------- */

    numtyp total_angle_pe = (numtyp)0.0;
    CUDA_CHECK(cudaMemcpy(&total_angle_pe, thermo_p.d_total_angle_pe, sizeof(numtyp), cudaMemcpyDeviceToHost));

    thermo_p.thermo_angle_pe = total_angle_pe;

    /* -------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

