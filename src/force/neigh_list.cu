#include "force/neigh_list.cuh"

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::preprocess(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;
    Run_p& run_p   = system.run_p;

    /* ------------------------------------------------------- */
    
    neigh.bin_size       = run_p.global_cut + run_p.skin;
    numtyp sphere_volume = 4.0 / 3.0 * M_PI * pow(neigh.bin_size, 3.0);  
    system.n_max_neigh   = static_cast<int>(run_p.max_rho * sphere_volume) + 1;
    system.n_max_neigh   = ((system.n_max_neigh + 15) / 16) * 16;  
    
    /* ------------------------------------------------------- */

    neigh_mem_alloc1(system);
    initialize_boxes(system);
    print_neigh_info(system);
    neigh_mem_alloc2(system);
    get_neighborbins(system);

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::neigh_mem_alloc1(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;
    Run_p& run_p   = system.run_p;

    /* ------------------------------------------------------- */

    neigh.n_totalbinsxyz.resize(3); 
    neigh.n_totalbins=0; 

    /* ------------------------------------------------------- */

    if (run_p.skin != static_cast<numtyp>(0.0)) {
        check_update_pre(system);
    }

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::initialize_boxes(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;
    Box& box       = system.box;

    /* ------------------------------------------------------- */

    neigh.n_totalbinsxyz[0] = int((box.xhi - box.xlo) / neigh.bin_size);
    neigh.n_totalbinsxyz[1] = int((box.yhi - box.ylo) / neigh.bin_size);
    neigh.n_totalbinsxyz[2] = int((box.zhi - box.zlo) / neigh.bin_size);

    neigh.n_totalbins = neigh.n_totalbinsxyz[0] * neigh.n_totalbinsxyz[1] * neigh.n_totalbinsxyz[2];

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::print_neigh_info(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;
    Run_p& run_p   = system.run_p;

    /* ------------------------------------------------------- */

    // print info header

    std::cout << "                                                                            " << std::endl;
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                           Neigh Information                                " << std::endl;
    std::cout << "                                                                            " << std::endl;
       
    std::cout << "   n_totalbinsxyz: ";
    for (int i = 0; i < 3; ++i)
    {
        std::cout << neigh.n_totalbinsxyz[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "   n_totalbins: ";
    std::cout << neigh.n_totalbins << " \n";

    std::cout << "   max neighbor number = "     << system.n_max_neigh                          << std::endl;
    std::cout << "   bin_size = "                << neigh.bin_size                              << std::endl;

    if (run_p.skin != static_cast<numtyp>(0.0)) {
        std::cout << "   check neighbor every "  << run_p.nl_f  << " step"                      << std::endl;
    } else {
        std::cout << "   update neighbor every " << run_p.nl_f  << " step"                      << std::endl;
    }
    
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl;

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::neigh_mem_alloc2(System& system)
{
    /* ------------------------------------------------------- */

    Atoms& atoms    = system.atoms;
    Neigh& neigh    = system.neigh;
    int N           = system.n_atoms;
    int n_max_neigh = system.n_max_neigh;

    /* ------------------------------------------------------- */

    neigh.h_neighborbinsID.resize(27*neigh.n_totalbins);

    CUDA_CHECK(cudaMalloc(&neigh.d_neighborbinsID,     sizeof(int)*27*neigh.n_totalbins));
    CUDA_CHECK(cudaMalloc(&neigh.binContent,           sizeof(int)*N));
    CUDA_CHECK(cudaMalloc(&neigh.binCounts,            sizeof(int)*neigh.n_totalbins));
    CUDA_CHECK(cudaMalloc(&neigh.binCounts_prefix_sum, sizeof(int)*neigh.n_totalbins));

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMalloc(&atoms.d_n_neigh,            sizeof(int)*N));   
    CUDA_CHECK(cudaMalloc(&atoms.d_neigh,              sizeof(int)*N*n_max_neigh)); 

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::get_neighborbins(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;

    /* ------------------------------------------------------- */

    int nBinsX =  neigh.n_totalbinsxyz[0];
    int nBinsY =  neigh.n_totalbinsxyz[1];
    int nBinsZ =  neigh.n_totalbinsxyz[2];

    for (int i = 0; i < neigh.n_totalbins; i++) 
    {
        int binZ =  i % nBinsZ;
        int binY = (i / nBinsZ) % nBinsY;
        int binX =  i / (nBinsZ * nBinsY);
        int k = 0;
        for (int dz = -1; dz <= 1; ++dz) 
        {
            for (int dy = -1; dy <= 1; ++dy) 
            {
                for (int dx = -1; dx <= 1; ++dx) 
                {
                    int neighborX = (binX + dx + nBinsX) % nBinsX;
                    int neighborY = (binY + dy + nBinsY) % nBinsY;
                    int neighborZ = (binZ + dz + nBinsZ) % nBinsZ;
                    int neighborIndex = neighborX * nBinsY * nBinsZ + neighborY * nBinsZ + neighborZ;
                    neigh.h_neighborbinsID[i * 27 + k] = neighborIndex;
                    k = k + 1;
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(neigh.d_neighborbinsID, neigh.h_neighborbinsID.data(), sizeof(int)*27*neigh.n_totalbins, cudaMemcpyHostToDevice));

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::check_update_pre(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;
    int N          = system.n_atoms;

    /* ------------------------------------------------------- */

    neigh.h_update_flag = 0;

    CUDA_CHECK(cudaMalloc(&neigh.d_update_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&neigh.last_pos,      sizeof(numtyp)*N*3));

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::check_update_post(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaFree(neigh.d_update_flag));
    CUDA_CHECK(cudaFree(neigh.last_pos));

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::postprocess(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;
    Run_p& run_p   = system.run_p;
    Atoms& atoms   = system.atoms;

    /* ------------------------------------------------------- */

    neigh.n_totalbinsxyz.clear();
    neigh.h_neighborbinsID.clear();

    CUDA_CHECK(cudaFree(neigh.d_neighborbinsID));
    CUDA_CHECK(cudaFree(neigh.binContent));
    CUDA_CHECK(cudaFree(neigh.binCounts));
    CUDA_CHECK(cudaFree(neigh.binCounts_prefix_sum));

    CUDA_CHECK(cudaFree(atoms.d_n_neigh));
    CUDA_CHECK(cudaFree(atoms.d_neigh));

    /* ------------------------------------------------------- */

    if (run_p.skin != static_cast<numtyp>(0.0)) {
        check_update_post(system);
    }

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_binCounts
(
    const int N,                const numtyp bin_size,     
    const int nBinsX,           const int nBinsY,          const int nBinsZ, 
    const numtyp xlo,           const numtyp ylo,          const numtyp zlo,          
    const numtyp *d_pos,        
    int *binCounts
)
{
    /* ------------------------------------------------------- */

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    /* ------------------------------------------------------- */

    int binX = (int)(((d_pos[i*3+0] - xlo) / bin_size));
    int binY = (int)(((d_pos[i*3+1] - ylo) / bin_size));   
    int binZ = (int)(((d_pos[i*3+2] - zlo) / bin_size)); 

    binX     = min(max(binX, 0), nBinsX - 1);  
    binY     = min(max(binY, 0), nBinsY - 1);
    binZ     = min(max(binZ, 0), nBinsZ - 1);   

    const int binIndex = binX * nBinsY * nBinsZ + binY * nBinsZ + binZ;
    atomicAdd(&binCounts[binIndex], 1);

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_assignAtomsToBins
(
    const int N,                const numtyp bin_size,     
    const int nBinsX,           const int nBinsY,               const int nBinsZ, 
    const numtyp xlo,           const numtyp ylo,               const numtyp zlo,    
    const numtyp *d_pos,            
    int *binCounts,             int *binCounts_prefix_sum,      int *binContent
)
{
    /* ------------------------------------------------------- */

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    /* ------------------------------------------------------- */

    int binX = (int)(((d_pos[i*3+0] - xlo) / bin_size));
    int binY = (int)(((d_pos[i*3+1] - ylo) / bin_size));   
    int binZ = (int)(((d_pos[i*3+2] - zlo) / bin_size)); 

    binX     = min(max(binX, 0), nBinsX - 1);  
    binY     = min(max(binY, 0), nBinsY - 1);
    binZ     = min(max(binZ, 0), nBinsZ - 1);   

    const int binIndex = binX * nBinsY * nBinsZ + binY * nBinsZ + binZ;
    int offset = atomicAdd(&binCounts[binIndex], 1);
    binContent[binCounts_prefix_sum[binIndex] + offset] = i;

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_build
(
    const int nbins,            const int N,
    const int *binCounts,       const int *binCounts_prefix_sum,   const int *binContent,
    const int *neighborbinsID,  const numtyp bin_size,             const numtyp *d_pos, 
    const int n_max_neigh,      int *d_n_neigh,                    int *d_neigh, 
    const numtyp lx,            const numtyp ly,                   const numtyp lz,
    const numtyp hlx,           const numtyp hly,                  const numtyp hlz,
    int threadsPerBin
)
{
    /* ------------------------------------------------------- */

    int tid    = threadIdx.x % threadsPerBin;  
    int binId  = blockIdx.x * (blockDim.x / threadsPerBin) + threadIdx.x / threadsPerBin; 

    if (binId >= nbins) return;

    /* ------------------------------------------------------- */

    const int atomCountInBin = binCounts[binId];
    if (atomCountInBin == 0) return;

    int atomsPerThread = (atomCountInBin + threadsPerBin - 1) / threadsPerBin;
    int startAtom      = tid * atomsPerThread;
    int endAtom        = min((tid + 1) * atomsPerThread, atomCountInBin);

    /* ------------------------------------------------------- */

    for (int offset = startAtom; offset < endAtom; offset++) 
    {
        const int i      = binContent[binCounts_prefix_sum[binId] + offset];
        const numtyp x_i = d_pos[i*3+0];
        const numtyp y_i = d_pos[i*3+1];
        const numtyp z_i = d_pos[i*3+2];

        int count = 0;

        for (int k = 0; k < 27; k++) 
        {
            const int neighborBinIndex       = neighborbinsID[binId * 27 + k];
            const int atomCountInNeighborBin = binCounts[neighborBinIndex];

            for (int j = 0; j < atomCountInNeighborBin; j++) 
            {
                const int neighborAtomID = binContent[binCounts_prefix_sum[neighborBinIndex] + j];
                
                /* ------------------------------------------------- */
                if (i >= neighborAtomID) continue;   // half neigh_list
                /* ------------------------------------------------- */

                numtyp dx = x_i - d_pos[neighborAtomID*3+0];
                numtyp dy = y_i - d_pos[neighborAtomID*3+1];
                numtyp dz = z_i - d_pos[neighborAtomID*3+2];

                dx = dx - lx * ((dx >= hlx) - (dx < -hlx));
                dy = dy - ly * ((dy >= hly) - (dy < -hly));
                dz = dz - lz * ((dz >= hlz) - (dz < -hlz));
                const numtyp r2 = dx * dx + dy * dy + dz * dz;

                if (r2 <= bin_size * bin_size) 
                {
                    d_neigh[count * N + i] = neighborAtomID;
                    count += 1;
                } 
            }
        }
        d_n_neigh[i] = count;
    }

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::build(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh    = system.neigh;
    Atoms& atoms    = system.atoms;
    Box& box        = system.box;
    Run_p& run_p    = system.run_p;
    int N           = system.n_atoms;
    int n_max_neigh = system.n_max_neigh;

    /* ------------------------------------------------------- */

    // Count

    CUDA_CHECK(cudaMemset(neigh.binCounts,            0, sizeof(int)*neigh.n_totalbins));
    CUDA_CHECK(cudaMemset(neigh.binCounts_prefix_sum, 0, sizeof(int)*neigh.n_totalbins));

    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    kernel_binCounts<<<numBlocks, blockSize>>>
    (
        N,                           neigh.bin_size,                
        neigh.n_totalbinsxyz[0],     neigh.n_totalbinsxyz[1],   neigh.n_totalbinsxyz[2],
        box.xlo,                     box.ylo,                   box.zlo,                       
        atoms.d_pos,              
        neigh.binCounts
    ); 
    
    // Prefix-Sums
    thrust::device_ptr<int> dev_binCounts(neigh.binCounts);
    thrust::device_ptr<int> dev_binCounts_prefix_sum(neigh.binCounts_prefix_sum);
    thrust::exclusive_scan(
        dev_binCounts, dev_binCounts + neigh.n_totalbins,
        dev_binCounts_prefix_sum
    );

    /* ------------------------------------------------------- */

    // Sort atoms to bins

    CUDA_CHECK(cudaMemset(neigh.binCounts, 0, sizeof(int)*neigh.n_totalbins));

    kernel_assignAtomsToBins<<<numBlocks, blockSize>>>
    (
        N,                           neigh.bin_size,                
        neigh.n_totalbinsxyz[0],     neigh.n_totalbinsxyz[1],     neigh.n_totalbinsxyz[2],
        box.xlo,                     box.ylo,                     box.zlo,                       
        atoms.d_pos, 
        neigh.binCounts,             neigh.binCounts_prefix_sum,  neigh.binContent
    );

    /* ------------------------------------------------------- */
    
    // Build neighbor list

    CUDA_CHECK(cudaMemset(atoms.d_n_neigh,  0, sizeof(int)*N));   
    CUDA_CHECK(deviceFill<int>(atoms.d_neigh, -1, N * n_max_neigh));

    int threadsPerBin = 16;
    int BinsPerBlock  = blockSize / threadsPerBin;
    numBlocks         = (neigh.n_totalbins + BinsPerBlock - 1) / BinsPerBlock;

    kernel_build<<<numBlocks,blockSize>>>
    (
        neigh.n_totalbins,       N,
        neigh.binCounts,         neigh.binCounts_prefix_sum,    neigh.binContent,      
        neigh.d_neighborbinsID,  neigh.bin_size,                atoms.d_pos,
        system.n_max_neigh,      atoms.d_n_neigh,               atoms.d_neigh,
        box.lx,                  box.ly,                        box.lz,
        box.hlx,                 box.hly,                       box.hlz,
        threadsPerBin
    );  

    /* ------------------------------------------------------- */

    if (run_p.skin != static_cast<numtyp>(0.0)) {
        CUDA_CHECK(cudaMemcpy(neigh.last_pos, atoms.d_pos, sizeof(numtyp)*N*3, cudaMemcpyDeviceToDevice));
    }

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

static __global__ void kernel_check_update
(
    const int N, 
    const numtyp *d_pos,     const numtyp *last_pos, 
    const numtyp hskin_sq,   int *update_flag,
    const numtyp lx,         const numtyp ly,          const numtyp lz,
    const numtyp hlx,        const numtyp hly,         const numtyp hlz
)
{
    /* ------------------------------------------------------- */

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    /* ------------------------------------------------------- */

    numtyp dx = d_pos[i*3+0] - last_pos[i*3+0];
    numtyp dy = d_pos[i*3+1] - last_pos[i*3+1];
    numtyp dz = d_pos[i*3+2] - last_pos[i*3+2];

    dx = dx - lx * ((dx >= hlx) - (dx < -hlx));
    dy = dy - ly * ((dy >= hly) - (dy < -hly));
    dz = dz - lz * ((dz >= hlz) - (dz < -hlz));
    const numtyp r2 = dx * dx + dy * dy + dz * dz;

    if (r2 >= hskin_sq) {
        atomicExch(update_flag, 1);
    }

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Neigh_list::check_update(System& system)
{
    /* ------------------------------------------------------- */

    Neigh& neigh   = system.neigh;
    Atoms& atoms   = system.atoms;
    Box& box       = system.box;
    Run_p& run_p   = system.run_p;

    /* ------------------------------------------------------- */

    neigh.h_update_flag = 0;
    CUDA_CHECK(cudaMemset(neigh.d_update_flag, 0, sizeof(int)));

    int N           = system.n_atoms;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;    
    numtyp hskin_sq = (run_p.skin * static_cast<numtyp>(0.5)) * (run_p.skin * static_cast<numtyp>(0.5));

    kernel_check_update<<<numBlocks, blockSize>>>
    (
        N, 
        atoms.d_pos,       neigh.last_pos,
        hskin_sq,          neigh.d_update_flag,
        box.lx,            box.ly,                      box.lz,
        box.hlx,           box.hly,                     box.hlz
    );

    /* ------------------------------------------------------- */

    CUDA_CHECK(cudaMemcpy(&neigh.h_update_flag, neigh.d_update_flag, sizeof(int), cudaMemcpyDeviceToHost));
    
    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */