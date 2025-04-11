#include "force/angle_harmonic.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

Angle_harmonic::Angle_harmonic() 
{
    angle_style_name = "harmonic";
}

/* ----------------------------------------------------------------------------------------------------------- */

std::string Angle_harmonic::getName() 
{
    return angle_style_name;
}

/* ----------------------------------------------------------------------------------------------------------- */\

bool Angle_harmonic::isEnabled(System& system) 
{
    /* ------------------------------------------------------- */

    std::string& input = system.input; 
    int n_angletypes    = system.n_angletypes;
    bool enabled       = false;

    /* ------------------------------------------------------- */

    // angle_coeff

    got_angle_coeff.resize(n_angletypes, 0);
    angle_coeff.resize(got_angle_coeff.size()*2); 

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

        // read angle_style
        
        if (key == "angle_style") 
        {
            if (args[0] == "harmonic") {enabled = true;}

            if (enabled)
            {
                std::string error    = "illegal angle_style harmonic command";
                std::string format   = "angle_style angle_style";
                std::string example0 = "angle_style harmonic";
                if (args.size() != 1) {
                    print_error_and_exit(line, error, format, {example0});
                }
            }
        }
    }

    /* ------------------------------------------------------- */

    // read angle_coeff

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
    
            if (key == "angle_coeff") 
            {
                std::string error    = "illegal angle_coeff command for angle_style harmonic";
                std::string format   = "angle_coeff type type k theta0";
                std::string example0 = "angle_coeff 1 30.0 120.0";

                if (args.size() != 3) {
                    print_error_and_exit(line, error, format, {example0});
                }

                int type = parse_int<int>(args[0], line, "type", {format, example0});

                if (type > n_angletypes) {
                    print_error({"type exceeds n_angle_types"});                     
                } 

                numtyp k      = parse_float<numtyp>(args[1], line, "k", {format, example0});
                numtyp theta0 = parse_float<numtyp>(args[2], line, "theta0", {format, example0});

                int index  = type - 1;
                got_angle_coeff[index] = 1;
                angle_coeff[index*2+0] = k;
                angle_coeff[index*2+1] = theta0;
            }
        } 
    }

    /* ------------------------------------------------------- */

    // check all angle_coeff

    if (enabled) 
    {
        bool all_angle_coeff = true;
        for (int i=0; i < got_angle_coeff.size(); i++) 
        {
            if (got_angle_coeff[i] == 0) 
            {
                all_angle_coeff = false;
            }
        }
        if (!all_angle_coeff) 
        {
            std::string error = "all angle_coeff are not set";
            print_error({error});      
        } else {
            CUDA_CHECK(cudaMemcpyToSymbol(gpu_angle_coeff, angle_coeff.data(), (angle_coeff.size() * sizeof(numtyp))));       
        }

    }

    /* ------------------------------------------------------- */

    return enabled;

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Angle_harmonic::print_angle_info(System& system) 
{
    std::cout << "   angle_style           harmonic" << std::endl;
    std::cout << "   angle_coeff           type k theta0" << std::endl;
    for (int i=0; i<system.n_angletypes; i++)
    {
        int type = i+1;
        int _id  = type-1;
        std::cout << "   angle_coeff           " << type << " " << angle_coeff[_id*2+0] << " " << angle_coeff[_id*2+1] << std::endl;
    }
    std::cout << "/* ---------------------------------------------------------------------- */" << std::endl;
    std::cout << "                                                                            " << std::endl; 
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_angle_harmonic_log
(
    const int N,                  
    const int *d_id,              const int *d_anglelist, 
    const numtyp *d_pos,          numtyp *d_force,               numtyp *d_angle_pe,            numtyp *d_viral,
    const numtyp lx,              const numtyp ly,               const numtyp lz,
    const numtyp hlx,             const numtyp hly,              const numtyp hlz
)
{
    /* ------------------------------------------------------- */

    const int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    /* ------------------------------------------------------- */

    const int angle_type     = d_anglelist[i*4+0];
    const int angle_atomi_id = d_anglelist[i*4+1];
    const int angle_atomj_id = d_anglelist[i*4+2];
    const int angle_atomk_id = d_anglelist[i*4+3];

    /* ------------------------------------------------------- */

    const int angle_type_index = angle_type - 1; 
    const numtyp spring_k      = gpu_angle_coeff[angle_type_index*2+0]; 
    const numtyp theta_deg     = gpu_angle_coeff[angle_type_index*2+1]; 
    const numtyp theta0        = theta_deg * (M_PI / 180.0);

    /* ------------------------------------------------------- */

    const int angle_atomi_index = angle_atomi_id - 1;
    const numtyp pos_x_i        = d_pos[angle_atomi_index*3+0];
    const numtyp pos_y_i        = d_pos[angle_atomi_index*3+1];
    const numtyp pos_z_i        = d_pos[angle_atomi_index*3+2];

    const int angle_atomj_index = angle_atomj_id - 1;
    const numtyp pos_x_j        = d_pos[angle_atomj_index*3+0];
    const numtyp pos_y_j        = d_pos[angle_atomj_index*3+1];
    const numtyp pos_z_j        = d_pos[angle_atomj_index*3+2];

    const int angle_atomk_index = angle_atomk_id - 1;
    const numtyp pos_x_k        = d_pos[angle_atomk_index*3+0];
    const numtyp pos_y_k        = d_pos[angle_atomk_index*3+1];
    const numtyp pos_z_k        = d_pos[angle_atomk_index*3+2];

    /* ------------------------------------------------------- */

    numtyp dxij = pos_x_i - pos_x_j;
    numtyp dyij = pos_y_i - pos_y_j;
    numtyp dzij = pos_z_i - pos_z_j;    

    dxij = dxij - lx * ((dxij >= hlx) - (dxij < -hlx));
    dyij = dyij - ly * ((dyij >= hly) - (dyij < -hly));
    dzij = dzij - lz * ((dzij >= hlz) - (dzij < -hlz));

    numtyp dxkj = pos_x_k - pos_x_j;
    numtyp dykj = pos_y_k - pos_y_j;
    numtyp dzkj = pos_z_k - pos_z_j;    

    dxkj = dxkj - lx * ((dxkj >= hlx) - (dxkj < -hlx));
    dykj = dykj - ly * ((dykj >= hly) - (dykj < -hly));
    dzkj = dzkj - lz * ((dzkj >= hlz) - (dzkj < -hlz));
    
    numtyp rsq1 = dxij * dxij + dyij * dyij + dzij * dzij;
    numtyp rsq2 = dxkj * dxkj + dykj * dykj + dzkj * dzkj;

    numtyp r1 = sqrt(rsq1);
    numtyp r2 = sqrt(rsq2);

    /* --------------------------------------------------- */

    numtyp dot    = dxij * dxkj + dyij * dykj + dzij * dzkj;
    numtyp ctheta = dot / (r1 * r2);

    if (ctheta > (numtyp)1.0)  ctheta = (numtyp)1.0;
    if (ctheta < (numtyp)-1.0) ctheta = (numtyp)-1.0;

    numtyp sintheta = sqrt(1.0 - ctheta*ctheta);
    numtyp SMALL    = 0.001;
    if (sintheta < SMALL) sintheta = SMALL;

    numtyp theta  = acos(ctheta);
    numtyp dtheta = theta - theta0;

    /* --------------------------------------------------- */

    numtyp a   = -2.0 * spring_k * dtheta / sintheta;
    numtyp a11 = a * ctheta / rsq1;
    numtyp a12 = -a / (r1 * r2);
    numtyp a22 = a * ctheta / rsq2;
    
    numtyp fijx = a11 * dxij + a12 * dxkj;
    numtyp fijy = a11 * dyij + a12 * dykj;
    numtyp fijz = a11 * dzij + a12 * dzkj;

    numtyp fkjx = a22 * dxkj + a12 * dxij;
    numtyp fkjy = a22 * dykj + a12 * dyij;
    numtyp fkjz = a22 * dzkj + a12 * dzij;

    atomicAdd(&d_force[angle_atomi_index*3+0], fijx);
    atomicAdd(&d_force[angle_atomi_index*3+1], fijy);
    atomicAdd(&d_force[angle_atomi_index*3+2], fijz);

    atomicAdd(&d_force[angle_atomk_index*3+0], fkjx);
    atomicAdd(&d_force[angle_atomk_index*3+1], fkjy);
    atomicAdd(&d_force[angle_atomk_index*3+2], fkjz);

    numtyp fjx = - (fijx + fkjx);
    numtyp fjy = - (fijy + fkjy);
    numtyp fjz = - (fijz + fkjz);

    atomicAdd(&d_force[angle_atomj_index*3+0], fjx);
    atomicAdd(&d_force[angle_atomj_index*3+1], fjy);
    atomicAdd(&d_force[angle_atomj_index*3+2], fjz);

    /* --------------------------------------------------- */

    numtyp angle_pe = spring_k * dtheta * dtheta;
    atomicAdd(&d_angle_pe[i], angle_pe);

    /* --------------------------------------------------- */

    numtyp vxx_ij = dxij * fijx;
    numtyp vxy_ij = dxij * fijy;
    numtyp vxz_ij = dxij * fijz;
    numtyp vyy_ij = dyij * fijy;
    numtyp vyz_ij = dyij * fijz;
    numtyp vzz_ij = dzij * fijz;

    numtyp vxx_kj = dxkj * fkjx;
    numtyp vxy_kj = dxkj * fkjy;
    numtyp vxz_kj = dxkj * fkjz;
    numtyp vyy_kj = dykj * fkjy;
    numtyp vyz_kj = dykj * fkjz;
    numtyp vzz_kj = dzkj * fkjz;

    numtyp vxx = (vxx_ij + vxx_kj) / 3.0;
    numtyp vxy = (vxy_ij + vxy_kj) / 3.0;
    numtyp vxz = (vxz_ij + vxz_kj) / 3.0;
    numtyp vyy = (vyy_ij + vyy_kj) / 3.0;
    numtyp vyz = (vyz_ij + vyz_kj) / 3.0;
    numtyp vzz = (vzz_ij + vzz_kj) / 3.0;

    atomicAdd(&d_viral[angle_atomi_index*6+0], vxx);
    atomicAdd(&d_viral[angle_atomi_index*6+1], vxy);
    atomicAdd(&d_viral[angle_atomi_index*6+2], vxz);
    atomicAdd(&d_viral[angle_atomi_index*6+3], vyy);
    atomicAdd(&d_viral[angle_atomi_index*6+4], vyz);
    atomicAdd(&d_viral[angle_atomi_index*6+5], vzz);

    atomicAdd(&d_viral[angle_atomj_index*6+0], vxx);
    atomicAdd(&d_viral[angle_atomj_index*6+1], vxy);
    atomicAdd(&d_viral[angle_atomj_index*6+2], vxz);
    atomicAdd(&d_viral[angle_atomj_index*6+3], vyy);
    atomicAdd(&d_viral[angle_atomj_index*6+4], vyz);
    atomicAdd(&d_viral[angle_atomj_index*6+5], vzz);

    atomicAdd(&d_viral[angle_atomk_index*6+0], vxx);
    atomicAdd(&d_viral[angle_atomk_index*6+1], vxy);
    atomicAdd(&d_viral[angle_atomk_index*6+2], vxz);
    atomicAdd(&d_viral[angle_atomk_index*6+3], vyy);
    atomicAdd(&d_viral[angle_atomk_index*6+4], vyz);
    atomicAdd(&d_viral[angle_atomk_index*6+5], vzz);

    /* --------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

__global__ void kernel_angle_harmonic
(
    const int N,                  
    const int *d_id,              const int *d_anglelist, 
    const numtyp *d_pos,          numtyp *d_force,              
    const numtyp lx,              const numtyp ly,               const numtyp lz,
    const numtyp hlx,             const numtyp hly,              const numtyp hlz
)
{
    /* ------------------------------------------------------- */

    const int i =  blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    /* ------------------------------------------------------- */

    const int angle_type     = d_anglelist[i*4+0];
    const int angle_atomi_id = d_anglelist[i*4+1];
    const int angle_atomj_id = d_anglelist[i*4+2];
    const int angle_atomk_id = d_anglelist[i*4+3];

    /* ------------------------------------------------------- */

    const int angle_type_index = angle_type - 1; 
    const numtyp spring_k      = gpu_angle_coeff[angle_type_index*2+0]; 
    const numtyp theta_deg     = gpu_angle_coeff[angle_type_index*2+1]; 
    const numtyp theta0        = theta_deg * (M_PI / 180.0);

    /* ------------------------------------------------------- */

    const int angle_atomi_index = angle_atomi_id - 1;
    const numtyp pos_x_i        = d_pos[angle_atomi_index*3+0];
    const numtyp pos_y_i        = d_pos[angle_atomi_index*3+1];
    const numtyp pos_z_i        = d_pos[angle_atomi_index*3+2];

    const int angle_atomj_index = angle_atomj_id - 1;
    const numtyp pos_x_j        = d_pos[angle_atomj_index*3+0];
    const numtyp pos_y_j        = d_pos[angle_atomj_index*3+1];
    const numtyp pos_z_j        = d_pos[angle_atomj_index*3+2];

    const int angle_atomk_index = angle_atomk_id - 1;
    const numtyp pos_x_k        = d_pos[angle_atomk_index*3+0];
    const numtyp pos_y_k        = d_pos[angle_atomk_index*3+1];
    const numtyp pos_z_k        = d_pos[angle_atomk_index*3+2];

    /* ------------------------------------------------------- */

    numtyp dxij = pos_x_i - pos_x_j;
    numtyp dyij = pos_y_i - pos_y_j;
    numtyp dzij = pos_z_i - pos_z_j;    

    dxij = dxij - lx * ((dxij >= hlx) - (dxij < -hlx));
    dyij = dyij - ly * ((dyij >= hly) - (dyij < -hly));
    dzij = dzij - lz * ((dzij >= hlz) - (dzij < -hlz));

    numtyp dxkj = pos_x_k - pos_x_j;
    numtyp dykj = pos_y_k - pos_y_j;
    numtyp dzkj = pos_z_k - pos_z_j;    

    dxkj = dxkj - lx * ((dxkj >= hlx) - (dxkj < -hlx));
    dykj = dykj - ly * ((dykj >= hly) - (dykj < -hly));
    dzkj = dzkj - lz * ((dzkj >= hlz) - (dzkj < -hlz));
    
    numtyp rsq1 = dxij * dxij + dyij * dyij + dzij * dzij;
    numtyp rsq2 = dxkj * dxkj + dykj * dykj + dzkj * dzkj;

    numtyp r1 = sqrt(rsq1);
    numtyp r2 = sqrt(rsq2);

    /* --------------------------------------------------- */

    numtyp dot    = dxij * dxkj + dyij * dykj + dzij * dzkj;
    numtyp ctheta = dot / (r1 * r2);

    if (ctheta > (numtyp)1.0)  ctheta = (numtyp)1.0;
    if (ctheta < (numtyp)-1.0) ctheta = (numtyp)-1.0;

    numtyp sintheta = sqrt(1.0 - ctheta*ctheta);
    numtyp SMALL    = 0.001;
    if (sintheta < SMALL) sintheta = SMALL;

    numtyp theta  = acos(ctheta);
    numtyp dtheta = theta - theta0;

    /* --------------------------------------------------- */

    numtyp a   = -2.0 * spring_k * dtheta / sintheta;
    numtyp a11 = a * ctheta / rsq1;
    numtyp a12 = -a / (r1 * r2);
    numtyp a22 = a * ctheta / rsq2;
    
    numtyp fijx = a11 * dxij + a12 * dxkj;
    numtyp fijy = a11 * dyij + a12 * dykj;
    numtyp fijz = a11 * dzij + a12 * dzkj;

    numtyp fkjx = a22 * dxkj + a12 * dxij;
    numtyp fkjy = a22 * dykj + a12 * dyij;
    numtyp fkjz = a22 * dzkj + a12 * dzij;

    atomicAdd(&d_force[angle_atomi_index*3+0], fijx);
    atomicAdd(&d_force[angle_atomi_index*3+1], fijy);
    atomicAdd(&d_force[angle_atomi_index*3+2], fijz);

    atomicAdd(&d_force[angle_atomk_index*3+0], fkjx);
    atomicAdd(&d_force[angle_atomk_index*3+1], fkjy);
    atomicAdd(&d_force[angle_atomk_index*3+2], fkjz);

    numtyp fjx = - (fijx + fkjx);
    numtyp fjy = - (fijy + fkjy);
    numtyp fjz = - (fijz + fkjz);

    atomicAdd(&d_force[angle_atomj_index*3+0], fjx);
    atomicAdd(&d_force[angle_atomj_index*3+1], fjy);
    atomicAdd(&d_force[angle_atomj_index*3+2], fjz);

    /* --------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

void Angle_harmonic::compute_force(System& system, unsigned int step)
{
    /* ------------------------------------------------------- */

    Atoms& atoms          = system.atoms;
    Box& box              = system.box;
    Run_p& run_p          = system.run_p;

    /* ------------------------------------------------------- */
    
    bool log = false;

    if (step % run_p.log_f == 0) {
        log = true;
    }

    /* ------------------------------------------------------- */

    int N           = system.n_angles;
    int blockSize   = 128;
    int numBlocks   = (N + blockSize - 1) / blockSize;

    if (log)
    {
        CUDA_CHECK(cudaMemset(atoms.d_angle_pe, 0, sizeof(numtyp)*N));
        kernel_angle_harmonic_log<<<numBlocks, blockSize>>>
        (
            N,                                  
            atoms.d_id,                   atoms.d_anglelist, 
            atoms.d_pos,                  atoms.d_force,                atoms.d_angle_pe,    atoms.d_viral,
            box.lx,                       box.ly,                       box.lz,
            box.hlx,                      box.hly,                      box.hlz
        ); 
    }
    else 
    {
        kernel_angle_harmonic<<<numBlocks, blockSize>>>
        (
            N,                                  
            atoms.d_id,                   atoms.d_anglelist, 
            atoms.d_pos,                  atoms.d_force,                
            box.lx,                       box.ly,                       box.lz,
            box.hlx,                      box.hly,                      box.hlz
        );            
    }

    /* ------------------------------------------------------- */
}

/* ----------------------------------------------------------------------------------------------------------- */

///////////////////////
REGISTER_ANGLE(Angle_harmonic)
///////////////////////