#include "read_data.cuh"

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::read_data(System& system)
{
    check_atom_style(system);
    read_system_data(system);   
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::check_atom_style(System& system)
{
    /* ------------------------------------------------------- */

    // open data file

    input_data = system.run_p.input_data_name;

    std::ifstream inputFile(input_data);
    std::string line;

    if (!inputFile) {
        print_error({"Cannot open data file: " + input_data});
    }

    /* ------------------------------------------------------- */

    // check atom style

    while (std::getline(inputFile, line)) 
    {
        if (line.find("Atoms") != std::string::npos) 
        {
            std::istringstream iss(line);
            std::string extracted1, extracted2, extracted3;
            if (!(iss >> extracted1 >> extracted2 >> extracted3)) {
                print_error({"Error reading atom style from data file."});
            }
            
            int data_atom_style; 
            if (extracted3 == "atomic") {
                data_atom_style = 0;
                std::cout << "   atom_style: atomic"  << std::endl;
            } else if (extracted3 == "bond") {
                data_atom_style = 1;
                std::cout << "   atom_style: bond"  << std::endl;
            } else if (extracted3 == "angle") {
                data_atom_style = 2;
                std::cout << "   atom_style: angle"  << std::endl;
            } else {
                print_error({"Unknown atom_style in data file."});
            }

            if (data_atom_style != system.atom_style) {
                print_error({"atom_style in run.in is inconsistent with the style in data file."});
            }
            break;
        }
    }

    inputFile.close();

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::read_box_info(System& system, std::ifstream& inputFile) 
{
    Box& box = system.box;
    std::string line;

    while (std::getline(inputFile, line)) 
    {
        if (line.find("xlo xhi") != std::string::npos) {
            std::istringstream iss(line);
            if (!(iss >> box.xlo >> box.xhi)) {
                print_error({"Error reading xlo xhi values."});
            }
        } else if (line.find("ylo yhi") != std::string::npos) {
            std::istringstream iss(line);
            if (!(iss >> box.ylo >> box.yhi)) {
                print_error({"Error reading ylo yhi values."});
            }
        } else if (line.find("zlo zhi") != std::string::npos) {
            std::istringstream iss(line);
            if (!(iss >> box.zlo >> box.zhi)) {
                print_error({"Error reading zlo zhi values."});
            }
            break; 
        }
    }

    if (box.xlo >= box.xhi || box.ylo >= box.yhi || box.zlo >= box.zhi) {
        print_error({"Invalid box dimensions."});
    }

    std::cout << "   xlo xhi: " << box.xlo << " " << box.xhi << std::endl;
    std::cout << "   ylo yhi: " << box.ylo << " " << box.yhi << std::endl;
    std::cout << "   zlo zhi: " << box.zlo << " " << box.zhi << std::endl;

    box.lx  = box.xhi - box.xlo;
    box.ly  = box.yhi - box.ylo;
    box.lz  = box.zhi - box.zlo;
    box.hlx = box.lx  * static_cast<numtyp>(0.5);
    box.hly = box.ly  * static_cast<numtyp>(0.5);
    box.hlz = box.lz  * static_cast<numtyp>(0.5);
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::read_masses(System& system, std::ifstream& inputFile) 
{
    std::string line;
    std::getline(inputFile, line);  
    std::string masssection_name;
    inputFile >> masssection_name;

    if (masssection_name != "Masses") { 
        print_error({"Masses section is missing."});
    }

    std::getline(inputFile, line); 

    for (int i = 0; i < system.n_atomtypes; ++i) 
    {
        int atomType;
        numtyp mass;
        if (!(inputFile >> atomType >> mass)) {
            print_error({"Error reading mass for atom type."});
        }
        std::cout << "   mass " << atomType << " " << mass << std::endl;
        masses.push_back(static_cast<numtyp>(mass));
    }

    if (system.n_atomtypes > masses.size()) {
        print_error({"All the masses are not set."});              
    }
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::read_header_info(System& system, std::ifstream& inputFile) 
{
    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        if (line.find("atoms") != std::string::npos) {
            if (!(iss >> system.n_atoms)) {
                print_error({"Error reading number of atoms from line: " + line});
            }
            std::cout << "   number of atoms: " << system.n_atoms << '\n';
        } else if (line.find("bonds") != std::string::npos) {
            if (!(iss >> system.n_bonds)) {
                print_error({"Error reading number of bonds from line: " + line});
            }
            std::cout << "   number of bonds: " << system.n_bonds << '\n';
        } else if (line.find("angles") != std::string::npos) {
            if (!(iss >> system.n_angles)) { 
                print_error({"Error reading number of angles from line: " + line});
            }
            std::cout << "   number of angles: " << system.n_angles << '\n';
        } else if (line.find("atom types") != std::string::npos) {
            if (!(iss >> system.n_atomtypes)) {
                print_error({"Error reading number of atom types from line: " + line});
            }
            std::cout << "   number of atom types: " << system.n_atomtypes << '\n';
        } else if (line.find("bond types") != std::string::npos) {
            if (!(iss >> system.n_bondtypes)) {
                print_error({"Error reading number of bond types from line: " + line});
            }
            std::cout << "   number of bond types: " << system.n_bondtypes << '\n';
        } else if (line.find("angle types") != std::string::npos) {
            if (!(iss >> system.n_angletypes)) {
                print_error({"Error reading number of angle types from line: " + line});
            }
            std::cout << "   number of angle types: " << system.n_angletypes << '\n';
        }

        if ((system.atom_style == 0 && line.find("atom types") != std::string::npos) ||
            (system.atom_style == 1 && line.find("bond types") != std::string::npos) ||
            (system.atom_style == 2 && line.find("angle types") != std::string::npos)) {
            break;
        }
    }
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::adjust_coordinate(numtyp& coord, numtyp lo, numtyp hi) 
{
    while (coord >= hi || coord < lo) {
        coord -= (hi - lo) * ((coord >= hi) - (coord < lo));
    }
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::read_system_data(System& system)
{
    /* ------------------------------------------------------- */

    // open data file 

    input_data = system.run_p.input_data_name;
    std::ifstream inputFile(input_data); 

    /* ------------------------------------------------------- */

    // read and print header

    read_header_info(system, inputFile);

    /* ------------------------------------------------------- */

    // read box and masses

    read_box_info(system, inputFile);
    read_masses(system, inputFile);

    /* ------------------------------------------------------- */

    // allocate memory

    atoms_mem_alloc(system);

    /* ------------------------------------------------------- */

    // read the atom section

    std::string line, atomsection_name;
    std::getline(inputFile, line);
    std::getline(inputFile, line);
    inputFile >> atomsection_name;

    if (atomsection_name != "Atoms") { 
        print_error({"Atoms section is missing."});
    }       

    std::getline(inputFile, line);

    Atoms& atoms = system.atoms;

    if (system.atom_style == 1 || system.atom_style == 2) {
        atoms.h_mol_id.resize(system.n_atoms);
    }

    for (int i = 0; i < system.n_atoms; ++i) 
    {
        std::getline(inputFile, line);
    
        if (system.atom_style == 1 || system.atom_style == 2) {  // Bond style or Angle style
            inputFile >> id >> mol >> type >> x >> y >> z;
            atoms.h_mol_id[id - 1] = mol;
        } else {                         // Atomic style
            inputFile >> id >> type >> x >> y >> z;
        }
        
        if (id == 0) {
            print_error({"Atom ID cannot be 0."});
        }

        int index = id - 1;

        atoms.h_id[index]    = id;  
        atoms.h_type[index]  = type;

        x = static_cast<numtyp>(x);
        y = static_cast<numtyp>(y);
        z = static_cast<numtyp>(z);

        adjust_coordinate(x, system.box.xlo, system.box.xhi);
        adjust_coordinate(y, system.box.ylo, system.box.yhi);
        adjust_coordinate(z, system.box.zlo, system.box.zhi);

        atoms.h_pos[index*3+0]   = x;
        atoms.h_pos[index*3+1]   = y;
        atoms.h_pos[index*3+2]   = z;
        atoms.h_uwpos[index*3+0] = x;
        atoms.h_uwpos[index*3+1] = y;
        atoms.h_uwpos[index*3+2] = z;
    }

    /* ------------------------------------------------------- */

    // check velocity section

    std::string section_name;
    check_velocities(system, inputFile, section_name);

    /* ------------------------------------------------------- */

    // copy system info from host to device

    atoms_mem_copy_2_gpu(system, masses);

    /* ------------------------------------------------------- */

    // Read and process bonds if needed

    if (system.atom_style == 1 || system.atom_style == 2) 
    {
        if (!has_ini_v)
        {
            if (section_name != "Bonds") {
                print_error({"Bonds section is missing."});
            }
            std::getline(inputFile, line);
        }
        else 
        {
            std::getline(inputFile, line); 
            std::getline(inputFile, line);
            inputFile >> section_name;
            if (section_name != "Bonds") {
                print_error({"Bonds section is missing."});
            }
            std::getline(inputFile, line);
        }

        process_bonds(system, inputFile);
    }

    /* ------------------------------------------------------- */

    // Read and process angles if needed

    if (system.atom_style == 2) 
    {    
        std::getline(inputFile, line); 
        std::getline(inputFile, line);
        inputFile >> section_name;
        if (section_name != "Angles") {
            print_error({"Angles section is missing."});
        }
        std::getline(inputFile, line);
        process_angles(system, inputFile);
    }

    /* ------------------------------------------------------- */
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::check_velocities(System& system, std::ifstream& inputFile, std::string& section_name) 
{
    std::string line;
    std::getline(inputFile, line);  
    std::getline(inputFile, line);  

    inputFile >> section_name;
    has_ini_v = false;

    Atoms& atoms = system.atoms;

    if (section_name == "Velocities") 
    {
        has_ini_v = true;
        std::cout << "   has initial Velocities" <<  '\n';
        std::getline(inputFile, line);

        for (int i = 0; i < system.n_atoms; ++i) 
        {
            std::getline(inputFile, line);

            inputFile >> id >> vx >> vy >> vz;

            int index = id - 1;

            atoms.h_vel[index*3+0] = static_cast<numtyp>(vx);
            atoms.h_vel[index*3+1] = static_cast<numtyp>(vy);
            atoms.h_vel[index*3+2] = static_cast<numtyp>(vz);
        }
    } 
    else 
    {
        std::cout << "   has no initial Velocities" <<  '\n';        
    }
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::process_bonds(System& system, std::ifstream& inputFile)
{
    /* ------------------------------------------------------- */

    // allocate memory

    Atoms& atoms = system.atoms;

    atoms.h_bondlist.resize(system.n_bonds * 3, -1);

    /* ------------------------------------------------------- */

    // read bond list

    std::string line;
        
    for (int i = 0; i < system.n_bonds; ++i) 
    {
        std::getline(inputFile, line);

        inputFile >> bondid >> bondtype >> bond_atomi_id >> bond_atomj_id;

        if (bondid == 0) {
            print_error({"Bond ID cannot be 0."});
        }

        int index = bondid - 1;

        atoms.h_bondlist[index*3+0] = bondtype;
        atoms.h_bondlist[index*3+1] = bond_atomi_id;
        atoms.h_bondlist[index*3+2] = bond_atomj_id;
    }

    /* ------------------------------------------------------- */

    bonds_mem_alloc(system);
    bonds_mem_copy_2_gpu(system);

    /* ------------------------------------------------------- */   
}

/* ---------------------------------------------------------------------------------------------------------------------------- */

void Read_data::process_angles(System& system, std::ifstream& inputFile)
{
    /* ------------------------------------------------------- */

    // allocate memory

    Atoms& atoms = system.atoms;

    atoms.h_anglelist.resize(system.n_angles * 4, -1);

    /* ------------------------------------------------------- */

    // read angle list

    std::string line;
        
    for (int i = 0; i < system.n_angles; ++i) 
    {
        std::getline(inputFile, line);

        inputFile >> angleid >> angletype >> angle_atomi_id >> angle_atomj_id >> angle_atomk_id;

        if (angleid == 0) {
            print_error({"Angle ID cannot be 0."});
        }

        int index = angleid - 1;

        atoms.h_anglelist[index*4+0] = angletype;
        atoms.h_anglelist[index*4+1] = angle_atomi_id;
        atoms.h_anglelist[index*4+2] = angle_atomj_id;
        atoms.h_anglelist[index*4+3] = angle_atomk_id;
    }

    /* ------------------------------------------------------- */

    angles_mem_alloc(system);
    angles_mem_copy_2_gpu(system);

    /* ------------------------------------------------------- */   
}

/* ---------------------------------------------------------------------------------------------------------------------------- */


void Read_data::write_data(System& system)
{
    /* ------------------------------------------------------- */

    // copy positon and velocity from device to host

    Atoms& atoms = system.atoms;
    Box& box     = system.box;

    CUDA_CHECK(cudaMemcpy(atoms.h_id.data(),   atoms.d_id,    sizeof(int)    * system.n_atoms,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(atoms.h_type.data(), atoms.d_type,  sizeof(int)    * system.n_atoms,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(atoms.h_pos.data(),  atoms.d_pos,   sizeof(numtyp) * system.n_atoms * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(atoms.h_vel.data(),  atoms.d_vel,   sizeof(numtyp) * system.n_atoms * 3, cudaMemcpyDeviceToHost));    
        
    /* ------------------------------------------------------- */

    // open data file and write header

    output_data = system.run_p.out_data_name;

    FILE* outputFile = fopen(output_data.c_str(), "w");

    if (outputFile == nullptr) {
        print_error({"Error opening file: " + output_data});
    }

    fprintf(outputFile, "# Lammps format data file\n");
    fprintf(outputFile, "                         \n");

    fprintf(outputFile, "%d atoms\n", system.n_atoms);
    if (system.atom_style == 1 || system.atom_style == 2) {
        fprintf(outputFile, "%d bonds\n", system.n_bonds);
        if (system.atom_style == 2) {
            fprintf(outputFile, "%d angles\n", system.n_angles);
        }
    }
    fprintf(outputFile, "%d atom types\n", system.n_atomtypes);
    if (system.atom_style == 1 || system.atom_style == 2) {
        fprintf(outputFile, "%d bond types\n", system.n_bondtypes);
        if (system.atom_style == 2) {
            fprintf(outputFile, "%d angle types\n", system.n_angletypes);
        }
    }

    fprintf(outputFile, "                           \n");

    /* ------------------------------------------------------- */

    // write box info

    fprintf(outputFile, "%.9f %.9f xlo xhi\n", box.xlo, box.xhi);
    fprintf(outputFile, "%.9f %.9f ylo yhi\n", box.ylo, box.yhi);
    fprintf(outputFile, "%.9f %.9f zlo zhi\n", box.zlo, box.zhi);
    fprintf(outputFile, "                           \n");

    /* ------------------------------------------------------- */

    // write mass info

    fprintf(outputFile, "Masses\n");
    fprintf(outputFile, "                           \n");
    for (int i = 0; i < system.n_atomtypes; ++i) {
        fprintf(outputFile, "%d %.9f\n", i+1, masses[i]);
    }
    fprintf(outputFile, "                           \n");

    /* ------------------------------------------------------- */

    // write atoms section

    if (system.atom_style == 0) {
        fprintf(outputFile, "Atoms  # atomic\n");
    } 
    
    if (system.atom_style == 1) {
        fprintf(outputFile, "Atoms  # bond\n");
    } 
    
    if (system.atom_style == 2) {
        fprintf(outputFile, "Atoms  # angle\n");
    }

    fprintf(outputFile, "                           \n");
    
    /* ------------------------------------------------------- */

    // write atoms info

    for (int i = 0; i < system.n_atoms; ++i) 
    {
        if (system.atom_style == 1 || system.atom_style == 2) 
        {
            int index = atoms.h_id[i] - 1;
            int mol   = atoms.h_mol_id[index];

            fprintf(outputFile, "%d %d %d %.9f %.9f %.9f\n",
                                atoms.h_id[i],
                                mol,
                                atoms.h_type[i],
                                atoms.h_pos[i*3+0],
                                atoms.h_pos[i*3+1],
                                atoms.h_pos[i*3+2]);
        } 
        else 
        {
            fprintf(outputFile, "%d %d %.9f %.9f %.9f\n",
                                atoms.h_id[i],
                                atoms.h_type[i],
                                atoms.h_pos[i*3+0],
                                atoms.h_pos[i*3+1],
                                atoms.h_pos[i*3+2]);
        }
    }
    fprintf(outputFile, "                           \n");

    // write velocity info

    fprintf(outputFile, "Velocities\n");
    fprintf(outputFile, "                           \n");
    for (int i = 0; i < system.n_atoms; ++i) 
    {
        fprintf(outputFile, "%d %.9f %.9f %.9f\n",
                            atoms.h_id[i],
                            atoms.h_vel[i*3+0],
                            atoms.h_vel[i*3+1],
                            atoms.h_vel[i*3+2]);
    }
    fprintf(outputFile, "                           \n");

    // check atom_style and write bond list

    if (system.atom_style == 1 || system.atom_style == 2) 
    {
        fprintf(outputFile, "Bonds\n");
        fprintf(outputFile, "                           \n");
        for (int i = 0; i < system.n_bonds; ++i) 
        {
            fprintf(outputFile, "%d %d %d %d\n",
                                i + 1,
                                atoms.h_bondlist[i*3+0],
                                atoms.h_bondlist[i*3+1],
                                atoms.h_bondlist[i*3+2]);
        }
    }

    // check atom_style and write angle list

    if (system.atom_style == 2) 
    {
        fprintf(outputFile, "                           \n");
        fprintf(outputFile, "Angles\n");
        fprintf(outputFile, "                           \n");
        for (int i = 0; i < system.n_angles; ++i) 
        {
            fprintf(outputFile, "%d %d %d %d %d\n",
                                i + 1,
                                atoms.h_anglelist[i*4+0],
                                atoms.h_anglelist[i*4+1],
                                atoms.h_anglelist[i*4+2],
                                atoms.h_anglelist[i*4+3]);
        }
    }

    fclose(outputFile);
}

/* ---------------------------------------------------------------------------------------------------------------------------- */