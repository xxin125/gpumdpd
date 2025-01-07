#include "main/read_in.cuh"

/* ----------------------------------------------------------------------------------------------------------- */

void Read_in::read_txt(const std::string filepath, std::string& input)
{
    std::ifstream file(filepath);
    std::string line;

    if (file.is_open()) 
    {
        while (std::getline(file, line)) 
        {
            size_t commentPos = line.find('#');
            if (commentPos != std::string::npos) 
            {
                line = line.substr(0, commentPos);
            }
            if (is_blank_line(line)) continue;
            input += line + "\n";
        }
        file.close();
    } 
    else 
    {
        std::string error = "cannot open input file: run.in";
        print_error({error});
    }
}

/* ----------------------------------------------------------------------------------------------------------- */

bool Read_in::is_blank_line(const std::string line) 
{
    return std::all_of(line.begin(), line.end(), [](unsigned char c) { return std::isspace(c); });
}

/* ----------------------------------------------------------------------------------------------------------- */

void Read_in::read(System& system)
{
    std::stringstream ss(system.input);
    std::string line;
    std::vector<std::string> args;
    std::string arg;

    while (std::getline(ss, line)) 
    {
        std::istringstream iss(line);
        args.clear();
        std::string key;
        iss >> key;

        while (iss >> arg) {
            args.push_back(arg);
        }

        /* --------------------------------------------------------------------- */

        if (key == "atom_style") 
        {
            got_atom_style = true;

            std::string error    = "illegal atom_style command";
            std::string format   = "atom_style style_name";
            std::string example0 = "atom_style atomic";
            std::string example1 = "atom_style bond";
            std::string example2 = "atom_style angle";

            if (args.size() != 1) {
                print_error_and_exit(line, error, format, {example0, example1, example2});
            }

            if (args[0] == "atomic") {
                system.atom_style = 0;
            } else if (args[0] == "bond") {
                system.atom_style = 1;
            } else if (args[0] == "angle") {
                system.atom_style = 2;
            } else {
                print_error_and_exit(line, error, format, {example0, example1, example2});
            }

            std::cout <<  "   atom_style      " << args[0] << std::endl;
        }

        /* --------------------------------------------------------------------- */
        
        Run_p& run_p = system.run_p; 

        /* --------------------------------------------------------------------- */

        if (key == "neighbor") 
        {
            got_neighbor = true;

            std::string error    = "illegal neighbor command";
            std::string format   = "neighbor global_cut skin max_rho check_frequency";
            std::string example0 = "neighbor 1.00 0.15 20.0 1";
            std::string example1 = "neighbor 1.00 0.00 20.0 1";

            if (args.size() != 4) {
                print_error_and_exit(line, error, format, {example0, example1});
            }

            run_p.global_cut = parse_float<numtyp>(args[0], line, "global_cut",      {format, example0, example1});
            run_p.skin       = parse_float<numtyp>(args[1], line, "skin",            {format, example0, example1});
            run_p.max_rho    = parse_float<numtyp>(args[2], line, "max_rho",         {format, example0, example1});
            run_p.nl_f       = parse_int<int>(     args[3], line, "check_frequency", {format, example0, example1});

            std::cout <<  "   neighbor        " << run_p.global_cut << " " << 
                                                   run_p.skin       << " " <<                 
                                                   run_p.max_rho    << " " <<
                                                   run_p.nl_f       << std::endl;
        }

        /* --------------------------------------------------------------------- */
                
        if (key == "timestep") 
        {
            got_timestep = true;

            std::string error    = "illegal timestep command";
            std::string format   = "timestep timestep";
            std::string example0 = "timestep 0.01";

            if (args.size() != 1) {
                print_error_and_exit(line, error, format, {example0});
            }
            run_p.dt = parse_float<numtyp>(args[0], line, "timestep", {format,example0});

            std::cout <<  "   timestep        " << run_p.dt << std::endl;
        }

        /* --------------------------------------------------------------------- */
                        
        if (key == "thermo") 
        {
            got_thermo = true;

            std::string error    = "illegal thermo command";
            std::string format   = "thermo frequency";
            std::string example0 = "thermo 10000";

            if (args.size() != 1) {
                print_error_and_exit(line, error, format, {example0});
            }
            run_p.log_f = parse_int<unsigned int>(args[0], line, "thermo", {format,example0});

            std::cout <<  "   thermo          " << run_p.log_f << std::endl;
        }

        /* --------------------------------------------------------------------- */
                                
        if (key == "run") 
        {
            got_run = true;

            std::string error    = "illegal run command";
            std::string format   = "run steps";
            std::string example0 = "run 1000000";

            if (args.size() != 1) {
                print_error_and_exit(line, error, format, {example0});
            }
            run_p.nsteps = parse_int<unsigned int>(args[0], line, "run", {format,example0});

            std::cout <<  "   run             " << run_p.nsteps << std::endl;
        }

        /* --------------------------------------------------------------------- */

        if (key == "read_data") 
        {
            got_read_data = true;

            std::string error    = "illegal read_data command";
            std::string format   = "read_data filename";
            std::string example0 = "read_data in.data";
            std::string example1 = "read_data in.data";

            if (args.size() != 1) {
                print_error_and_exit(line, error, format, {example0, example1});
            }

            run_p.input_data_name = args[0];

            std::cout <<  "   read_data       " << run_p.input_data_name << std::endl;   
        }

        /* --------------------------------------------------------------------- */

        if (key == "write_data") 
        {
            run_p.write_data_flag = true;

            std::string error    = "illegal write_data command";
            std::string format   = "write_data filename";
            std::string example0 = "write_data out.data";

            if (args.size() != 1) {
                print_error_and_exit(line, error, format, {example0});
            }

            run_p.out_data_name = args[0];
            std::cout <<  "   write_data      " << run_p.out_data_name << std::endl;
        }

        /* --------------------------------------------------------------------- */

        if (key == "dump") 
        {
            run_p.write_dump_flag = true;

            std::string error    = "illegal dump command";
            std::string format   = "dump start_step frequency wrapped/unwrapped force_yes/force_no vel_yes/vel_no";
            std::string example0 = "dump 0 10000 wrapped force_no vel_no";
            std::string example1 = "dump 1000 10000 unwrapped force_no vel_yes";

            if (args.size() != 5) {
                print_error_and_exit(line, error, format, {example0, example1});
            }

            run_p.dump_start = parse_int<unsigned int>(args[0], line, "start_step", {format,example0, example1});
            run_p.dump_f     = parse_int<unsigned int>(args[1], line, "frequency",  {format,example0, example1});         

            if (args[2] == "wrapped") {
                run_p.wrapped_flag = true;
            } else if (args[2] == "unwrapped") {
                run_p.wrapped_flag = false;
            } else {
                print_error_and_exit(line, error, format, {example0, example1});
            }

            if (args[3] == "force_yes") {
                run_p.dumpforce_flag = true;
            } else if (args[3] == "force_no") {
                run_p.dumpforce_flag = false;
            } else {
                print_error_and_exit(line, error, format, {example0, example1});
            }

            if (args[4] == "vel_yes") {
                run_p.dumpvel_flag = true;
            } else if (args[4] == "vel_no") {
                run_p.dumpvel_flag = false;
            } else {            
                print_error_and_exit(line, error, format, {example0, example1});
            }

            std::cout <<  "   dump            " << run_p.dump_start << " " << run_p.dump_f << " " << args[2] << " " << args[3] << " " << args[4] << std::endl;
        }

        /* --------------------------------------------------------------------- */

        if (key == "pair_style") {
            got_pair_style = true;
        }
                    
        /* --------------------------------------------------------------------- */

        if (key == "bond_style") {
            got_bond_style = true;
        }

        /* --------------------------------------------------------------------- */

        if (key == "angle_style") {
            got_angle_style = true;
        }

        /* --------------------------------------------------------------------- */

        std::unordered_set<std::string> allowed_keys = 
        {
            "atom_style", "neighbor",   "timestep",   "thermo",     "run", 
            "read_data",  "write_data", "dump", 
            "pair_style", "pair_coeff", "bond_style", "bond_coeff", "angle_style", "angle_coeff", 
            "group",      "fix",        "compute"
        };

        if (allowed_keys.find(key) == allowed_keys.end()) {
            print_error({"unknown command " + key});
        }

        /* --------------------------------------------------------------------- */
    }
    
    check(system);
}

/* ----------------------------------------------------------------------------------------------------------- */

void Read_in::check(const System& system) 
{
    if (!got_atom_style) 
    {
        std::string error    = "atom_style command is missing";
        std::string format   = "atom_style style_name";
        std::string example0 = "atom_style atomic";
        std::string example1 = "atom_style bond";
        std::string example2 = "atom_style angle";
        print_error_and_exit("run.in", error, format, {example0, example1, example2});  
    } 
    
    else if (!got_neighbor) 
    {
        std::string error    = "illegal neighbor command";
        std::string format   = "neighbor global_cut skin max_rho check_frequency";
        std::string example0 = "neighbor 1.00 0.15 20.0 1";
        std::string example1 = "neighbor 1.00 0.00 20.0 1";
        print_error_and_exit("run.in", error, format, {example0, example1});   
    }   

    else if (!got_timestep) 
    {
        std::string error    = "timestep command is missing";
        std::string format   = "timestep timestep";
        std::string example0 = "timestep 0.01";
        print_error_and_exit("run.in", error, format, {example0});   
    }   

    else if (!got_thermo) 
    {
        std::string error    = "thermo command is missing";
        std::string format   = "thermo frequency";
        std::string example0 = "thermo 10000";
        print_error_and_exit("run.in", error, format, {example0});   
    }  

    else if (!got_run) 
    {
        std::string error    = "run command is missing";
        std::string format   = "run steps";
        std::string example0 = "run 1000000";
        print_error_and_exit("run.in", error, format, {example0});   
    }  

    else if (!got_read_data) 
    {
        std::string error    = "read_data command is missing";
        std::string format   = "read_data filename";
        std::string example0 = "read_data in.data";
        print_error_and_exit("run.in", error, format, {example0});   
    } 

    else if (!got_pair_style) 
    {
        std::string error    = "pair_style command is missing";
        print_error({error});   
    } 

    else if (system.atom_style == 1 && !got_bond_style)
    {
        std::string error    = "bond_style command is missing";
        print_error({error}); 
    }   
    
    else if (system.atom_style == 2 && (!got_bond_style || !got_angle_style))
    {
        std::string error;
        if (!got_bond_style && !got_angle_style) {
            error = "bond_style and angle_style commands are missing";
        } else if (!got_bond_style) {
            error = "bond_style command is missing";
        } else {
            error = "angle_style command is missing";
        }
        print_error({error});
    }  
}

/* ----------------------------------------------------------------------------------------------------------- */

