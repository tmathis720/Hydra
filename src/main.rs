// Import the necessary modules and types

fn main() {
    // Load the mesh from a Gmsh file
    print_mascot()

    // Further logic for the simulation goes here...
}

fn print_mascot() {
    let hydra_mascot = r#"
              ___
           __/{o \\
        __/  _/    \    ___
     __/  _/     \  \__/o  \
    /  \_/ \__/o  \  _     )\_
    \  o \  _    __/o \___/   \    __
      \__/ o\___/       \__    \__/o \
           \__           _\___/  o   )
             /  __/\     /    \_______/
           _/__/o   \___/
         /o      \_/
         \__/

        HYDRA - v0.2.0

  HYDRA is a Rust-based project 
    designed to provide a flexible, 
        modular framework for solving 
            partial differential equations (PDEs) 
                using finite volume methods (FVM).

    * Software is still in development! *
  Visit the repository at: 
        https://github.com/tmathis720/HYDRA
    "#;

    println!("{}", hydra_mascot);
}
