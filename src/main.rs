// Import the necessary modules and types

fn main() {
    // Front matters

    // Print the mascot!
    print_mascot()

    // Time-stepping loop

    // Write final outputs

    // End matters

    // end of program
}

fn print_mascot() {
    let hydra_mascot = r#"
    *     * *     * ******  ******     *    
    *     *  *   *  *     * *     *   * *   
    *     *   * *   *     * *     *  *   *  
    *******    *    *     * ******  *     * 
    *     *    *    *     * *   *   ******* 
    *     *    *    *     * *    *  *     * 
    *     *    *    ******  *     * *     * 
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
