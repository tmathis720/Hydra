pub mod flux_solver;
pub mod scalar_transport;
pub mod eddy_viscosity_solver;
pub mod crank_nicolson_solver;
pub mod flux_limiter;
pub mod semi_implicit_solver;
//pub mod algebraic_multigrid;
//pub mod pressure_solver;

// Export solvers so they can be used in other parts of the codebase
pub use flux_solver::FluxSolver;
pub use scalar_transport::ScalarTransportSolver;
pub use eddy_viscosity_solver::EddyViscositySolver;
pub use crank_nicolson_solver::CrankNicolsonSolver;
pub use flux_limiter::FluxLimiter;
pub use semi_implicit_solver::SemiImplicitSolver;
