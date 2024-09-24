use crate::MeshEntity;

pub trait Solver {
    /// Initialize the solver with the mesh and other required data structures.
    fn initialize(&mut self);

    /// Assemble the system matrix and RHS vector using mesh entities and topological relationships.
    fn assemble_system(&mut self);

    /// Solve the system and return the solution vector.
    fn solve(&mut self) -> Vec<f64>;

    /// Return the current solution associated with a specific mesh entity.
    fn get_solution_for_entity(&self, entity: &MeshEntity) -> f64;
}
