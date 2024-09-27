use crate::domain::Section;
use crate::domain::mesh::Mesh;
use crate::solver::Matrix;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use std::sync::Arc;

pub trait TimeStepper<P: TimeDependentProblem> {
    fn step(&mut self, problem: &P, time: P::Time, dt: P::Time, state: &mut P::State) -> Result<(), TimeSteppingError>;
}

pub trait TimeDependentProblem {
    type State;
    type Time;

    // Function to compute the right-hand side (RHS) of the ODE: du/dt = f(t, u)
    fn compute_rhs(&self, time: Self::Time, state: &Self::State, rhs: &mut Self::State) -> Result<(), ProblemError>;

    // Optional: If needed for implicit methods to form a linear system
    fn compute_jacobian(&self, time: Self::Time, state: &Self::State) -> Option<dyn Matrix<Scalar = f64>>;

    // Set initial conditions
    fn set_initial_conditions(&self, state: &mut Self::State);

    // Compute boundary conditions
    fn apply_boundary_conditions(&self, time: Self::Time, state: &mut Self::State);

    // 
    fn initial_condition(&self, position: &[f64]) -> Self::State;

    fn boundary_condition(
        &self,
        time: Self::Time,
        position: &[f64],
    ) -> Option<Self::State>;

    fn source_term(
        &self,
        time: Self::Time,
        position: &[f64],
    ) -> Self::State;

    fn coefficient(&self, position: &[f64]) -> f64;



}

pub enum TimeSteppingError {
    SolverError(String),
    ConvergenceFailure,
    StepSizeUnderflow,
    // Additional errors can be added here
}

pub enum ProblemError {
    EvaluationError(String),
    SingularMassMatrix,
    MissingJacobian,
    // Additional errors can be added here
}

// Type alias for functions associated with coefficients, boundary conditions, etc.
pub type CoefficientFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
pub type BoundaryConditionFn = Box<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;
pub type InitialConditionFn<State> = Box<dyn Fn(&[f64]) -> State + Send + Sync>;
pub type SourceTermFn<State, Time> = Box<dyn Fn(Time, &[f64]) -> State + Send + Sync>;

// Structure for a PDE problem with mesh and associated functions
pub struct PDEProblem<State, Time> {
    pub mesh: Mesh,
    pub coefficient_section: Section<CoefficientFn>,
    pub boundary_condition_section: Section<BoundaryConditionFn>,
    pub initial_condition_fn: InitialConditionFn<State>,
    pub boundary_condition_fn: BoundaryConditionFn,
    pub source_term_fn: SourceTermFn<State, Time>,
    pub coefficient_fn: CoefficientFn,
}

impl<State, Time> PDEProblem<State, Time> {
    // Additional helper methods can be added here if necessary
}

impl<State, Time> TimeDependentProblem for PDEProblem<State, Time> {
    type State = State;
    type Time = Time;

    // Implementing the compute_rhs method
    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        rhs: &mut Self::State,
    ) -> Result<(), ProblemError> {
        // Iterate over cells in the mesh
        for cell in self.mesh.get_cells() {
            // Get the centroid of the current cell
            let centroid = self.mesh.get_cell_centroid(&cell);

            // Retrieve the coefficient function for this cell from the section
            let coeff_fn = if let Some(coeff_fn) = self.coefficient_section.restrict(&cell) {
                coeff_fn
            } else {
                &self.coefficient_fn // Fallback to the default coefficient function
            };

            let coeff = coeff_fn(&centroid);

            // Perform computations using the coefficient (modify `rhs` accordingly)
            // TODO: Actual update logic for `rhs`
        }

        Ok(())
    }

    // Optional Jacobian computation for implicit solvers
    fn compute_jacobian(&self, _time: Self::Time, _state: &Self::State) -> Option<SparseMatrix> {
        // Placeholder for now, can return None or implement as needed
        None
    }

    // Set the initial conditions for the problem
    fn set_initial_conditions(&self, state: &mut Self::State) {
        for cell in self.mesh.get_cells() {
            let position = self.mesh.get_cell_centroid(&cell);
            *state = self.initial_condition(&position);
        }
    }

    // Apply boundary conditions to the current state
    fn apply_boundary_conditions(&self, time: Self::Time, state: &mut Self::State) {
        for boundary in self.mesh.get_boundary_entities() {
            let position = self.mesh.get_entity_position(&boundary);
            if let Some(bc_value) = self.boundary_condition(time, &position) {
                *state = bc_value;
            }
        }
    }

    // Helper functions for setting initial and boundary conditions
    fn initial_condition(&self, position: &[f64]) -> Self::State {
        (self.initial_condition_fn)(position)
    }

    fn boundary_condition(
        &self,
        time: Self::Time,
        position: &[f64],
    ) -> Option<Self::State> {
        Some((self.boundary_condition_fn)(time, position))
    }

    fn source_term(
        &self,
        time: Self::Time,
        position: &[f64],
    ) -> Self::State {
        (self.source_term_fn)(time, position)
    }

    fn coefficient(&self, position: &[f64]) -> f64 {
        (self.coefficient_fn)(position)
    }
}

