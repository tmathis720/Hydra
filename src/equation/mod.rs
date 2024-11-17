use fields::{Fields, Fluxes};

use crate::{
    boundary::bc_handler::BoundaryConditionHandler,
    time_stepping::{TimeDependentProblem, TimeSteppingError},
    Matrix, Mesh, Vector,
};

pub mod equation;
pub mod reconstruction;
pub mod gradient;
pub mod flux_limiter;

pub mod fields;
pub mod manager;
pub mod energy_equation;
/* pub mod turbulence_models; */
pub mod momentum_equation;

pub trait PhysicalEquation<T> {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields<T>,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}

impl<T> TimeDependentProblem for Box<dyn PhysicalEquation<T>> {
    type State = Vec<f64>; // Replace with the actual state type.
    type Time = f64;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Implement based on PhysicalEquation requirements.
        unimplemented!()
    }

    fn initial_state(&self) -> Self::State {
        vec![0.0; 10] // Replace with actual initial state logic.
    }

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
        time
    }

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
        None // Replace with matrix logic if needed.
    }

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        Ok(()) // Replace with solver logic.
    }
}
