use crate::{equation::fields::UpdateState, linalg::Matrix};
use std::ops::Add;

#[derive(Debug)]
pub enum TimeSteppingError {
    InvalidStep,
    SolverError(String),
}

pub trait TimeDependentProblem {
    type State: Clone + UpdateState;
    type Time: Copy + PartialOrd + Add<Output = Self::Time> + From<f64> + Into<f64>;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>>;

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}

pub trait TimeStepper<P>
where
    P: TimeDependentProblem + Sized,
{
    fn current_time(&self) -> P::Time;

    fn set_current_time(&mut self, time: P::Time);

    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
        tol: f64,
    ) -> Result<P::Time, TimeSteppingError>;

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);

    fn set_time_step(&mut self, dt: P::Time);

    fn get_time_step(&self) -> P::Time;
}

pub struct FixedTimeStepper<P>
where
    P: TimeDependentProblem,
{
    current_time: P::Time,
    time_step: P::Time,
    start_time: P::Time,
    end_time: P::Time,
}

impl<P> FixedTimeStepper<P>
where
    P: TimeDependentProblem,
{
    pub fn new(start_time: P::Time, end_time: P::Time, time_step: P::Time) -> Self {
        FixedTimeStepper {
            current_time: start_time,
            time_step,
            start_time,
            end_time,
        }
    }
}

impl<P> TimeStepper<P> for FixedTimeStepper<P>
where
    P: TimeDependentProblem,
{
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut derivative = state.clone();
        problem.compute_rhs(current_time, state, &mut derivative)?;

        state.update_state(&derivative, dt.into());

        self.current_time = self.current_time + dt;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
        _tol: f64,
    ) -> Result<P::Time, TimeSteppingError> {
        Err(TimeSteppingError::InvalidStep)
    }

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time) {
        self.start_time = start_time;
        self.end_time = end_time;
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }

    fn get_time_step(&self) -> P::Time {
        self.time_step
    }
}
