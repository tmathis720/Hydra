#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::boundary::{Inflow, Outflow, FreeSurfaceBoundary, NoSlipBoundary};
    use crate::solver::{FluxSolver, SemiImplicitSolver};
    use crate::timestep::ExplicitEuler;
    use nalgebra::Vector3;

    #[test]
    fn test_inflow_outflow_boundary_conditions() {
        let mut time_stepper = ExplicitEuler { solver: FluxSolver };  // Initialize TimeStepper

        let mut left_element = Element {
            id: 0,
            mass: 1.0,
            pressure: 10.0,
            velocity: Vector3::new(1.0, 0.0, 0.0),  // Inflow velocity
            momentum: Vector3::new(10.0, 0.0, 0.0),
            ..Default::default()
        };
        let mut right_element = Element {
            id: 1,
            mass: 1.0,
            pressure: 5.0,
            velocity: Vector3::new(-1.0, 0.0, 0.0),  // Outflow velocity
            momentum: Vector3::new(-5.0, 0.0, 0.0),
            ..Default::default()
        };

        let mut face = Face::new(0, vec![0, 1], Vector3::new(0.0, 0.0, 0.0), 1.0, ..Face::default());

        let inflow = Inflow { rate: 0.1 };
        let outflow = Outflow { rate: 0.1 };

        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        let total_time = 10.0;
        let dt = 0.01;
        for _ in 0..((total_time / dt) as usize) {

            inflow.apply_boundary(&mut left_element, dt);
            outflow.apply_boundary(&mut right_element, dt);

            // Compute the 3D flux
            let flux_3d = flux_solver.compute_flux_3d(&face, &left_element, &right_element);

            // Apply the flux to update the face velocity
            flux_solver.apply_flux_3d(&mut face, flux_3d, dt);

            // Update momentum using semi-implicit solver
            left_element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux_3d.x * (left_element.momentum.x / left_element.mass),
                left_element.momentum,
                dt,
            );
            right_element.momentum = semi_implicit_solver.semi_implicit_update(
                flux_3d.x * (right_element.momentum.x / right_element.mass),
                right_element.momentum,
                dt,
            );

            assert!(left_element.mass > 1.0, "Mass should increase with inflow");
            assert!(right_element.mass < 1.0, "Mass should decrease with outflow");
        }
    }

    #[test]
    fn test_free_surface_boundary_conditions() {
        let mut time_stepper = ExplicitEuler { solver: FluxSolver };

        let mut element = Element {
            id: 0,
            mass: 1.0,
            pressure: 10.0,
            velocity: Vector3::new(0.0, 0.0, 1.0),  // Upward velocity at free surface
            momentum: Vector3::new(0.0, 0.0, 10.0),
            ..Default::default()
        };

        let free_surface = FreeSurfaceBoundary { pressure_at_surface: 1.0 };
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        let total_time = 10.0;
        let dt = 0.01;
        for _ in 0..((total_time / dt) as usize) {

            free_surface.apply(&mut mesh);

            // Compute flux relative to the free surface
            let flux_3d = flux_solver.compute_flux_3d(&element, free_surface.pressure_at_surface);

            // Update the momentum based on the flux
            element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux_3d.z * (element.momentum.z / element.mass),
                element.momentum,
                dt,
            );

            // Assert the pressure is adjusting toward the free surface pressure
            assert!(element.pressure > free_surface.pressure_at_surface, "Pressure should decrease towards the surface");
        }
    }

    #[test]
    fn test_no_slip_boundary_conditions() {
        let mut time_stepper = ExplicitEuler{ solver: FluxSolver };
        let mut dt: f64 = 0.01;
        let mut element = Element {
            id: 0,
            velocity: Vector3::new(2.0, 0.0, 0.0),  // Initial velocity
            momentum: Vector3::new(10.0, 0.0, 0.0),
            ..Default::default()
        };

        let no_slip_boundary = NoSlipBoundary {};
        let flux_solver = FluxSolver {};

        let total_time = 10.0;
        let dt = 0.01;
        for _ in 0..((total_time / dt) as usize) {

            no_slip_boundary.apply_boundary(&mut element, dt);

            // No flux should be applied at no-slip boundaries
            let flux_3d = flux_solver.compute_flux_3d(&element);

            // Apply no-slip conditions (should result in zero flux)
            assert_eq!(flux_3d, Vector3::new(0.0, 0.0, 0.0), "Flux should be zero at no-slip boundary");
            assert_eq!(element.velocity, Vector3::new(0.0, 0.0, 0.0), "Velocity should be zero at no-slip boundary");
        }
    }

    #[test]
    fn test_open_boundary_conditions() {
        let mut time_stepper = ExplicitEuler{ solver: FluxSolver };
        let dt = 0.01;
        let mut inflow_element = Element {
            id: 0,
            mass: 2.0,
            pressure: 10.0,
            velocity: Vector3::new(1.0, 0.0, 0.0),  // Inflow velocity
            momentum: Vector3::new(10.0, 0.0, 0.0),
            ..Default::default()
        };

        let mut outflow_element = Element {
            id: 1,
            mass: 1.0,
            pressure: 5.0,
            velocity: Vector3::new(-1.0, 0.0, 0.0),  // Outflow velocity
            momentum: Vector3::new(-5.0, 0.0, 0.0),
            ..Default::default()
        };

        let face = Face {
            id: 0,
            nodes: vec![0, 1],  // Nodes shared between left and right elements
            velocity: Vector3::new(1.0, 0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
            ..Face::default()
        };
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};
        let dt = 0.01;

        let total_time = 10.0;
        let dt = 0.01;
        for _ in 0..((total_time / dt) as usize) {
            

            // Compute 3D flux between inflow and outflow elements
            let flux_3d = flux_solver.compute_flux_3d(&face, &inflow_element, &outflow_element);

            // Apply the flux to update face velocity and element momentum
            flux_solver.apply_flux_3d(&mut face, flux_3d, dt);
            inflow_element.momentum += flux_3d * dt;
            outflow_element.momentum -= flux_3d * dt;

            assert!(inflow_element.momentum.x > 0.0, "Momentum should increase at inflow");
            assert!(outflow_element.momentum.x < 0.0, "Momentum should decrease at outflow");
        }
    }

    #[test]
    fn test_periodic_boundary_conditions() {
        let mut time_stepper = ExplicitEuler { solver: FluxSolver };

        let mut elements = vec![
            Element { id: 0, pressure: 10.0, velocity: Vector3::new(2.0, 0.0, 0.0), ..Default::default() },
            Element { id: 1, pressure: 5.0, velocity: Vector3::new(-2.0, 0.0, 0.0), ..Default::default() }
        ];

        let face = Face {
            id: 0,
            nodes: vec![0, 1],  // Nodes shared between left and right elements
            velocity: Vector3::new(0.0, 0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
            ..Face::default()
        };
        let flux_solver = FluxSolver {};
        let dt = 0.01;

        let total_time = 10.0;
        let dt = 0.01;
        for _ in 0..((total_time / dt) as usize) {

            // Compute 3D flux for periodic boundary conditions
            let flux_3d = flux_solver.compute_flux_3d(&face, &elements[0], &elements[1]);

            // Update momentum with periodic flux
            elements[0].momentum += flux_3d * dt;
            elements[1].momentum -= flux_3d * dt;

            assert!(elements[0].momentum.x > 0.0, "Momentum should increase for one side of the periodic boundary");
            assert!(elements[1].momentum.x < 0.0, "Momentum should decrease for the other side");
        }
    }
}
