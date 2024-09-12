#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face, Mesh};
    use crate::boundary::{BoundaryElement, BoundaryType};
    use crate::solver::{CrankNicolsonSolver, EddyViscositySolver, FluxSolver};
    use crate::timestep::{TimeStepper, ExplicitEuler};
    use std::rc::Rc;
    use std::cell::RefCell;

    #[test]
    fn test_complex_integration_with_horizontal_diffusion() {
        let dt = 0.01;
        let total_time = 10.0;

        // Define multiple elements in the domain
        let elements = vec![
            Element { id: 0, element_type: 2, nodes: vec![0, 1], faces: vec![0], mass: 1.0, neighbor_ref: 0, pressure: 10.0, height: 0.0, area: 1.0, momentum: 2.0, velocity: (0.0, 0.0, 0.0) },
            Element { id: 1, element_type: 2, nodes: vec![1, 2], faces: vec![1], mass: 1.0, neighbor_ref: 0, pressure: 8.0, height: 0.0, area: 1.0, momentum: 1.5, velocity: (0.0, 0.0, 0.0) },
            Element { id: 2, element_type: 2, nodes: vec![2, 3], faces: vec![2], mass: 1.0, neighbor_ref: 0, pressure: 6.0, height: 0.0, area: 1.0, momentum: 1.0, velocity: (0.0, 0.0, 0.0) },
        ];

        // Define faces between elements
        let faces = vec![
            Face { id: 0, nodes: (1, 2), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 1, nodes: (2, 3), velocity: (0.0, 0.0), area: 1.0 },
        ];

        // Create the mesh
        let mut mesh = Mesh {
            nodes: vec![],
            neighbors: vec![],
            elements,
            faces,
            face_element_relations: vec![], // Populate this as needed
        };

        // Set up boundary conditions
        let boundary_element: Vec<BoundaryElement> = elements
            .into_iter()
            .map(|element| BoundaryElement {
                element: Rc::new(RefCell::new(element)),
                boundary_type: BoundaryType::Periodic,
            }).collect();

        // Instantiate the solvers
        let crank_nicolson_solver = CrankNicolsonSolver {};
        let eddy_viscosity_solver = EddyViscositySolver { nu_t: 0.1 }; // Eddy viscosity coefficient
        let mut flux_solver = FluxSolver {};

        // Define a time stepper
        let time_stepper = ExplicitEuler { dt };

        // Run the simulation over time
        for _ in 0..(total_time / dt) as usize {
            // Apply boundary conditions to the entire boundary
            for boundary in &mut mesh.boundaries {
                boundary.apply(&mut mesh, &mut flow_field, time_step);
            }
            // Flux and pressure updates
            for i in 0..mesh.faces.len() {
                let (left_element, right_element) = mesh.elements.split_at_mut(i + 1);
                let left_element = &mut left_element[i];
                let right_element = &mut right_element[0];

                let pressure_diff = left_element.pressure - right_element.pressure;
                let flux = pressure_diff * mesh.faces[i].area;

                // Update momentum using Crank-Nicolson method
                left_element.momentum = crank_nicolson_solver.crank_nicolson_update(flux, left_element.momentum, dt);
                right_element.momentum = crank_nicolson_solver.crank_nicolson_update(-flux, right_element.momentum, dt);

                // Adjust pressures
                let pressure_transfer = 0.01 * flux * dt;

                left_element.pressure -= pressure_transfer;
                right_element.pressure += pressure_transfer;

                left_element.pressure = left_element.pressure.max(0.0);
                right_element.pressure = right_element.pressure.max(0.0);
            }

            // Apply horizontal eddy viscosity
            for i in 0..mesh.elements.len() - 1 {
                let (left_element, right_element) = mesh.elements.split_at_mut(i + 1);
                let left_element = &mut left_element[i];
                let right_element = &mut right_element[0];

                eddy_viscosity_solver.apply_diffusion(left_element, right_element, dt);
            }

            // Time stepping using the updated `mesh` and `flux_solver`
            time_stepper.step(&mut mesh, &mut flux_solver);
        }

        // Final assertions
        for element in &mesh.elements {
            assert!(element.momentum > 0.0, "Momentum should remain positive in all elements");
            assert!(element.pressure > 0.0, "Pressure should remain positive in all elements");
        }
    }

}
