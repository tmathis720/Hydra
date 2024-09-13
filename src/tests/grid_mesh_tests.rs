#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::{FluxSolver, SemiImplicitSolver};

    #[test]
    fn test_non_uniform_grid() {
        let dt = 0.01;
        let total_time = 10.0;

        // Create a set of elements with non-uniform sizes (varying mass)
        let mut elements: Vec<Element> = vec![
            Element { id: 0, element_type: 2, nodes: vec![0, 1], faces: vec![0], mass: 2.0, neighbor_ref: 0, pressure: 15.0, momentum: 3.0, height: 0.0, area: 0.0, velocity: (0.0, 0.0, 0.0) },
            Element { id: 1, element_type: 2, nodes: vec![1, 2], faces: vec![1], mass: 1.5, neighbor_ref: 0, pressure: 12.0, momentum: 2.5, height: 0.0, area: 0.0, velocity: (0.0, 0.0, 0.0) },
            Element { id: 2, element_type: 2, nodes: vec![2, 3], faces: vec![2], mass: 1.0, neighbor_ref: 0, pressure: 10.0, momentum: 2.0, height: 0.0, area: 0.0, velocity: (0.0, 0.0, 0.0) },
        ];

        // Define faces between elements (different areas for non-uniform grid)
        let faces = vec![
            Face { id: 0, nodes: vec![1, 2], velocity: (0.0, 0.0, 0.0), area: 1.0 },
            Face { id: 1, nodes: vec![2, 3], velocity: (0.0, 0.0, 0.0), area: 0.8 },
        ];

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            for i in 0..faces.len() {
                let flux = flux_solver.compute_flux(&faces[i], &elements[i], &elements[i + 1]);

                // Update momentum using semi-implicit solver
                elements[i].momentum = semi_implicit_solver.semi_implicit_update(
                    -flux * (elements[i].momentum / elements[i].mass),
                    elements[i].momentum,
                    dt,
                );
                elements[i + 1].momentum = semi_implicit_solver.semi_implicit_update(
                    flux * (elements[i + 1].momentum / elements[i + 1].mass),
                    elements[i + 1].momentum,
                    dt,
                );

                // Ensure momentum and mass are conserved
                assert!(elements[i].momentum > 0.0, "Momentum should remain positive in element {}", i);
                assert!(elements[i + 1].momentum > 0.0, "Momentum should remain positive in element {}", i + 1);
            }
        }
    }


    // Helper function to create a basic face between two elements
    fn create_face() -> Face {
        Face {
            id: 0,
            nodes: vec![1, 2], // Nodes shared between left and right elements
            velocity: (0.0, 0.0, 0.0), // Initial velocity is zero
            area: 1.0, // Example face area
        }
    }

    #[test]
    fn test_flux_positive_when_left_pressure_higher() {
        // Create the left element (higher pressure)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1, 0],
            faces: vec![0, 1],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 2.0, // Higher pressure in the left element
            momentum: 0.0,
        };

        // Create the right element (lower pressure)
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2, 0],
            faces: vec![1, 2],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 1.0, // Lower pressure in the right element
            momentum: 0.0,
        };

        let face = create_face();
        let flux_solver = FluxSolver {};

        // Compute flux and expect positive flux (flow from left to right)
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert!(flux > 0.0, "Flux should be positive since left element has higher pressure");
    }

    #[test]
    fn test_flux_negative_when_right_pressure_higher() {
        // Create the left element (lower pressure)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1, 0],
            faces: vec![0, 1],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 1.0, // Lower pressure in the left element
            momentum: 0.0,
        };

        // Create the right element (higher pressure)
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2, 0],
            faces: vec![1, 2],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 3.0, // Higher pressure in the right element
            momentum: 0.0,
        };

        let face = create_face();
        let flux_solver = FluxSolver {};

        // Compute flux and expect negative flux (flow from right to left)
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert!(flux < 0.0, "Flux should be negative since right element has higher pressure");
    }

    #[test]
    fn test_flux_zero_when_pressures_equal() {
        // Create the left element (equal pressure)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1, 0],
            faces: vec![0, 1],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 2.0, // Same pressure in the left element
            momentum: 0.0,
        };

        // Create the right element (equal pressure)
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2, 0],
            faces: vec![1, 2],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 2.0, // Same pressure in the right element
            momentum: 0.0,
        };

        let face = create_face();
        let flux_solver = FluxSolver {};

        // Compute flux and expect zero flux (no flow)
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert_eq!(flux, 0.0, "Flux should be zero when both elements have equal pressure");
    }

    #[test]
    fn test_multi_element_grid() {
        let dt = 0.01;
        let total_time = 20.0;

        // Define a small grid of elements (e.g., 4 elements in a line)
        let mut elements: Vec<Element> = vec![
            Element { id: 0, element_type: 2, nodes: vec![0, 1], faces: vec![0], mass: 1.0, neighbor_ref: 0, pressure: 10.0, momentum: 3.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
            Element { id: 1, element_type: 2, nodes: vec![1, 2], faces: vec![1], mass: 1.0, neighbor_ref: 0, pressure: 7.0, momentum: 2.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
            Element { id: 2, element_type: 2, nodes: vec![2, 3], faces: vec![2], mass: 1.0, neighbor_ref: 0, pressure: 5.0, momentum: 1.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
            Element { id: 3, element_type: 2, nodes: vec![3, 4], faces: vec![3], mass: 1.0, neighbor_ref: 0, pressure: 3.0, momentum: 1.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
        ];

        // Define faces between elements
        let faces = vec![
            Face { id: 0, nodes: vec![1, 2], velocity: (0.0, 0.0, 0.0), area: 1.0 },
            Face { id: 1, nodes: vec![2, 3], velocity: (0.0, 0.0, 0.0), area: 1.0 },
            Face { id: 2, nodes: vec![3, 4], velocity: (0.0, 0.0, 0.0), area: 1.0 },
        ];

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            for i in 0..faces.len() {
                let flux = flux_solver.compute_flux(&faces[i], &elements[i], &elements[i + 1]);

                // Update momentum with semi-implicit solver
                elements[i].momentum = semi_implicit_solver.semi_implicit_update(
                    -flux * (elements[i].momentum / elements[i].mass),
                    elements[i].momentum,
                    dt,
                );
                elements[i + 1].momentum = semi_implicit_solver.semi_implicit_update(
                    flux * (elements[i + 1].momentum / elements[i + 1].mass),
                    elements[i + 1].momentum,
                    dt,
                );

                // Assert positive momentum
                assert!(elements[i].momentum > 0.0, "Momentum should remain positive in element {}", i);
                assert!(elements[i + 1].momentum > 0.0, "Momentum should remain positive in element {}", i + 1);
            }
        }
    }

    use crate::input::gmsh::GmshParser;  // Ensure this path aligns with your code structure
    

    #[test]
    fn test_circle_mesh_import() {
        // Assuming you saved the circle mesh as `circle.msh`
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/circular_lake.msh2")
            .expect("Failed to load circle mesh");

        // Test for basic validity
        assert!(!nodes.is_empty(), "Circle mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Circle mesh elements should not be empty");

        // Expected counts based on your Gmsh file (you may need to adjust these numbers)
        let expected_node_count = 424;
        let expected_element_count = 849;

        // Check that the number of nodes matches the expected value
        assert_eq!(nodes.len(), expected_node_count, "Incorrect number of nodes in circle mesh");

        // Check that the number of elements matches the expected value
        assert_eq!(elements.len(), expected_element_count, "Incorrect number of elements in circle mesh");

        // You can also add tests for face counts, but they might need to be derived if not explicitly in Gmsh
    }

    #[test]
    fn test_coastal_island_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/coastal_island.msh2")
            .expect("Failed to load Coastal Island mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Coastal Island mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Coastal Island mesh elements should not be empty");

        // Add specific tests based on expected structure
        assert_eq!(nodes.len(), 1075, "Incorrect number of nodes in Coastal Island mesh");
        assert_eq!(elements.len(), 2154, "Incorrect number of elements in Coastal Island mesh");
    }

    #[test]
    fn test_lagoon_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/elliptical_lagoon.msh2")
            .expect("Failed to load Lagoon mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Lagoon mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Lagoon mesh elements should not be empty");

        // Further checks on expected properties
        assert_eq!(nodes.len(), 848, "Incorrect number of nodes in Lagoon mesh");
        assert_eq!(elements.len(), 1697, "Incorrect number of elements in Lagoon mesh");
    }

    #[test]
    fn test_meandering_river_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/meandering_river.msh2")
            .expect("Failed to load Meandering River mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Meandering River mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Meandering River mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 695, "Incorrect number of nodes in Meandering River mesh");
        assert_eq!(elements.len(), 1386, "Incorrect number of elements in Meandering River mesh");
    }

    #[test]
    fn test_polygon_estuary_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/polygon_estuary.msh2")
            .expect("Failed to load Polygon Estuary mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Polygon Estuary mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Polygon Estuary mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 469, "Incorrect number of nodes in Polygon Estuary mesh");
        assert_eq!(elements.len(), 941, "Incorrect number of elements in Polygon Estuary mesh");
    }

    #[test]
    fn test_rectangle_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/rectangle.msh2")
            .expect("Failed to load Rectangle mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Rectangle mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Rectangle mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 78, "Incorrect number of nodes in Rectangle mesh");
        assert_eq!(elements.len(), 158, "Incorrect number of elements in Rectangle mesh");
    }

    #[test]
    fn test_rectangle_channel_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/rectangular_channel.msh2")
            .expect("Failed to load Rectangular Channel mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Rectangular Channel mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Rectangular Channel mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 149, "Incorrect number of nodes in Rectangular Channel mesh");
        assert_eq!(elements.len(), 300, "Incorrect number of elements in Rectangular Channel mesh");
    }

    #[test]
    fn test_triangle_basin_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/triangular_basin.msh2")
            .expect("Failed to load Triangular Basin mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Triangular Basin mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Triangular Basin mesh elements should not be empty");


        // Further checks on the structure
        assert_eq!(nodes.len(), 66, "Incorrect number of nodes in Triangular Basin mesh");
        assert_eq!(elements.len(), 133, "Incorrect number of elements in Triangular Basin mesh");
    }
}
