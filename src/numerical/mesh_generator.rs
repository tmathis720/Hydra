use crate::domain::{Node, Element, Face, Mesh};

pub struct MeshGenerator;

impl MeshGenerator {
    /// Generates a rectangular mesh with a specified width, height, and resolution.
    pub fn generate_rectangle(width: f64, height: f64, nx: usize, ny: usize) -> Mesh {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut faces = Vec::new();

        // Generate nodes
        let dx = width / (nx as f64);
        let dy = height / (ny as f64);
        let mut node_id = 0;

        for j in 0..=ny {
            for i in 0..=nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                nodes.push(Node {
                    id: node_id,
                    position: (x, y, 0.0),
                });
                node_id += 1;
            }
        }

        // Generate elements (quadrilaterals, element_type = 3 for 4-node quadrangle)
        let mut element_id = 0;
        for j in 0..ny {
            for i in 0..nx {
                let n1 = j * (nx + 1) + i;
                let n2 = n1 + 1;
                let n3 = n1 + (nx + 1) + 1;
                let n4 = n1 + (nx + 1);

                elements.push(Element {
                    id: element_id,
                    nodes: vec![n1, n2, n3, n4],
                    faces: vec![], // Initialized but left empty
                    pressure: 0.0,
                    height: 0.0,
                    area: 0.0,
                    neighbor_ref: 0,
                    mass: 1.0,
                    momentum: 0.0,
                    element_type: 3, // 4-node quadrangle
                    velocity: (0.0, 0.0, 0.0),
                });
                element_id += 1;
            }
        }

        // Generate faces for the rectangle (boundary edges, element_type = 1 for 2-node line)
        let mut face_id = 0;
        for j in 0..ny {
            for i in 0..nx {
                let n1 = j * (nx + 1) + i;
                let n2 = n1 + 1;
                let n3 = n1 + (nx + 1);
                let n4 = n3 + 1;

                faces.push(Face {
                    id: face_id,
                    nodes: (n1, n2),
                    velocity: (0.0, 0.0),
                    area: 1.0,
                });
                face_id += 1;
                faces.push(Face {
                    id: face_id,
                    nodes: (n2, n4),
                    velocity: (0.0, 0.0),
                    area: 1.0,
                });
                face_id += 1;
                faces.push(Face {
                    id: face_id,
                    nodes: (n4, n3),
                    velocity: (0.0, 0.0),
                    area: 1.0,
                });
                face_id += 1;
                faces.push(Face {
                    id: face_id,
                    nodes: (n3, n1),
                    velocity: (0.0, 0.0),
                    area: 1.0,
                });
                face_id += 1;
            }
        }

        Mesh {
            nodes,
            elements,
            faces,
            face_element_relations: vec![],
            neighbors: vec![],
        }
    }

    /// Generates a triangular mesh (2D) based on a circle geometry.
    pub fn generate_circle(radius: f64, num_divisions: usize) -> Mesh {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut faces = Vec::new();

        // Center node
        nodes.push(Node {
            id: 0,
            position: (0.0, 0.0, 0.0),
        });

        // Generate outer nodes on the circle
        let mut node_id = 1;
        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            let x = radius * theta.cos();
            let y = radius * theta.sin();
            nodes.push(Node {
                id: node_id,
                position: (x, y, 0.0),
            });
            node_id += 1;
        }

        // Generate elements (triangles, element_type = 2 for 3-node triangle)
        let mut element_id = 0;
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;
            elements.push(Element {
                id: element_id,
                nodes: vec![0, i + 1, next + 1],
                faces: vec![],
                pressure: 0.0,
                height: 0.0,
                area: 0.0,
                neighbor_ref: 0,
                mass: 1.0,
                momentum: 0.0,
                element_type: 2, // 3-node triangle
                velocity: (0.0, 0.0, 0.0),
            });
            element_id += 1;
        }

        // Generate faces on the outer boundary (2-node line)
        let mut face_id = 0;
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;
            faces.push(Face {
                id: face_id,
                nodes: (i + 1, next + 1),
                velocity: (0.0, 0.0),
                area: 1.0,
            });
            face_id += 1;
        }

        Mesh {
            nodes,
            elements,
            faces,
            face_element_relations: vec![],
            neighbors: vec![],
        }
    }

    /// Generates an elliptical mesh.
    pub fn generate_ellipse(a: f64, b: f64, num_divisions: usize) -> Mesh {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut faces = Vec::new();

        // Center node
        nodes.push(Node {
            id: 0,
            position: (0.0, 0.0, 0.0),
        });

        // Generate outer nodes on the ellipse
        let mut node_id = 1;
        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            let x = a * theta.cos();
            let y = b * theta.sin();
            nodes.push(Node {
                id: node_id,
                position: (x, y, 0.0),
            });
            node_id += 1;
        }

        // Generate elements (triangles, element_type = 2 for 3-node triangle)
        let mut element_id = 0;
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;
            elements.push(Element {
                id: element_id,
                nodes: vec![0, i + 1, next + 1],
                faces: vec![],
                pressure: 0.0,
                height: 0.0,
                area: 0.0,
                neighbor_ref: 0,
                mass: 1.0,
                momentum: 0.0,
                element_type: 2, // 3-node triangle
                velocity: (0.0, 0.0, 0.0),
            });
            element_id += 1;
        }

        // Generate faces on the outer boundary (2-node line)
        let mut face_id = 0;
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;
            faces.push(Face {
                id: face_id,
                nodes: (i + 1, next + 1),
                velocity: (0.0, 0.0),
                area: 1.0,
            });
            face_id += 1;
        }

        Mesh {
            nodes,
            elements,
            faces,
            face_element_relations: vec![],
            neighbors: vec![],
        }
    }

    /// Generates a simple mesh for a 3D cube.
    pub fn generate_cube(side_length: f64) -> Mesh {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let faces = Vec::new();

        // Generate 8 corner nodes of the cube
        let half_side = side_length / 2.0;
        let cube_coords = [
            (-half_side, -half_side, -half_side),
            (half_side, -half_side, -half_side),
            (half_side, half_side, -half_side),
            (-half_side, half_side, -half_side),
            (-half_side, -half_side, half_side),
            (half_side, -half_side, half_side),
            (half_side, half_side, half_side),
            (-half_side, half_side, half_side),
        ];

        for (id, &(x, y, z)) in cube_coords.iter().enumerate() {
            nodes.push(Node {
                id: id as u32,
                position: (x, y, z),
            });
        }

        // Generate hexahedral element (element_type = 5 for 8-node hexahedron)
        elements.push(Element {
            id: 0,
            nodes: vec![0, 1, 2, 3, 4, 5, 6, 7],
            faces: vec![],
            pressure: 0.0,
            height: 0.0,
            area: 0.0,
            neighbor_ref: 0,
            mass: 1.0,
            momentum: 0.0,
            element_type: 5, // 8-node hexahedron
            velocity: (0.0, 0.0, 0.0),
        });

        // (Optional) Generate faces for the cube
        // You can define 6 faces for the cube using the corner nodes.

        Mesh {
            nodes,
            elements,
            faces,
            face_element_relations: vec![],
            neighbors: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::numerical::mesh_generator::MeshGenerator;
    use crate::domain::Mesh;

    #[test]
    fn test_generate_rectangle() {
        let width = 10.0;
        let height = 5.0;
        let nx = 4;
        let ny = 2;

        let mesh: Mesh = MeshGenerator::generate_rectangle(width, height, nx, ny);

        // Check the number of nodes
        let expected_node_count = (nx + 1) * (ny + 1);
        assert_eq!(mesh.nodes.len(), expected_node_count, "Node count should be correct for rectangle mesh");

        // Check the number of elements (quadrilaterals)
        let expected_element_count = nx * ny;
        assert_eq!(mesh.elements.len(), expected_element_count, "Element count should be correct for rectangle mesh");

        // Check the number of faces
        let expected_face_count = 4 * nx * ny;
        assert_eq!(mesh.faces.len(), expected_face_count, "Face count should be correct for rectangle mesh");
    }

    #[test]
    fn test_generate_circle() {
        let radius = 5.0;
        let num_divisions = 6;

        let mesh: Mesh = MeshGenerator::generate_circle(radius, num_divisions);

        // Check the number of nodes
        let expected_node_count = num_divisions + 1; // center + outer nodes
        assert_eq!(mesh.nodes.len(), expected_node_count, "Node count should be correct for circle mesh");

        // Check the number of elements (triangles)
        let expected_element_count = num_divisions; // same as number of divisions
        assert_eq!(mesh.elements.len(), expected_element_count, "Element count should be correct for circle mesh");

        // Check the number of faces (outer boundary edges)
        let expected_face_count = num_divisions;
        assert_eq!(mesh.faces.len(), expected_face_count, "Face count should be correct for circle mesh");
    }

    #[test]
    fn test_generate_ellipse() {
        let a = 6.0;
        let b = 3.0;
        let num_divisions = 8;

        let mesh: Mesh = MeshGenerator::generate_ellipse(a, b, num_divisions);

        // Check the number of nodes
        let expected_node_count = num_divisions + 1; // center + outer nodes
        assert_eq!(mesh.nodes.len(), expected_node_count, "Node count should be correct for ellipse mesh");

        // Check the number of elements (triangles)
        let expected_element_count = num_divisions;
        assert_eq!(mesh.elements.len(), expected_element_count, "Element count should be correct for ellipse mesh");

        // Check the number of faces (outer boundary edges)
        let expected_face_count = num_divisions;
        assert_eq!(mesh.faces.len(), expected_face_count, "Face count should be correct for ellipse mesh");
    }

    #[test]
    fn test_generate_cube() {
        let side_length = 2.0;

        let mesh: Mesh = MeshGenerator::generate_cube(side_length);

        // Check the number of nodes (8 for a cube)
        let expected_node_count = 8;
        assert_eq!(mesh.nodes.len(), expected_node_count, "Node count should be correct for cube");

        // Check the number of elements (1 hexahedron)
        let expected_element_count = 1;
        assert_eq!(mesh.elements.len(), expected_element_count, "Element count should be correct for cube");

        // (Optional) Check the number of faces (this can be added later if faces are generated for the cube)
    }
}
