use crate::domain::{Node, Element, Face, Mesh};
use nalgebra::Vector3;

pub struct MeshGenerator;

impl MeshGenerator {
    /// Generates a 2D rectangular mesh with a specified width, height, and resolution (nx, ny).
    pub fn generate_rectangle(width: f64, height: f64, nx: usize, ny: usize) -> Mesh {
        let nodes = MeshGenerator::generate_grid_nodes_2d(width, height, nx, ny);
        let elements = MeshGenerator::generate_quadrilateral_elements(nx, ny);
        let faces = MeshGenerator::generate_faces_2d(&elements);

        Mesh {
            nodes,
            elements,
            faces,
            ..Mesh::default()
        }
    }

    /// Generates a 3D rectangular mesh with a specified width, height, depth, and resolution (nx, ny, nz).
    pub fn generate_rectangle_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
        let nodes = MeshGenerator::generate_grid_nodes_3d(width, height, depth, nx, ny, nz);
        let elements = MeshGenerator::generate_hexahedral_elements(nx, ny, nz);
        let faces = MeshGenerator::generate_faces_3d(&elements);

        Mesh {
            nodes,
            elements,
            faces,
            ..Mesh::default()
        }
    }

    /// Generates a triangular mesh based on a circle geometry.
    pub fn generate_circle(radius: f64, num_divisions: usize) -> Mesh {
        let nodes = MeshGenerator::generate_circle_nodes(radius, num_divisions);
        let elements = MeshGenerator::generate_triangular_elements(num_divisions);
        let faces = MeshGenerator::generate_boundary_faces(&elements);

        Mesh {
            nodes,
            elements,
            faces,
            ..Mesh::default()
        }
    }

    /// Generates an elliptical mesh with a specified major and minor radius and resolution.
    pub fn generate_ellipse(a: f64, b: f64, num_divisions: usize) -> Mesh {
        let nodes = MeshGenerator::generate_ellipse_nodes(a, b, num_divisions);
        let elements = MeshGenerator::generate_triangular_elements(num_divisions);
        let faces = MeshGenerator::generate_boundary_faces(&elements);

        Mesh {
            nodes,
            elements,
            faces,
            ..Mesh::default()
        }
    }

    /// Generates a 3D cube mesh.
    pub fn generate_cube(side_length: f64) -> Mesh {
        let nodes = MeshGenerator::generate_cube_nodes(side_length);
        let elements = MeshGenerator::generate_hexahedral_elements(1, 1, 1);
        let faces = MeshGenerator::generate_faces_3d(&elements);

        Mesh {
            nodes,
            elements,
            faces,
            ..Mesh::default()
        }
    }

    // ----- Helper Functions -----

    /// Generates 2D grid nodes for a rectangular mesh.
    fn generate_grid_nodes_2d(width: f64, height: f64, nx: usize, ny: usize) -> Vec<Node> {
        let mut nodes = Vec::new();
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let mut node_id = 0;

        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push(Node {
                    id: node_id,
                    position: Vector3::new(i as f64 * dx, j as f64 * dy, 0.0),
                });
                node_id += 1;
            }
        }
        nodes
    }

    /// Generates 3D grid nodes for a rectangular mesh.
    fn generate_grid_nodes_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Vec<Node> {
        let mut nodes = Vec::new();
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let dz = depth / nz as f64;
        let mut node_id = 0;

        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    nodes.push(Node {
                        id: node_id,
                        position: Vector3::new(i as f64 * dx, j as f64 * dy, k as f64 * dz),
                    });
                    node_id += 1;
                }
            }
        }
        nodes
    }

    /// Generates circle nodes for a circular mesh.
    fn generate_circle_nodes(radius: f64, num_divisions: usize) -> Vec<Node> {
        let mut nodes = Vec::new();
        nodes.push(Node {
            id: 0,
            position: Vector3::new(0.0, 0.0, 0.0),
        });

        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            nodes.push(Node {
                id: i as u32 + 1,
                position: Vector3::new(radius * theta.cos(), radius * theta.sin(), 0.0),
            });
        }
        nodes
    }

    /// Generates ellipse nodes for an elliptical mesh.
    fn generate_ellipse_nodes(a: f64, b: f64, num_divisions: usize) -> Vec<Node> {
        let mut nodes = Vec::new();
        nodes.push(Node {
            id: 0,
            position: Vector3::new(0.0, 0.0, 0.0),
        });

        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            nodes.push(Node {
                id: i as u32 + 1,
                position: Vector3::new(a * theta.cos(), b * theta.sin(), 0.0),
            });
        }
        nodes
    }

    /// Generates cube nodes for a simple cube mesh.
    fn generate_cube_nodes(side_length: f64) -> Vec<Node> {
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

        cube_coords
            .iter()
            .enumerate()
            .map(|(id, &(x, y, z))| Node {
                id: id as u32,
                position: Vector3::new(x, y, z),
            })
            .collect()
    }

    /// Generates quadrilateral elements for a 2D rectangular mesh.
    fn generate_quadrilateral_elements(nx: usize, ny: usize) -> Vec<Element> {
        let mut elements = Vec::new();
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
                    element_type: 3, // Quadrilateral
                    ..Element::default()
                });
                element_id += 1;
            }
        }
        elements
    }

    /// Generates hexahedral elements for a 3D rectangular mesh.
    fn generate_hexahedral_elements(nx: usize, ny: usize, nz: usize) -> Vec<Element> {
        let mut elements = Vec::new();
        let mut element_id = 0;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let n1 = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                    let n2 = n1 + 1;
                    let n3 = n1 + (nx + 1);
                    let n4 = n3 + 1;
                    let n5 = n1 + (ny + 1) * (nx + 1);
                    let n6 = n5 + 1;
                    let n7 = n5 + (nx + 1);
                    let n8 = n7 + 1;

                    elements.push(Element {
                        id: element_id,
                        nodes: vec![n1, n2, n4, n3, n5, n6, n8, n7],
                        element_type: 4, // Hexahedral
                        ..Element::default()
                    });
                    element_id += 1;
                }
            }
        }
        elements
    }

    /// Generates triangular elements for circular/elliptical meshes.
    fn generate_triangular_elements(num_divisions: usize) -> Vec<Element> {
        let mut elements = Vec::new();
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;

            elements.push(Element {
                id: i as u32,
                nodes: vec![0, i + 1, next + 1],
                element_type: 2, // Triangular
                ..Element::default()
            });
        }
        elements
    }

    /// Generates faces for a 2D rectangular mesh.
    fn generate_faces_2d(elements: &[Element]) -> Vec<Face> {
        let mut faces = Vec::new();
        let mut face_id = 0;

        for element in elements {
            let (n1, n2, n3, n4) = (element.nodes[0], element.nodes[1], element.nodes[2], element.nodes[3]);

            // Bottom face
            faces.push(Face {
                id: face_id,
                nodes: vec![n1, n2],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Right face
            faces.push(Face {
                id: face_id,
                nodes: vec![n2, n3],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Top face
            faces.push(Face {
                id: face_id,
                nodes: vec![n3, n4],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Left face
            faces.push(Face {
                id: face_id,
                nodes: vec![n4, n1],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;
        }
        faces
    }

    /// Generates faces for a 3D rectangular mesh.
    fn generate_faces_3d(elements: &[Element]) -> Vec<Face> {
        let mut faces = Vec::new();
        let mut face_id = 0;

        for element in elements {
            let (n1, n2, n3, n4, n5, n6, n7, n8) = (
                element.nodes[0], element.nodes[1], element.nodes[2], element.nodes[3],
                element.nodes[4], element.nodes[5], element.nodes[6], element.nodes[7]
            );

            // Front face
            faces.push(Face {
                id: face_id,
                nodes: vec![n1, n2, n4, n3],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Back face
            faces.push(Face {
                id: face_id,
                nodes: vec![n5, n6, n8, n7],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Left face
            faces.push(Face {
                id: face_id,
                nodes: vec![n1, n5, n7, n3],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Right face
            faces.push(Face {
                id: face_id,
                nodes: vec![n2, n6, n8, n4],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Bottom face
            faces.push(Face {
                id: face_id,
                nodes: vec![n1, n2, n6, n5],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;

            // Top face
            faces.push(Face {
                id: face_id,
                nodes: vec![n3, n4, n8, n7],
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
            face_id += 1;
        }
        faces
    }

    /// Generate boundary faces for circular or elliptical meshes.
    fn generate_boundary_faces(elements: &[Element]) -> Vec<Face> {
        let mut faces = Vec::new();
        for (i, element) in elements.iter().enumerate() {
            faces.push(Face {
                id: i as u32,
                nodes: vec![element.nodes[1], element.nodes[2]], // Boundary edge
                velocity: Vector3::new(0.0, 0.0, 0.0),
                area: 1.0,
                ..Face::default()
            });
        }
        faces
    }
}
