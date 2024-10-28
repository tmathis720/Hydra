use crate::domain::{mesh::Mesh, MeshEntity};

pub struct MeshGenerator;

impl MeshGenerator {
    /// Generates a 2D rectangular mesh with a specified width, height, and resolution (nx, ny).
    pub fn generate_rectangle_2d(width: f64, height: f64, nx: usize, ny: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_grid_nodes_2d(width, height, nx, ny);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position);
        }
        Self::generate_quadrilateral_cells(&mut mesh, nx, ny);
        mesh
    }

    /// Generates a 3D rectangular mesh with a specified width, height, depth, and resolution (nx, ny, nz).
    pub fn generate_rectangle_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_grid_nodes_3d(width, height, depth, nx, ny, nz);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position);
        }
        Self::generate_hexahedral_cells(&mut mesh, nx, ny, nz);
        Self::_generate_faces_3d(&mut mesh, nx, ny, nz);
        mesh
    }

    /// Generates a circular 2D mesh with a given radius and number of divisions.
    pub fn generate_circle(radius: f64, num_divisions: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_circle_nodes(radius, num_divisions);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position);
        }
        Self::generate_triangular_cells(&mut mesh, num_divisions);
        mesh
    }

    // --- Internal Helper Functions ---

    /// Generate 2D grid nodes for rectangular mesh
    fn generate_grid_nodes_2d(width: f64, height: f64, nx: usize, ny: usize) -> Vec<[f64; 3]> {
        let mut nodes = Vec::new();
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([i as f64 * dx, j as f64 * dy, 0.0]);
            }
        }
        nodes
    }

    /// Generate 3D grid nodes for rectangular mesh
    fn generate_grid_nodes_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
        let mut nodes = Vec::new();
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let dz = depth / nz as f64;
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    nodes.push([i as f64 * dx, j as f64 * dy, k as f64 * dz]);
                }
            }
        }
        nodes
    }

    /// Generate circle nodes for circular mesh
    fn generate_circle_nodes(radius: f64, num_divisions: usize) -> Vec<[f64; 3]> {
        let mut nodes = Vec::new();
        nodes.push([0.0, 0.0, 0.0]);
        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            nodes.push([radius * theta.cos(), radius * theta.sin(), 0.0]);
        }
        nodes
    }

    /// Generate quadrilateral cells for a 2D rectangular mesh
    fn generate_quadrilateral_cells(mesh: &mut Mesh, nx: usize, ny: usize) {
        let mut cell_id = 0;
        for j in 0..ny {
            for i in 0..nx {
                let n1 = j * (nx + 1) + i;
                let n2 = n1 + 1;
                let n3 = n1 + (nx + 1) + 1;
                let n4 = n1 + (nx + 1);
                let cell = MeshEntity::Cell(cell_id);
                cell_id += 1;
                mesh.add_entity(cell.clone());
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n1));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n2));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n3));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n4));
            }
        }
    }

    /// Generate hexahedral cells for a 3D rectangular mesh
    fn generate_hexahedral_cells(mesh: &mut Mesh, nx: usize, ny: usize, nz: usize) {
        let mut cell_id = 0;
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
                    let cell = MeshEntity::Cell(cell_id);
                    cell_id += 1;
                    mesh.add_entity(cell.clone());
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n1));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n2));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n3));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n4));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n5));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n6));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n7));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n8));
                }
            }
        }
    }

    /// Generate triangular cells for a circular mesh
    fn generate_triangular_cells(mesh: &mut Mesh, num_divisions: usize) {
        let mut cell_id = 0;
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;
            let cell = MeshEntity::Cell(cell_id);
            cell_id += 1;
            mesh.add_entity(cell.clone());
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(0));
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(i + 1));
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(next + 1));
        }
    }

    /// Generate faces for a 3D rectangular mesh.
    fn _generate_faces_3d(mesh: &mut Mesh, nx: usize, ny: usize, nz: usize) {
        let mut face_id = 0;
        
        // Loop over all cells to add faces
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

                    // Define the vertices for each face of a hexahedron
                    let faces = [
                        (n1, n2, n4, n3), // front face
                        (n5, n6, n8, n7), // back face
                        (n1, n5, n7, n3), // left face
                        (n2, n6, n8, n4), // right face
                        (n3, n4, n8, n7), // top face
                        (n1, n2, n6, n5), // bottom face
                    ];

                    // Add each face to the mesh
                    for &(v1, v2, v3, v4) in &faces {
                        let face = MeshEntity::Face(face_id);
                        face_id += 1;
                        mesh.add_entity(face.clone());
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v1));
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v2));
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v3));
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v4));
                    }
                }
            }
        }
    }

}
