use crate::domain::{mesh::Mesh, MeshEntity};

pub struct MeshGenerator;

impl MeshGenerator {
    /// Generates a 2D rectangular mesh with a specified width, height, and resolution (nx, ny).
    pub fn generate_rectangle_2d(width: f64, height: f64, nx: usize, ny: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_grid_nodes_2d(width, height, nx, ny);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position).unwrap();
        }
        Self::generate_quadrilateral_cells(&mut mesh, nx, ny);
        mesh
    }

    /// Generates a 3D rectangular mesh with a specified width, height, depth, and resolution (nx, ny, nz).
    pub fn generate_rectangle_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_grid_nodes_3d(width, height, depth, nx, ny, nz);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position).unwrap();
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
            mesh.set_vertex_coordinates(id, position).unwrap();
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
                mesh.add_entity(cell.clone()).unwrap();
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n1)).unwrap();
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n2)).unwrap();
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n3)).unwrap();
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n4)).unwrap();
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
                    mesh.add_entity(cell.clone()).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n1)).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n2)).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n3)).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n4)).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n5)).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n6)).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n7)).unwrap();
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n8)).unwrap();
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
            mesh.add_entity(cell.clone()).unwrap();
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(0)).unwrap();
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(i + 1)).unwrap();
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(next + 1)).unwrap();
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
                        mesh.add_entity(face.clone()).unwrap();
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v1)).unwrap();
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v2)).unwrap();
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v3)).unwrap();
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v4)).unwrap();
                    }
                }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::MeshGenerator;
    use crate::domain::MeshEntity;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_generate_rectangle_2d() {
        let width = 2.0;
        let height = 1.0;
        let nx = 2;
        let ny = 1;

        let mesh = MeshGenerator::generate_rectangle_2d(width, height, nx, ny);
        // The mesh should have (nx+1)*(ny+1) vertices
        let expected_vertices = (nx + 1) * (ny + 1);
        // The mesh should have nx*ny cells
        let expected_cells = nx * ny;

        // Check vertex count
        let vertex_count = mesh.count_entities(&MeshEntity::Vertex(0));
        assert_eq!(vertex_count, expected_vertices, "Incorrect number of vertices for 2D rectangle.");

        // Check cell count
        let cell_count = mesh.count_entities(&MeshEntity::Cell(0));
        assert_eq!(cell_count, expected_cells, "Incorrect number of cells for 2D rectangle.");

        // Verify vertex coordinates
        // Expected coordinates: 
        // For nx=2, ny=1: vertices at (0,0), (1,0), (2,0), (0,1), (1,1), (2,1)
        for i in 0..=nx {
            for j in 0..=ny {
                let id = j * (nx + 1) + i;
                let coords = mesh.get_vertex_coordinates(id).expect("Vertex not found");
                assert!(approx_eq(coords[0], i as f64 * (width/nx as f64), 1e-12));
                assert!(approx_eq(coords[1], j as f64 * (height/ny as f64), 1e-12));
                assert!(approx_eq(coords[2], 0.0, 1e-12));
            }
        }
    }

    #[test]
    fn test_generate_rectangle_3d() {
        let width = 2.0;
        let height = 1.0;
        let depth = 3.0;
        let nx = 2;
        let ny = 1;
        let nz = 2;

        let mesh = MeshGenerator::generate_rectangle_3d(width, height, depth, nx, ny, nz);
        // (nx+1)*(ny+1)*(nz+1) vertices
        let expected_vertices = (nx + 1) * (ny + 1) * (nz + 1);
        // nx*ny*nz cells
        let expected_cells = nx * ny * nz;

        let vertex_count = mesh.count_entities(&MeshEntity::Vertex(0));
        assert_eq!(vertex_count, expected_vertices, "Incorrect number of vertices for 3D rectangle.");

        let cell_count = mesh.count_entities(&MeshEntity::Cell(0));
        assert_eq!(cell_count, expected_cells, "Incorrect number of cells for 3D rectangle.");

        // Check a few sample coordinates:
        // vertex at (i, j, k) => x = i*dx, y = j*dy, z = k*dz
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let dz = depth / nz as f64;

        // A vertex in the "top corner" (i=nx, j=ny, k=nz):
        let corner_id = nz * (ny + 1)*(nx + 1) + ny*(nx + 1) + nx;
        let corner_coords = mesh.get_vertex_coordinates(corner_id).expect("Corner vertex not found");
        assert!(approx_eq(corner_coords[0], nx as f64 * dx, 1e-12));
        assert!(approx_eq(corner_coords[1], ny as f64 * dy, 1e-12));
        assert!(approx_eq(corner_coords[2], nz as f64 * dz, 1e-12));
    }

    #[test]
    fn test_generate_circle() {
        let radius = 1.0;
        let divisions = 4; // e.g., 4 divisions form a square-like approximation

        let mesh = MeshGenerator::generate_circle(radius, divisions);

        // The mesh should have num_divisions + 1 vertices (including center)
        let expected_vertices = divisions + 1;
        // It forms num_divisions triangular cells
        let expected_cells = divisions;

        let vertex_count = mesh.count_entities(&MeshEntity::Vertex(0));
        assert_eq!(vertex_count, expected_vertices, "Incorrect vertex count for circle mesh.");

        let cell_count = mesh.count_entities(&MeshEntity::Cell(0));
        assert_eq!(cell_count, expected_cells, "Incorrect cell count for circle mesh.");

        // Check center vertex at index 0
        let center_coords = mesh.get_vertex_coordinates(0).expect("Center vertex not found");
        assert!(approx_eq(center_coords[0], 0.0, 1e-12));
        assert!(approx_eq(center_coords[1], 0.0, 1e-12));
        assert!(approx_eq(center_coords[2], 0.0, 1e-12));

        // Check boundary vertices are on a circle of given radius
        for i in 1..=divisions {
            let coords = mesh.get_vertex_coordinates(i).expect("Boundary vertex not found");
            let r = (coords[0]*coords[0] + coords[1]*coords[1]).sqrt();
            assert!(approx_eq(r, radius, 1e-12), "Vertex not on the expected radius.");
            assert!(approx_eq(coords[2], 0.0, 1e-12));
        }

        // Check that each cell forms a triangle with the center vertex
        for c in 0..cell_count {
            // We know each cell is formed as (center, i, i+1)
            // Just ensure no panic occurs and we have correct indexing
            // Detailed topological checks can be done if needed
            let cells = mesh.get_cells();
            let cell_entity = cells[c];
            let cone = mesh.sieve.cone(&cell_entity).unwrap_or_default();
            assert_eq!(cone.len(), 3, "Each cell should have exactly 3 vertices for triangular mesh.");
        }
    }

    #[test]
    fn test_empty_meshes() {
        // Edge cases: if nx or ny is zero
        let mesh_2d = MeshGenerator::generate_rectangle_2d(2.0, 1.0, 0, 1);
        // If nx=0, we actually have no cells, just a line of vertices:
        assert_eq!(mesh_2d.count_entities(&MeshEntity::Vertex(0)), 2, "Expect a line of vertices when nx=0");
        assert_eq!(mesh_2d.count_entities(&MeshEntity::Cell(0)), 0, "No cells should be generated if nx=0");

        let mesh_circle = MeshGenerator::generate_circle(1.0, 0);
        // If num_divisions=0, circle degenerates to a single point
        assert_eq!(mesh_circle.count_entities(&MeshEntity::Vertex(0)), 1, "Only center vertex if no divisions");
        assert_eq!(mesh_circle.count_entities(&MeshEntity::Cell(0)), 0, "No cells without divisions");
    }
}
