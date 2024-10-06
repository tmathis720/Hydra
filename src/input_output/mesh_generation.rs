use crate::domain::{Mesh, MeshEntity};

impl Mesh {
    /// Generate a structured 2D grid mesh
    pub fn structured_2d(nx: usize, ny: usize, x_len: f64, y_len: f64) -> Self {
        let mut mesh = Mesh::new();
        let dx = x_len / (nx - 1) as f64;
        let dy = y_len / (ny - 1) as f64;

        let mut vertex_id = 0;
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                mesh.set_vertex_coordinates(vertex_id, [x, y, 0.0]);
                vertex_id += 1;
            }
        }

        for i in 0..(nx - 1) {
            for j in 0..(ny - 1) {
                let cell_id = i * (ny - 1) + j;
                let v0 = i * ny + j;
                let v1 = v0 + 1;
                let v2 = (i + 1) * ny + j;
                let v3 = v2 + 1;
                let cell = MeshEntity::Cell(cell_id);
                mesh.add_entity(cell);
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(v0));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(v1));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(v2));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(v3));
            }
        }
        mesh
    }

    /// Generate a structured 3D grid mesh
    pub fn structured_3d(nx: usize, ny: usize, nz: usize, x_len: f64, y_len: f64, z_len: f64) -> Self {
        // Similar to structured_2d but with 3D coordinates and hexahedral cells
        todo!()
    }
}
