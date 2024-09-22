use crate::domain::{Mesh, DPoint, DPointType};
use nalgebra::Vector3;

pub struct MeshGenerator;

impl MeshGenerator {
    /// Generates a 2D rectangular mesh with a specified width, height, and resolution (nx, ny).
    pub fn generate_rectangle(width: f64, height: f64, nx: usize, ny: usize) -> Mesh {
        let mut mesh = Mesh::new();
        MeshGenerator::generate_grid_dpoints_2d(&mut mesh, width, height, nx, ny);
        MeshGenerator::generate_quadrilateral_elements(&mut mesh, nx, ny);
        mesh
    }

    /// Generates a 3D rectangular mesh with a specified width, height, depth, and resolution (nx, ny, nz).
    pub fn generate_rectangle_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
        let mut mesh = Mesh::new();
        MeshGenerator::generate_grid_dpoints_3d(&mut mesh, width, height, depth, nx, ny, nz);
        MeshGenerator::generate_hexahedral_elements(&mut mesh, nx, ny, nz);
        mesh
    }

    /// Generates a triangular mesh based on a circle geometry.
    pub fn generate_circle(radius: f64, num_divisions: usize) -> Mesh {
        let mut mesh = Mesh::new();
        MeshGenerator::generate_circle_dpoints(&mut mesh, radius, num_divisions);
        MeshGenerator::generate_triangular_elements(&mut mesh, num_divisions);
        mesh
    }

    /// Generates an elliptical mesh with a specified major and minor radius and resolution.
    pub fn generate_ellipse(a: f64, b: f64, num_divisions: usize) -> Mesh {
        let mut mesh = Mesh::new();
        MeshGenerator::generate_ellipse_dpoints(&mut mesh, a, b, num_divisions);
        MeshGenerator::generate_triangular_elements(&mut mesh, num_divisions);
        mesh
    }

    /// Generates a 3D cube mesh.
    pub fn generate_cube(side_length: f64) -> Mesh {
        let mut mesh = Mesh::new();
        MeshGenerator::generate_cube_dpoints(&mut mesh, side_length);
        MeshGenerator::generate_hexahedral_elements(&mut mesh, 1, 1, 1);
        mesh
    }

    // ----- Helper Functions -----

    /// Generates 2D grid DPoints for a rectangular mesh.
    fn generate_grid_dpoints_2d(mesh: &mut Mesh, width: f64, height: f64, nx: usize, ny: usize) {
        let dx = width / nx as f64;
        let dy = height / ny as f64;

        for j in 0..=ny {
            for i in 0..=nx {
                let id = j * (nx + 1) + i;
                let dpoint = DPoint::new(id, DPointType::Vertex, 1);
                mesh.add_point(id, DPointType::Vertex);
            }
        }
    }

    /// Generates 3D grid DPoints for a rectangular mesh.
    fn generate_grid_dpoints_3d(mesh: &mut Mesh, width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) {
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let dz = depth / nz as f64;

        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let id = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                    let dpoint = DPoint::new(id, DPointType::Vertex, 1);
                    mesh.add_point(id, DPointType::Vertex);
                }
            }
        }
    }

    /// Generates DPoints for a circular mesh.
    fn generate_circle_dpoints(mesh: &mut Mesh, radius: f64, num_divisions: usize) {
        let center = DPoint::new(0, DPointType::Vertex, 1);
        mesh.add_point(0, DPointType::Vertex);

        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            let id = i + 1;
            mesh.add_point(id, DPointType::Vertex);
        }
    }

    /// Generates DPoints for an elliptical mesh.
    fn generate_ellipse_dpoints(mesh: &mut Mesh, a: f64, b: f64, num_divisions: usize) {
        let center = DPoint::new(0, DPointType::Vertex, 1);
        mesh.add_point(0, DPointType::Vertex);

        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            let id = i + 1;
            mesh.add_point(id, DPointType::Vertex);
        }
    }

    /// Generates DPoints for a 3D cube mesh.
    fn generate_cube_dpoints(mesh: &mut Mesh, side_length: f64) {
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

        for (id, _) in cube_coords.iter().enumerate() {
            mesh.add_point(id, DPointType::Vertex);
        }
    }

    /// Generates quadrilateral elements for a 2D rectangular mesh.
    fn generate_quadrilateral_elements(mesh: &mut Mesh, nx: usize, ny: usize) {
        for j in 0..ny {
            for i in 0..nx {
                let n1 = j * (nx + 1) + i;
                let n2 = n1 + 1;
                let n3 = n1 + (nx + 1) + 1;
                let n4 = n1 + (nx + 1);

                let id = mesh.dpoints.len();
                let element = DPoint::new(id, DPointType::Cell, 4);
                mesh.add_point(id, DPointType::Cell);
            }
        }
    }

    /// Generates hexahedral elements for a 3D rectangular mesh.
    fn generate_hexahedral_elements(mesh: &mut Mesh, nx: usize, ny: usize, nz: usize) {
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

                    let id = mesh.dpoints.len();
                    mesh.add_point(id, DPointType::Cell);
                }
            }
        }
    }

    /// Generates triangular elements for circular/elliptical meshes.
    fn generate_triangular_elements(mesh: &mut Mesh, num_divisions: usize) {
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;

            let id = mesh.dpoints.len();
            let element = DPoint::new(id, DPointType::Cell, 3);
            mesh.add_point(id, DPointType::Cell);
        }
    }
}
