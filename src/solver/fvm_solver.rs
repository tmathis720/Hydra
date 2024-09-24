use crate::Mesh;
use crate::{DirichletBC, NeumannBC};
use crate::Solver;
use crate::MeshEntity;
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use crate::geometry::{CellShape, FaceShape, Geometry}; // Import geometry module

/// FVM Solver Structure
pub struct FVMSolver {
    pub mesh: Mesh,                    // Mesh structure (cells, faces, etc.)
    pub dirichlet_bc: DirichletBC,     // Dirichlet boundary conditions
    pub neumann_bc: NeumannBC,         // Neumann boundary conditions
    pub matrix: DMatrix<f64>,          // System matrix
    pub rhs: DVector<f64>,             // RHS vector
    pub solution: DVector<f64>,        // Solution vector
    pub cell_id_to_index: HashMap<usize, usize>, // Mapping from cell IDs to indices
    pub index_to_cell_id: HashMap<usize, usize>, // Mapping from indices to cell IDs
    pub entity_to_index: HashMap<MeshEntity, usize>, // Mapping from MeshEntity to indices
    pub face_areas: HashMap<usize, f64>, // Mapping from face IDs to face areas
}

impl FVMSolver {
    /// Create a new FVM solver instance
    pub fn new(mesh: Mesh, dirichlet_bc: DirichletBC, neumann_bc: NeumannBC) -> Self {
        // Get the list of cells
        let cells = mesh.get_cells();
        let num_cells = cells.len();
        let matrix = DMatrix::zeros(num_cells, num_cells);
        let rhs = DVector::zeros(num_cells);
        let solution = DVector::zeros(num_cells);

        // Create mapping from cell IDs to indices
        let mut cell_id_to_index = HashMap::new();
        let mut index_to_cell_id = HashMap::new();
        let mut entity_to_index = HashMap::new();
        for (i, cell) in cells.iter().enumerate() {
            if let MeshEntity::Cell(cell_id) = *cell {
                cell_id_to_index.insert(cell_id, i);
                index_to_cell_id.insert(i, cell_id);
                entity_to_index.insert(*cell, i);
            }
        }

        FVMSolver {
            mesh,
            dirichlet_bc,
            neumann_bc,
            matrix,
            rhs,
            solution,
            cell_id_to_index,
            index_to_cell_id,
            entity_to_index,
            face_areas: HashMap::new(),
        }
    }

    /// Assemble the system matrix and RHS vector using FVM and the mesh structure
    fn assemble_matrix_rhs(&mut self) {
        // Reset matrix and rhs
        self.matrix.fill(0.0);
        self.rhs.fill(0.0);

        // Get the list of cells (now owned, not references)
        let cells = self.mesh.get_cells();

        // Create a Geometry instance
        let geometry = Geometry::new();

        // Compute face areas and store them
        self.compute_face_areas();

        // Loop over cells
        for cell in cells {
            let cell_id = if let MeshEntity::Cell(id) = cell { id } else { continue; };
            let i = self.cell_id_to_index[&cell_id];

            // Get cell vertices
            let cell_vertices = self.mesh.get_cell_vertices(&cell);

            // Compute cell volume
            let cell_volume = geometry.compute_cell_volume(CellShape::Tetrahedron, &cell_vertices);

            // Initialize diagonal term
            let mut diagonal_coefficient = 0.0;

            // Get faces of the cell
            if let Some(faces) = self.mesh.get_faces_of_cell(&cell) {
                for face in faces {
                    let face_id = face.id();

                    // Get face vertices
                    let face_vertices = self.mesh.get_face_vertices(&face);

                    // Get face area from precomputed values
                    let face_area = *self.face_areas.get(&face_id).unwrap();

                    // Compute face centroid
                    let face_centroid = geometry.compute_face_centroid(FaceShape::Triangle, &face_vertices);

                    // Get neighboring cells sharing this face
                    let neighbor_cells = self.mesh.get_cells_sharing_face(&face);

                    if neighbor_cells.len() == 2 {
                        // Internal face
                        let mut neighbor_iter = neighbor_cells.iter();
                        let cell_a = neighbor_iter.next().unwrap();
                        let cell_b = neighbor_iter.next().unwrap();

                        let other_cell = if cell_a.id() == cell_id { cell_b } else { cell_a };

                        if let MeshEntity::Cell(other_cell_id) = other_cell {
                            let j = self.cell_id_to_index[&other_cell_id];

                            // Compute distance between cell centroids
                            let other_cell_vertices = self.mesh.get_cell_vertices(other_cell);
                            let other_cell_centroid = geometry.compute_cell_centroid(CellShape::Tetrahedron, &other_cell_vertices);
                            let cell_centroid = geometry.compute_cell_centroid(CellShape::Tetrahedron, &cell_vertices);
                            let distance = Geometry::compute_distance(&cell_centroid, &other_cell_centroid);

                            // Calculate coefficient
                            let coefficient = self.calculate_flux_coefficient(face_area, distance);

                            // Update matrix entries
                            self.matrix[(i, i)] += coefficient;
                            self.matrix[(i, j)] -= coefficient;
                        }
                    } else if neighbor_cells.len() == 1 {
                        // Boundary face
                        if self.neumann_bc.is_bc(&face) {
                            // Neumann BC
                            let flux = self.neumann_bc.get_flux(&face);
                            self.rhs[i] += flux * face_area;
                        }
                        if self.dirichlet_bc.is_bc(&face) {
                            // Dirichlet BC
                            let value = self.dirichlet_bc.get_value(&face);

                            // Compute distance from cell centroid to face centroid
                            let cell_centroid = geometry.compute_cell_centroid(CellShape::Tetrahedron, &cell_vertices);
                            let distance = Geometry::compute_distance(&cell_centroid, &face_centroid);

                            // Coefficient for the face
                            let coefficient = self.calculate_flux_coefficient(face_area, distance);

                            // Modify matrix and RHS
                            self.matrix[(i, i)] += coefficient;
                            self.rhs[i] += coefficient * value;
                        }
                    } else {
                        // Should not happen
                    }
                }
            }

            // Multiply RHS by cell volume if needed
            self.rhs[i] *= cell_volume;
        }

        // Apply boundary conditions
        self.dirichlet_bc.apply_bc(&mut self.matrix, &mut self.rhs, &self.entity_to_index);
        self.neumann_bc.apply_bc(&mut self.rhs, &self.entity_to_index, &self.face_areas);
    }

    /// Compute face areas and store them
    fn compute_face_areas(&mut self) {
        let geometry = Geometry::new();
        for face in self.mesh.get_faces() {
            let face_id = face.id();
            let face_vertices = self.mesh.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
            };
            let area = geometry.compute_face_area(face_shape, &face_vertices);
            self.face_areas.insert(face_id, area);
        }
    }

    /// Calculate flux coefficient based on face area and distance
    fn calculate_flux_coefficient(&self, face_area: f64, distance: f64) -> f64 {
        let k = 1.0; // Diffusivity
        k * face_area / distance
    }

    /// Solve the linear system using LU decomposition (suitable for small systems)
    fn solve_system(&mut self) -> Vec<f64> {
        // Solve the system Ax = b
        // For small systems, we can use LU decomposition
        let lu = self.matrix.clone().lu();
        let x = lu.solve(&self.rhs).expect("Failed to solve system");
        self.solution = x;
        self.solution.clone().data.as_vec().clone()
    }
}

impl Solver for FVMSolver {
    fn initialize(&mut self) {
        // Any necessary initialization here
    }

    fn assemble_system(&mut self) {
        self.assemble_matrix_rhs();
    }

    fn solve(&mut self) -> Vec<f64> {
        self.solve_system()
    }

    fn get_solution_for_entity(&self, entity: &MeshEntity) -> f64 {
        if let MeshEntity::Cell(cell_id) = entity {
            if let Some(&i) = self.cell_id_to_index.get(cell_id) {
                return self.solution[i];
            }
        }
        0.0
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{DirichletBC, NeumannBC};
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::mesh::Mesh;

    // A helper function to create a simple test mesh
    fn create_test_mesh() -> Mesh {
        // Create a simple 1D mesh with 2 cells and 3 faces (including boundary faces)
        let mut mesh = Mesh::new();

        // Create mesh entities
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let face1 = MeshEntity::Face(1); // Left boundary face
        let face2 = MeshEntity::Face(2); // Internal face between cell1 and cell2
        let face3 = MeshEntity::Face(3); // Right boundary face

        // Add entities to mesh
        mesh.add_entity(cell1);
        mesh.add_entity(cell2);
        mesh.add_entity(face1);
        mesh.add_entity(face2);
        mesh.add_entity(face3);

        // Define relationships using sieve
        // Cell1 is connected to face1 and face2
        mesh.add_relationship(cell1, face1);
        mesh.add_relationship(cell1, face2);

        // Cell2 is connected to face2 and face3
        mesh.add_relationship(cell2, face2);
        mesh.add_relationship(cell2, face3);

        // For faces, define which cells they are connected to
        // Face2 is connected to cell1 and cell2 (internal face)
        mesh.add_relationship(face2, cell1);
        mesh.add_relationship(face2, cell2);

        // Face1 and Face3 are boundary faces connected to their respective cells
        mesh.add_relationship(face1, cell1);
        mesh.add_relationship(face3, cell2);

        // In this simple example, we can assume that face areas and distances are 1.0
        // In practice, these would be calculated based on geometry

        mesh
    }

    // A helper function to create test boundary conditions
    fn create_test_boundary_conditions(mesh: &Mesh) -> (DirichletBC, NeumannBC) {
        let mut dirichlet_bc = DirichletBC::new();
        let mut neumann_bc = NeumannBC::new();

        // Apply Dirichlet BC on left boundary face (face1)
        dirichlet_bc.set_bc(MeshEntity::Face(1), 100.0);  // Dirichlet BC with value 100 at face 1

        // Apply Neumann BC on right boundary face (face3)
        neumann_bc.set_bc(MeshEntity::Face(3), 5.0);      // Neumann BC with flux 5.0 at face 3

        (dirichlet_bc, neumann_bc)
    }

    #[test]
    fn test_fvm_solver_assembly() {
        let mesh = create_test_mesh();
        let (dirichlet_bc, neumann_bc) = create_test_boundary_conditions(&mesh);

        // Create the FVM solver instance
        let mut solver = FVMSolver::new(mesh, dirichlet_bc, neumann_bc);

        solver.assemble_system();

        // Ensure that the matrix and RHS are populated after assembly
        assert!(solver.matrix.nrows() > 0);
        assert!(solver.rhs.len() > 0);
    }

    #[test]
    fn test_fvm_solver_solution() {
        let mesh = create_test_mesh();
        let (dirichlet_bc, neumann_bc) = create_test_boundary_conditions(&mesh);

        // Create the FVM solver instance
        let mut solver = FVMSolver::new(mesh, dirichlet_bc, neumann_bc);

        solver.assemble_system();

        // Solve the system
        let solution = solver.solve();

        // Check that the solution is populated
        assert_eq!(solution.len(), solver.solution.len());

        // Optionally, print the solution for inspection
        println!("Solution: {:?}", solution);
    }
}
