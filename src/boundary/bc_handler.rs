use rustc_hash::FxHashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::section::Section;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use faer::MatMut;

pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// BoundaryCondition represents various types of boundary conditions
/// that can be applied to mesh entities. These include:
/// * Dirichlet: A constant boundary condition where a specific value is enforced.
/// * Neumann: A flux or derivative boundary condition.
/// * Robin: A linear combination of Dirichlet and Neumann boundary conditions.
/// * DirichletFn: A time-dependent Dirichlet boundary condition using a function.
/// * NeumannFn: A time-dependent Neumann boundary condition using a function.
/// Example usage:
/// 
///    let bc_handler = BoundaryConditionHandler::new();  
///    let entity = MeshEntity::Vertex(1);  
///    let condition = BoundaryCondition::Dirichlet(10.0);  
///    bc_handler.set_bc(entity, condition);  
/// 

#[derive(Clone)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    DirichletFn(BoundaryConditionFn),
    NeumannFn(BoundaryConditionFn),
}

/// The BoundaryConditionHandler struct is responsible for managing
/// boundary conditions associated with specific mesh entities. 
/// It supports adding, retrieving, and applying boundary conditions 
/// for a given set of mesh entities and interacting with the linear system 
/// matrices and right-hand side vectors.
/// Example usage:
/// 
///    let handler = BoundaryConditionHandler::new();  
///    let entity = MeshEntity::Edge(5);  
///    let bc = BoundaryCondition::Neumann(5.0);  
///    handler.set_bc(entity, bc);  
/// 

pub struct BoundaryConditionHandler {
    conditions: Section<BoundaryCondition>,
}

impl BoundaryConditionHandler {
    /// Creates a new BoundaryConditionHandler with an empty section 
    /// to store boundary conditions.
    /// Example usage:
    /// 
    ///    let handler = BoundaryConditionHandler::new();  
    /// 

    pub fn new() -> Self {
        Self {
            conditions: Section::new(),
        }
    }

    /// Sets a boundary condition for a specific mesh entity.
    /// The boundary condition can be Dirichlet, Neumann, Robin, or a functional form.
    /// 
    /// # Arguments:
    /// * `entity` - The mesh entity (such as a vertex, edge, or face).
    /// * `condition` - The boundary condition to apply to the entity.
    ///
    /// Example usage:
    /// 
    ///    let entity = MeshEntity::Vertex(3);  
    ///    let bc = BoundaryCondition::Dirichlet(1.0);  
    ///    handler.set_bc(entity, bc);  
    /// 
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.set_data(entity, condition);
    }

    /// Retrieves the boundary condition applied to a specific mesh entity, if it exists.
    /// 
    /// # Arguments:
    /// * `entity` - A reference to the mesh entity.
    ///
    /// Returns an `Option<BoundaryCondition>` indicating the boundary condition 
    /// if one has been set for the entity.
    ///
    /// Example usage:
    /// 
    ///    let entity = MeshEntity::Edge(2);  
    ///    if let Some(bc) = handler.get_bc(&entity) {  
    ///        println!("Boundary condition found!");  
    ///    }  
    /// 
    pub fn get_bc(&self, entity: &MeshEntity) -> Option<BoundaryCondition> {
        self.conditions.restrict(entity)
    }

    /// Applies the boundary conditions to the system matrices and right-hand side vectors.
    /// This modifies the matrix and the right-hand side based on the type of boundary 
    /// conditions for the provided boundary entities.
    ///
    /// # Arguments:
    /// * `matrix` - The mutable matrix (MatMut) to modify.
    /// * `rhs` - The mutable right-hand side vector (MatMut).
    /// * `boundary_entities` - A list of mesh entities representing the boundary.
    /// * `entity_to_index` - A hash map that associates each mesh entity with an index 
    ///   in the system.
    /// * `time` - The current time for time-dependent boundary conditions.
    ///
    /// Example usage:
    /// 
    ///    let mut matrix = MatMut::new();  
    ///    let mut rhs = MatMut::new();  
    ///    let boundary_entities = vec![MeshEntity::Vertex(1), MeshEntity::Edge(2)];  
    ///    let entity_to_index: FxHashMap<MeshEntity, usize> = FxHashMap::default();  
    ///    handler.apply_bc(&mut matrix, &mut rhs, &boundary_entities, &entity_to_index, 0.0);  
    /// 
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        boundary_entities: &[MeshEntity],
        entity_to_index: &FxHashMap<MeshEntity, usize>,
        time: f64,
    ) {
        for entity in boundary_entities {
            if let Some(bc) = self.get_bc(entity) {
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(
                            matrix,
                            rhs,
                            *entity_to_index.get(entity).unwrap(),
                            value,
                        );
                    }
                    BoundaryCondition::Neumann(flux) => {
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(
                            rhs,
                            *entity_to_index.get(entity).unwrap(),
                            flux,
                        );
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let robin_bc = RobinBC::new();
                        robin_bc.apply_robin(
                            matrix,
                            rhs,
                            *entity_to_index.get(entity).unwrap(),
                            alpha,
                            beta,
                        );
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(
                            matrix,
                            rhs,
                            *entity_to_index.get(entity).unwrap(),
                            value,
                        );
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(
                            rhs,
                            *entity_to_index.get(entity).unwrap(),
                            value,
                        );
                    }
                }
            }
        }
    }
}

/// The BoundaryConditionApply trait defines the `apply` method, which is used to apply 
/// a boundary condition to a given mesh entity. It modifies the matrix and right-hand 
/// side (rhs) of the system based on the boundary condition type (Dirichlet, Neumann, Robin, etc.).
/// 
/// The `apply` method should be implemented by any type that represents boundary conditions 
/// and needs to be applied to the system matrix and rhs.
///
/// Example usage:
/// 
///    let bc = BoundaryCondition::Dirichlet(1.0);  
///    let entity = MeshEntity::Vertex(1);  
///    bc.apply(&entity, &mut rhs, &mut matrix, &entity_to_index, 0.0);  
/// 
pub trait BoundaryConditionApply {
    /// Applies a boundary condition to a specific mesh entity, modifying the system matrix and rhs.
    ///
    /// # Arguments:
    /// * `entity` - The mesh entity to which the boundary condition is applied.
    /// * `rhs` - The mutable right-hand side vector that may be modified.
    /// * `matrix` - The mutable system matrix that may be modified.
    /// * `entity_to_index` - A hash map that associates each mesh entity with its index in the system.
    /// * `time` - The current time for applying time-dependent boundary conditions.
    ///
    /// Example usage:
    /// 
    ///    let bc = BoundaryCondition::Neumann(5.0);  
    ///    let entity = MeshEntity::Edge(3);  
    ///    bc.apply(&entity, &mut rhs, &mut matrix, &entity_to_index, 0.0);  
    /// 
    fn apply(&self, entity: &MeshEntity, rhs: &mut MatMut<f64>, matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64);
}

/// The implementation of BoundaryConditionApply for BoundaryCondition allows different types
/// of boundary conditions (Dirichlet, Neumann, Robin, and their functional forms) to modify 
/// the matrix and rhs according to the specific logic of each boundary condition type.
/// 
/// Example usage:
/// 
///    let bc = BoundaryCondition::DirichletFn(Arc::new(|time, coords| time + coords[0]));  
///    let entity = MeshEntity::Vertex(2);  
///    bc.apply(&entity, &mut rhs, &mut matrix, &entity_to_index, 0.5);  
/// 
impl BoundaryConditionApply for BoundaryCondition {
    /// Applies the boundary condition to a specific mesh entity.
    ///
    /// # Arguments:
    /// * `entity` - The mesh entity to which the boundary condition is applied.
    /// * `rhs` - The mutable right-hand side vector.
    /// * `matrix` - The mutable system matrix.
    /// * `entity_to_index` - A hash map mapping each mesh entity to its index in the system.
    /// * `time` - The current time for time-dependent boundary conditions.
    fn apply(&self, entity: &MeshEntity, rhs: &mut MatMut<f64>, matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
        match self {
            BoundaryCondition::Dirichlet(value) => {
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, *entity_to_index.get(entity).unwrap(), *value);
            }
            BoundaryCondition::Neumann(flux) => {
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, *entity_to_index.get(entity).unwrap(), *flux);
            }
            BoundaryCondition::Robin { alpha: _, beta: _ } => {
                // Robin boundary condition logic to be implemented here
            }
            BoundaryCondition::DirichletFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];  // Placeholder for mesh entity coordinates
                let value = fn_bc(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, *entity_to_index.get(entity).unwrap(), value);
            }
            BoundaryCondition::NeumannFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];  // Placeholder for mesh entity coordinates
                let value = fn_bc(time, &coords);
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, *entity_to_index.get(entity).unwrap(), value);
            }
        }
    }
}
