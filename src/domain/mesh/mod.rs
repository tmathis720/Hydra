pub mod entities;
pub mod geometry;
pub mod reordering;
pub mod boundary;
pub mod hierarchical;
pub mod topology;
pub mod geometry_validation;
pub mod boundary_validation;
pub mod adjacency_validation;

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crossbeam::channel::{Sender, Receiver};
use lazy_static::lazy_static;
use log::{info, warn, error}; // Using `log` crate for structured logging.
use thiserror::Error; // Using `thiserror` crate for robust error handling.

// Delegate methods to corresponding modules

/// Represents errors that can occur in the `Mesh` module.
#[derive(Debug, Error)]
pub enum MeshError {
    #[error("Entity {0} already exists in the mesh.")]
    EntityExists(String),

    #[error("Entity {0} not found in the mesh.")]
    EntityNotFound(String),

    #[error("Failed to add relationship: {0}")]
    RelationshipError(String),

    #[error("Boundary synchronization error: {0}")]
    BoundarySyncError(String),

    #[error("Geometry calculation error: {0}")]
    GeometryError(String),

    #[error("Topology validation error: {0}")]
    TopologyError(String),

    #[error("Invalid entity type: {0}")]
    InvalidEntityType(String),

    #[error("Connectivity error: Face {0} error {1}")]
    ConnectivityError(String, String),

    #[error("Failed to retrieve supporting entities for {0}")]
    ConnectivityQueryError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),

    #[error("No neighboring vertices found for vertex {0}")]
    NoNeighborsError(String),
}

/// A trait for structured logging in the `Mesh` module.
pub trait MeshLogger: std::fmt::Debug + Send + Sync {
    fn log_info(&self, message: &str);
    fn log_warn(&self, message: &str);
    fn log_error(&self, error: &MeshError);
}

/// Default implementation of `MeshLogger` using the `log` crate.
#[derive(Debug)]
pub struct DefaultMeshLogger;

impl DefaultMeshLogger {
    /// Constructor for DefaultMeshLogger
    pub fn new() -> Self {
        DefaultMeshLogger
    }
}

impl MeshLogger for DefaultMeshLogger {
    fn log_info(&self, message: &str) {
        info!("{}", message);
    }

    fn log_warn(&self, message: &str) {
        warn!("{}", message);
    }

    fn log_error(&self, error: &MeshError) {
        error!("{:?}", error);
    }
}

/// Represents the mesh structure, which is composed of a sieve for entity management,  
/// a set of mesh entities, vertex coordinates, and channels for boundary data.  
#[derive(Clone, Debug)]
pub struct Mesh {
    /// The sieve structure used for organizing the mesh entities' relationships.  
    pub sieve: Arc<Sieve>,  
    /// A thread-safe, read-write lock for managing mesh entities.  
    /// This set contains all `MeshEntity` objects in the mesh.  
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,  
    /// A map from vertex indices to their 3D coordinates.  
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,  
    /// An optional channel sender for transmitting boundary data related to mesh entities.  
    pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,  
    /// An optional channel receiver for receiving boundary data related to mesh entities.  
    pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,  
    /// The logger used for logging events and errors in the mesh.
    pub logger: Arc<dyn MeshLogger>,
}

lazy_static! {
    static ref GLOBAL_MESH: Arc<RwLock<Mesh>> = Arc::new(RwLock::new(Mesh::new()));
}

impl Mesh {
    /// Creates a new instance of the `Mesh` struct with initialized components.  
    /// 
    /// This method sets up the sieve, entity set, vertex coordinate map,  
    /// and a channel for boundary data communication between mesh components.  
    ///
    /// The `Sender` and `Receiver` are unbounded channels used to pass boundary  
    /// data between mesh modules asynchronously.
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam::channel::unbounded();
        Mesh {
            sieve: Arc::new(Sieve::new()),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender),
            boundary_data_receiver: Some(receiver),
            logger: Arc::new(DefaultMeshLogger::new()), // Corrected instantiation
        }
    }

    pub fn global() -> Arc<RwLock<Mesh>> {
        GLOBAL_MESH.clone()
    }

    /// Logs an error when an operation fails.
    pub fn handle_error(&self, error: MeshError) {
        self.logger.log_error(&error);
    }
}

#[cfg(test)]
pub mod tests;
