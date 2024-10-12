use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crossbeam::channel::{Sender, Receiver};
use std::sync::RwLock;

impl Mesh {
    /// Synchronizes the boundary data by first sending the local boundary data  
    /// and then receiving any updated boundary data from other sources.  
    ///
    /// This function ensures that boundary data, such as vertex coordinates,  
    /// is consistent across all mesh partitions.  
    ///
    /// Example usage:
    /// 
    ///    mesh.sync_boundary_data();  
    ///
    pub fn sync_boundary_data(&mut self) {
        self.send_boundary_data();
        self.receive_boundary_data();
    }

    /// Sets the communication channels for boundary data transmission.  
    ///
    /// The sender channel is used to transmit the local boundary data, and  
    /// the receiver channel is used to receive boundary data from other  
    /// partitions or sources.  
    ///
    /// Example usage:
    /// 
    ///    mesh.set_boundary_channels(sender, receiver);  
    ///
    pub fn set_boundary_channels(
        &mut self,
        sender: Sender<FxHashMap<MeshEntity, [f64; 3]>>,
        receiver: Receiver<FxHashMap<MeshEntity, [f64; 3]>>,
    ) {
        self.boundary_data_sender = Some(sender);
        self.boundary_data_receiver = Some(receiver);
    }

    /// Receives boundary data from the communication channel and updates the mesh.  
    ///
    /// This method listens for incoming boundary data (such as vertex coordinates)  
    /// from the receiver channel and updates the local mesh entities and coordinates.  
    ///
    /// Example usage:
    /// 
    ///    mesh.receive_boundary_data();  
    ///
    pub fn receive_boundary_data(&mut self) {
        if let Some(ref receiver) = self.boundary_data_receiver {
            if let Ok(boundary_data) = receiver.recv() {
                let mut entities = self.entities.write().unwrap();
                for (entity, coords) in boundary_data {
                    // Update vertex coordinates if the entity is a vertex.
                    if let MeshEntity::Vertex(id) = entity {
                        self.vertex_coordinates.insert(id, coords);
                    }
                    entities.insert(entity);
                }
            }
        }
    }

    /// Sends the local boundary data (such as vertex coordinates) through  
    /// the communication channel to other partitions or sources.  
    ///
    /// This method collects the vertex coordinates for all mesh entities  
    /// and sends them using the sender channel.  
    ///
    /// Example usage:
    /// 
    ///    mesh.send_boundary_data();  
    ///
    pub fn send_boundary_data(&self) {
        if let Some(ref sender) = self.boundary_data_sender {
            let mut boundary_data = FxHashMap::default();
            let entities = self.entities.read().unwrap();
            for entity in entities.iter() {
                if let MeshEntity::Vertex(id) = entity {
                    if let Some(coords) = self.vertex_coordinates.get(id) {
                        boundary_data.insert(*entity, *coords);
                    }
                }
            }

            // Send the boundary data through the sender channel.
            if let Err(e) = sender.send(boundary_data) {
                eprintln!("Failed to send boundary data: {:?}", e);
            }
        }
    }
}
