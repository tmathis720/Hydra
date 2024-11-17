use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crossbeam::channel::{Sender, Receiver};
use std::time::Duration;

impl Mesh {
    /// Synchronizes the boundary data by first sending the local boundary data
    /// and then receiving any updated boundary data from other sources.
    ///
    /// Ensures boundary data, such as vertex coordinates, is consistent across
    /// all mesh partitions.
    pub fn sync_boundary_data(&mut self) {
        self.send_boundary_data();
        if let Err(e) = self.receive_boundary_data() {
            eprintln!("Error during boundary data synchronization: {:?}", e);
        }
    }

    /// Sets the communication channels for boundary data transmission.
    ///
    /// Configures the sender for transmitting local boundary data and the receiver
    /// for receiving boundary data from other sources.
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
    /// Listens for incoming boundary data and updates local mesh entities and coordinates.
    /// Returns an error if receiving boundary data fails or times out.
    pub fn receive_boundary_data(&mut self) -> Result<(), String> {
        if let Some(ref receiver) = self.boundary_data_receiver {
            match receiver.recv_timeout(Duration::from_millis(500)) {
                Ok(boundary_data) => {
                    let mut entities = self.entities.write().unwrap();
                    for (entity, coords) in boundary_data {
                        if let MeshEntity::Vertex(id) = entity {
                            self.vertex_coordinates.insert(id, coords);
                        }
                        entities.insert(entity);
                    }
                    Ok(())
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    Err("Timeout while waiting for boundary data".to_string())
                }
                Err(e) => {
                    Err(format!("Failed to receive boundary data: {:?}", e))
                }
            }
        } else {
            Err("Boundary data receiver channel is not set".to_string())
        }
    }

    /// Sends the local boundary data through the communication channel to other partitions.
    ///
    /// Collects vertex coordinates for all mesh entities and transmits them using the sender.
    /// Logs an error if the sending operation fails.
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

            if let Err(e) = sender.send(boundary_data) {
                eprintln!("Failed to send boundary data: {:?}", e);
            }
        } else {
            eprintln!("Boundary data sender channel is not set");
        }
    }
}
