use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crossbeam::channel::{Sender, Receiver};
use std::time::Duration;

impl Mesh {
    /// Synchronizes the boundary data across mesh partitions.
    ///
    /// This method ensures that the boundary data (such as vertex coordinates)
    /// is consistent by performing a two-step process:
    /// 1. Sending the local boundary data to other mesh partitions.
    /// 2. Receiving updated boundary data from other sources and integrating it
    ///    into the local mesh.
    ///
    /// Logs an error if the receiving step encounters an issue, such as a timeout
    /// or a communication error.
    pub fn sync_boundary_data(&mut self) {
        self.send_boundary_data(); // Send local boundary data to other partitions
        if let Err(e) = self.receive_boundary_data() {
            // Handle any errors that occur during data reception
            eprintln!("Error during boundary data synchronization: {:?}", e);
        }
    }

    /// Configures the communication channels for boundary data exchange.
    ///
    /// This method sets up the sender and receiver channels used for transmitting
    /// and receiving boundary data. The sender is responsible for sending local
    /// boundary data to other mesh partitions, while the receiver listens for incoming
    /// boundary data updates.
    ///
    /// # Arguments
    /// - `sender`: A channel used to send boundary data (e.g., vertex coordinates).
    /// - `receiver`: A channel used to receive boundary data updates.
    pub fn set_boundary_channels(
        &mut self,
        sender: Sender<FxHashMap<MeshEntity, [f64; 3]>>,
        receiver: Receiver<FxHashMap<MeshEntity, [f64; 3]>>,
    ) {
        self.boundary_data_sender = Some(sender); // Assign sender for transmitting data
        self.boundary_data_receiver = Some(receiver); // Assign receiver for incoming data
    }

    /// Receives boundary data updates and integrates them into the mesh.
    ///
    /// This method listens for incoming boundary data using the configured receiver
    /// channel. It updates local entities and vertex coordinates based on the received
    /// data. If the reception times out or encounters an error, it returns an appropriate
    /// error message.
    ///
    /// # Returns
    /// - `Ok(())` if boundary data is successfully received and integrated.
    /// - `Err(String)` if the operation fails due to timeout or communication errors.
    pub fn receive_boundary_data(&mut self) -> Result<(), String> {
        if let Some(ref receiver) = self.boundary_data_receiver {
            match receiver.recv_timeout(Duration::from_millis(500)) {
                Ok(boundary_data) => {
                    // Lock the entities set to modify mesh data
                    let mut entities = self.entities.write().unwrap();

                    // Update vertex coordinates and entities from the received data
                    for (entity, coords) in boundary_data {
                        if let MeshEntity::Vertex(id) = entity {
                            self.vertex_coordinates.insert(id, coords);
                        }
                        entities.insert(entity);
                    }
                    Ok(())
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    // Return an error if reception times out
                    Err("Timeout while waiting for boundary data".to_string())
                }
                Err(e) => {
                    // Handle other communication errors
                    Err(format!("Failed to receive boundary data: {:?}", e))
                }
            }
        } else {
            // Return an error if the receiver channel is not set
            Err("Boundary data receiver channel is not set".to_string())
        }
    }

    /// Sends the local boundary data to other mesh partitions.
    ///
    /// This method collects vertex coordinates for all vertices in the mesh and
    /// transmits them using the configured sender channel. If the sender channel
    /// is not set or the operation fails, an error message is logged.
    pub fn send_boundary_data(&self) {
        if let Some(ref sender) = self.boundary_data_sender {
            let mut boundary_data = FxHashMap::default(); // Initialize data map
            let entities = self.entities.read().unwrap(); // Access the entities set

            // Collect vertex coordinates for all vertices in the mesh
            for entity in entities.iter() {
                if let MeshEntity::Vertex(id) = entity {
                    if let Some(coords) = self.vertex_coordinates.get(id) {
                        boundary_data.insert(*entity, *coords); // Add data to the map
                    }
                }
            }

            // Attempt to send the boundary data through the channel
            if let Err(e) = sender.send(boundary_data) {
                eprintln!("Failed to send boundary data: {:?}", e);
            }
        } else {
            // Log an error if the sender channel is not configured
            eprintln!("Boundary data sender channel is not set");
        }
    }
}
