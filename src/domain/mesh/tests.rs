#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel::unbounded;
    use crate::domain::mesh::Mesh;

    #[test]
    fn test_send_receive_boundary_data() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        // Set up vertex coordinates.
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(2, [4.0, 5.0, 6.0]);

        // Add boundary entities.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);

        // Set up a separate sender and receiver for testing.
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate sending the boundary data.
        mesh.send_boundary_data();

        // Create a second mesh instance to simulate the receiver.
        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(mesh.boundary_data_sender.clone().unwrap(), mesh.boundary_data_receiver.clone().unwrap());

        // Simulate receiving the boundary data.
        mesh_receiver.receive_boundary_data();

        // Verify that the receiver mesh has the updated vertex coordinates.
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[4.0, 5.0, 6.0]));
    }

    /* #[test]
    fn test_receive_empty_data() {
        let mut mesh = Mesh::new();
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate receiving without sending any data.
        mesh.receive_boundary_data();

        // Ensure no data has been added.
        assert!(mesh.vertex_coordinates.is_empty());
    } */

    #[test]
    fn test_send_without_receiver() {
        let mut mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(3);
        mesh.vertex_coordinates.insert(3, [7.0, 8.0, 9.0]);
        mesh.add_entity(vertex);

        // Simulate sending the boundary data without setting a receiver.
        mesh.send_boundary_data();

        // No receiver to process, but this should not panic or fail.
        assert!(mesh.vertex_coordinates.get(&3).is_some());
    }

    #[test]
    fn test_add_entity() {
        let mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(1);
        mesh.add_entity(vertex);
        assert!(mesh.entities.read().unwrap().contains(&vertex));
    }

    #[test]
    fn test_iter_vertices() {
        let mut mesh = Mesh::new();
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        let vertices: Vec<_> = mesh.iter_vertices().collect();
        assert_eq!(vertices, vec![&1]);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::domain::mesh::hierarchical::MeshNode;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::mesh::Mesh;
    use crossbeam::channel::unbounded;
    use rustc_hash::FxHashMap;

    #[test]
    fn test_full_mesh_integration() {
        // Step 1: Create a new mesh and add entities (vertices, edges, cells)
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let cell1 = MeshEntity::Cell(1);

        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(cell1);
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);

        // Step 2: Set up and sync boundary data.
        let (sender, receiver) = unbounded();
        mesh.set_boundary_channels(sender, receiver);
        mesh.send_boundary_data();

        // Create another mesh instance to simulate receiving boundary data.
        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(mesh.boundary_data_sender.clone().unwrap(), mesh.boundary_data_receiver.clone().unwrap());
        mesh_receiver.receive_boundary_data();

        // Verify that the receiver mesh has the correct boundary data.
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[0.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[1.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&3), Some(&[0.0, 1.0, 0.0]));

        // Step 3: Refine a hierarchical mesh node.
        let mut node = MeshNode::Leaf(cell1);
        node.refine(|&cell| [
            MeshEntity::Cell(2), MeshEntity::Cell(3), MeshEntity::Cell(4), MeshEntity::Cell(5)
        ]);

        // Verify that the node has been refined into a branch.
        if let MeshNode::Branch { ref children, .. } = node {
            assert_eq!(children.len(), 4);
            assert_eq!(children[0], MeshNode::Leaf(MeshEntity::Cell(2)));
            assert_eq!(children[1], MeshNode::Leaf(MeshEntity::Cell(3)));
            assert_eq!(children[2], MeshNode::Leaf(MeshEntity::Cell(4)));
            assert_eq!(children[3], MeshNode::Leaf(MeshEntity::Cell(5)));
        } else {
            panic!("Expected the node to be refined into a branch.");
        }

        // Step 4: Apply RCM ordering to the mesh and verify order.
        let rcm_order = mesh.rcm_ordering(vertex1);
        assert!(rcm_order.len() > 0); // RCM ordering should produce a non-empty order.

        // Step 5: Apply constraints at the hanging nodes after refinement.
        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
        ];
        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);

        // Verify that the hanging node constraints were applied correctly.
        assert_eq!(parent_dofs, [1.0, 1.5, 2.0, 2.5]);
    }
}

