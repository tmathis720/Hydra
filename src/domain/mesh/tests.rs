#[cfg(test)]
mod tests {
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel::unbounded;
    use crate::domain::mesh::Mesh;

    /// Tests that boundary data can be sent from one mesh and received by another.  
    #[test]
    fn test_send_receive_boundary_data() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(2, [4.0, 5.0, 6.0]);
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);

        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        mesh.send_boundary_data();

        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(
            mesh.boundary_data_sender.clone().unwrap(),
            mesh.boundary_data_receiver.clone().unwrap(),
        );

        let _ = mesh_receiver.receive_boundary_data();
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[4.0, 5.0, 6.0]));
    }

    /// Tests sending boundary data without a receiver does not cause a failure.
    #[test]
    fn test_send_without_receiver() {
        let mut mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(3);
        mesh.vertex_coordinates.insert(3, [7.0, 8.0, 9.0]);
        mesh.add_entity(vertex);

        mesh.send_boundary_data();
        assert!(mesh.vertex_coordinates.get(&3).is_some());
    }

    /// Tests the addition of a new entity to the mesh.  
    /// Tests the addition of a new entity to the mesh.  
    /// Verifies that the entity is successfully added to the mesh's entity set.  
    /// Tests the addition of a new entity to the mesh.
    /// Verifies that the entity is successfully added to the mesh's entity set.  
    #[test]
    fn test_add_entity() {
        let mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(1);
        mesh.add_entity(vertex);
        assert!(mesh.entities.read().unwrap().contains(&vertex));
    }

    /// Tests the iterator over the mesh's vertex coordinates.  
    /// Tests the iterator over the mesh's vertex coordinates.  
    /// Verifies that the iterator returns the correct vertex IDs.  
    /// Tests the iterator over the mesh's vertex coordinates.
    /// Verifies that the iterator returns the correct vertex IDs.  
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
    use crate::domain::mesh::hierarchical::MeshNode;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::mesh::Mesh;
    use crossbeam::channel::unbounded;

    /// Full integration test for mesh operations including entity addition,  
    /// boundary data synchronization, and applying constraints at hanging nodes.
    #[test]
    fn test_full_mesh_integration() {
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

        let (sender, receiver) = unbounded();
        mesh.set_boundary_channels(sender, receiver);
        mesh.send_boundary_data();

        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(
            mesh.boundary_data_sender.clone().unwrap(),
            mesh.boundary_data_receiver.clone().unwrap(),
        );
        let _ = mesh_receiver.receive_boundary_data();

        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[0.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[1.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&3), Some(&[0.0, 1.0, 0.0]));

        let mut node = MeshNode::Leaf(cell1);
        node.refine(|&_cell| [
            MeshEntity::Cell(2), MeshEntity::Cell(3), MeshEntity::Cell(4), MeshEntity::Cell(5)
        ]);

        if let MeshNode::Branch { ref children, .. } = node {
            assert_eq!(children.len(), 4);
            assert_eq!(children[0], MeshNode::Leaf(MeshEntity::Cell(2)));
            assert_eq!(children[1], MeshNode::Leaf(MeshEntity::Cell(3)));
            assert_eq!(children[2], MeshNode::Leaf(MeshEntity::Cell(4)));
            assert_eq!(children[3], MeshNode::Leaf(MeshEntity::Cell(5)));
        } else {
            panic!("Expected the node to be refined into a branch.");
        }

        let rcm_order = mesh.rcm_ordering(vertex1);
        assert!(rcm_order.len() > 0);

        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
        ];
        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);
        assert_eq!(parent_dofs, [1.0, 1.5, 2.0, 2.5]);
    }
}
