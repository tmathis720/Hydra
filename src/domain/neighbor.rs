use std::collections::HashMap;

/// Build neighbor relationships based on shared edges
pub fn build_neighbors(mesh: &mut Mesh) {
    let mut edge_to_element: HashMap<(usize,usize), Vec<usize>> = HashMap::new();

    // Populate the map with edges and the elements that share them
    for element in &mesh.elements {
        let node_ids = element.node_ids;
        let edges = [
            (node_ids[0], node_ids[1]),
            (node_ids[1], node_ids[2]),
            (node_ids[2], node_ids[0]),
        ];

        for &(n1, n2) in &edges {
            let edge = (n1.min(n2), n1.max(n2)); // Store edges as undirected
            edge_to_element.entry(edge).or_default().push(element.id);

        }
    }

    // Assign neighbors based on shared edges
    for element in &mut mesh.elements {
        let node_ids = element.node_ids;
        let edges = [
            (node_ids[0], node_ids[1]),
            (node_ids[1], node_ids[2]),
            (node_ids[2], node_ids[0]),
        ];

        for &(n1, n2) in &edges {
            let edge = (n1.min(n2), n1.max(n2));
            if let Some(neighboring_elements) = edge_to_element.get(&edge) {
                for &neighbor_id in neighboring_elements {
                    if neighbor_id != element.id {
                        element.neighbors.push(neighbor_id);
                    }
                }
            }
        }
    }
}