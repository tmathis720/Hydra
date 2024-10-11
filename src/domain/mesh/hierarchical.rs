use crate::domain::mesh_entity::MeshEntity;
use std::boxed::Box;

/// Represents a hierarchical mesh node, which can either be a leaf (non-refined)
/// or a branch (refined into smaller child elements).
#[derive(Debug, PartialEq)]
pub enum MeshNode<T> {
    Leaf(T),
    Branch {
        data: T,
        children: Box<[MeshNode<T>; 4]>, // 2D quadtree; change to `[MeshNode<T>; 8]` for 3D.
    },
}

impl<T: Clone> MeshNode<T> {
    /// Refines a leaf node into a branch with initialized child nodes.
    pub fn refine<F>(&mut self, init_child_data: F)
    where
        F: Fn(&T) -> [T; 4],  // Function to generate child data from the parent.
    {
        if let MeshNode::Leaf(data) = self {
            let children = init_child_data(data);
            *self = MeshNode::Branch {
                data: data.clone(),
                children: Box::new([
                    MeshNode::Leaf(children[0].clone()),
                    MeshNode::Leaf(children[1].clone()),
                    MeshNode::Leaf(children[2].clone()),
                    MeshNode::Leaf(children[3].clone()),
                ]),
            };
        }
    }

    /// Coarsens a branch back into a leaf node by collapsing children.
    pub fn coarsen(&mut self) {
        if let MeshNode::Branch { data, .. } = self {
            *self = MeshNode::Leaf(data.clone());
        }
    }

    /// Apply constraints at hanging nodes to ensure continuity between parent and child elements.
    pub fn apply_hanging_node_constraints(&self, parent_dofs: &mut [f64], child_dofs: &mut [[f64; 4]; 4]) {
        if let MeshNode::Branch { .. } = self {
            for i in 0..parent_dofs.len() {
                parent_dofs[i] = child_dofs.iter().map(|d| d[i]).sum::<f64>() / 4.0;
            }
        }
    }

    /// An iterator over all leaf nodes in the mesh hierarchy.
    pub fn leaf_iter(&self) -> LeafIterator<T> {
        LeafIterator { stack: vec![self] }
    }
}

/// Iterator for traversing through leaf nodes in the hierarchical mesh.
pub struct LeafIterator<'a, T> {
    stack: Vec<&'a MeshNode<T>>,
}

impl<'a, T> Iterator for LeafIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match node {
                MeshNode::Leaf(data) => return Some(data),
                MeshNode::Branch { children, .. } => {
                    self.stack.extend(&**children);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::hierarchical::MeshNode;

    #[test]
    fn test_refine_leaf() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        if let MeshNode::Branch { children, .. } = node {
            assert_eq!(children[0], MeshNode::Leaf(11));
            assert_eq!(children[1], MeshNode::Leaf(12));
            assert_eq!(children[2], MeshNode::Leaf(13));
            assert_eq!(children[3], MeshNode::Leaf(14));
        } else {
            panic!("Node should have been refined to a branch.");
        }
    }

    #[test]
    fn test_coarsen_branch() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);
        node.coarsen();

        assert_eq!(node, MeshNode::Leaf(10));
    }

    #[test]
    fn test_apply_hanging_node_constraints() {
        let node = MeshNode::Branch {
            data: 0,
            children: Box::new([
                MeshNode::Leaf(1),
                MeshNode::Leaf(2),
                MeshNode::Leaf(3),
                MeshNode::Leaf(4),
            ]),
        };

        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ];

        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);

        assert_eq!(parent_dofs, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_leaf_iterator() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        let leaves: Vec<_> = node.leaf_iter().collect();
        assert_eq!(leaves, [&11, &12, &13, &14]);
    }
}
