use std::boxed::Box;

/// Represents a hierarchical mesh node, which can either be a leaf (non-refined)
/// or a branch (refined into smaller child elements).
///
/// - **Leaf**: Represents an unrefined mesh node containing data of type `T`.
/// - **Branch**: Represents a refined mesh node with child elements and its own data.
///
/// In 2D, each branch contains 4 child elements (quadtree), and in 3D, each branch
/// would contain 8 child elements (octree).
#[derive(Debug, PartialEq)]
pub enum MeshNode<T> {
    /// A leaf node representing an unrefined element with data of type `T`.
    Leaf(T),

    /// A branch node representing a refined element with child elements and data.
    /// In 2D, the branch contains 4 child nodes.
    Branch {
        /// Data associated with the branch.
        data: T,
        /// Child nodes of the branch stored as a boxed array.
        children: Box<[MeshNode<T>; 4]>, // For 3D, use `[MeshNode<T>; 8]`.
    },
}

impl<T: Clone> MeshNode<T> {
    /// Refines a leaf node into a branch with initialized child nodes.
    ///
    /// This method converts a `MeshNode::Leaf` into a `MeshNode::Branch` using
    /// the provided closure to generate data for the child nodes based on the
    /// current leaf's data.
    ///
    /// # Arguments
    /// * `init_child_data` - A closure that generates an array of data for the
    ///   child nodes based on the parent node's data.
    pub fn refine<F>(&mut self, init_child_data: F)
    where
        F: Fn(&T) -> [T; 4],
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

    /// Coarsens a branch node back into a leaf node by collapsing its child elements.
    ///
    /// This method converts a `MeshNode::Branch` into a `MeshNode::Leaf`, retaining
    /// the parent node's data and removing all child elements.
    pub fn coarsen(&mut self) {
        if let MeshNode::Branch { data, .. } = self {
            *self = MeshNode::Leaf(data.clone());
        }
    }

    /// Applies hanging node constraints to ensure continuity between parent and child elements.
    ///
    /// This function averages the degrees of freedom (DOFs) from the child nodes and
    /// assigns the result to the parent node.
    ///
    /// # Arguments
    /// * `parent_dofs` - A mutable slice representing the DOFs of the parent node.
    /// * `child_dofs` - A mutable array of slices representing the DOFs of each child node.
    pub fn apply_hanging_node_constraints(&self, parent_dofs: &mut [f64], child_dofs: &mut [[f64; 4]; 4]) {
        if let MeshNode::Branch { .. } = self {
            for i in 0..parent_dofs.len() {
                parent_dofs[i] = child_dofs.iter().map(|d| d[i]).sum::<f64>() / 4.0;
            }
        }
    }

    /// Returns an iterator over all leaf nodes in the hierarchical mesh.
    ///
    /// This iterator traverses the entire mesh hierarchy in a depth-first manner
    /// and yields references to the data of all leaf nodes.
    ///
    /// # Returns
    /// * `LeafIterator<T>` - An iterator that yields references to leaf node data.
    pub fn leaf_iter(&self) -> LeafIterator<T> {
        LeafIterator { stack: vec![self] }
    }
}

/// An iterator for traversing through leaf nodes in a hierarchical mesh.
///
/// This iterator traverses all nodes in the hierarchy but only yields
/// the data from leaf nodes.
pub struct LeafIterator<'a, T> {
    /// A stack used for depth-first traversal of the mesh hierarchy.
    stack: Vec<&'a MeshNode<T>>,
}

impl<'a, T> Iterator for LeafIterator<'a, T> {
    type Item = &'a T;

    /// Advances the iterator to the next leaf node, if any.
    ///
    /// If the current node is a branch, its children are added to the stack
    /// for traversal in depth-first order. Only leaf nodes yield data.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match node {
                MeshNode::Leaf(data) => return Some(data),
                MeshNode::Branch { children, .. } => {
                    // Push children onto the stack in reverse order for correct order.
                    for child in children.iter().rev() {
                        self.stack.push(child);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
