use std::boxed::Box;

/// Represents a hierarchical mesh node, which can either be a leaf (non-refined)  
/// or a branch (refined into smaller child elements).  
/// 
/// In 2D, each branch contains 4 child elements (quadtree), while in 3D, each branch  
/// would contain 8 child elements (octree).
///
/// Example usage:
/// 
///    let mut node = MeshNode::Leaf(10);  
///    node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);  
///    assert!(matches!(node, MeshNode::Branch { .. }));  
/// 
#[derive(Debug, PartialEq)]
pub enum MeshNode<T> {
    /// A leaf node representing an unrefined element containing data of type `T`.  
    Leaf(T),
    
    /// A branch node representing a refined element with child elements.  
    /// The branch contains its own data and an array of 4 child nodes (for 2D).  
    Branch {
        data: T,
        children: Box<[MeshNode<T>; 4]>,  // 2D quadtree; change to `[MeshNode<T>; 8]` for 3D.
    },
}

impl<T: Clone> MeshNode<T> {
    /// Refines a leaf node into a branch with initialized child nodes.  
    ///
    /// The `init_child_data` function is used to generate the data for each child  
    /// element based on the parent node's data.  
    ///
    /// Example usage:
    /// 
    ///    let mut node = MeshNode::Leaf(10);  
    ///    node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);  
    ///
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

    /// Coarsens a branch back into a leaf node by collapsing its child elements.  
    ///
    /// This method turns a branch back into a leaf node, retaining the data of the  
    /// parent node but removing its child elements.  
    ///
    /// Example usage:
    /// 
    ///    node.coarsen();  
    ///
    pub fn coarsen(&mut self) {
        if let MeshNode::Branch { data, .. } = self {
            *self = MeshNode::Leaf(data.clone());
        }
    }

    /// Applies constraints at hanging nodes to ensure continuity between the parent  
    /// and its child elements.  
    ///
    /// This function adjusts the degrees of freedom (DOFs) at the parent node by  
    /// averaging the DOFs from its child elements.  
    ///
    /// Example usage:
    /// 
    ///    node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);  
    ///
    pub fn apply_hanging_node_constraints(&self, parent_dofs: &mut [f64], child_dofs: &mut [[f64; 4]; 4]) {
        if let MeshNode::Branch { .. } = self {
            for i in 0..parent_dofs.len() {
                parent_dofs[i] = child_dofs.iter().map(|d| d[i]).sum::<f64>() / 4.0;
            }
        }
    }

    /// Returns an iterator over all leaf nodes in the mesh hierarchy.  
    ///
    /// This iterator allows traversal of the entire hierarchical mesh,  
    /// returning only the leaf nodes.  
    ///
    /// Example usage:
    /// 
    ///    let leaves: Vec<_> = node.leaf_iter().collect();  
    ///
    pub fn leaf_iter(&self) -> LeafIterator<T> {
        LeafIterator { stack: vec![self] }
    }
}

/// An iterator for traversing through leaf nodes in the hierarchical mesh.  
/// 
/// This iterator traverses all nodes in the hierarchy but only returns  
/// the leaf nodes.
pub struct LeafIterator<'a, T> {
    stack: Vec<&'a MeshNode<T>>,
}

impl<'a, T> Iterator for LeafIterator<'a, T> {
    type Item = &'a T;

    /// Returns the next leaf node in the traversal.  
    /// If the current node is a branch, its children are pushed onto the stack  
    /// for traversal in depth-first order.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match node {
                MeshNode::Leaf(data) => return Some(data),
                MeshNode::Branch { children, .. } => {
                    // Push children onto the stack in reverse order to get them in the desired order.
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
    use crate::domain::mesh::hierarchical::MeshNode;

    #[test]
    /// Test that verifies refining a leaf node into a branch works as expected.  
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
    /// Test that verifies coarsening a branch node back into a leaf works as expected.  
    fn test_coarsen_branch() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);
        node.coarsen();

        assert_eq!(node, MeshNode::Leaf(10));
    }

    #[test]
    /// Test that verifies applying hanging node constraints works correctly by  
    /// averaging the degrees of freedom from the child elements to the parent element.  
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
    /// Test that verifies the leaf iterator correctly traverses all leaf nodes in  
    /// the mesh hierarchy and returns them in the expected order.  
    fn test_leaf_iterator() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        let leaves: Vec<_> = node.leaf_iter().collect();
        assert_eq!(leaves, [&11, &12, &13, &14]);
    }
}
