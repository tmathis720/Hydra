use crate::domain::Element;

pub struct PeriodicBoundary {
    pub elements: Vec<Element>,
}

impl PeriodicBoundary {
    pub fn apply_boundary(&self, elements: &mut Vec<Element>) {
        if let Some(first_element) = elements.first().cloned() {
            if let Some(last_element) = elements.last_mut() {
                *last_element = first_element;  // Now safe since first_element is cloned
            }
        }
    }
}
