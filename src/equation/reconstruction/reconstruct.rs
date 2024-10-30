pub fn reconstruct_face_values(
    cell_value: f64,
    gradient: Gradient,
    cell_center: Point,
    face_center: Point,
) -> f64 {
    cell_value + gradient.dot(&(face_center - cell_center))
}
