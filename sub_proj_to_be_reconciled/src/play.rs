use na::{U2, U3, Dynamic, ArrayStorage, VecStorage, SMatrix};

// Statically sized and statically allocated 2x3 matrix using 32-bit floats.

fn main() {
    type Matrix2x3f = SMatrix<f32, 2, 3>;
    let a = Matrix2x3f::zero();
}