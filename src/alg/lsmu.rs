use nalgebra::{DMatrix, QR, DVector,DMatrixSlice, Scalar};
use ndarray::{ArrayView2, ArrayView1, arr2, Array1};

pub fn lee_seung_multiplicative_update_rule(
    matrix_to_factorize: DMatrix<f64>,
    num_synergies: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let m = matrix_to_factorize.nrows();
    let n = matrix_to_factorize.ncols();

    let mut w = DMatrix::<f64>::new_random(m, num_synergies).abs();
    let mut h = DMatrix::<f64>::new_random(num_synergies, n).abs();

    let num_iterations = 1000000;
    for _ in 0..num_iterations {
        // Can parallelize each of these
        let b = w.transpose() * w.clone() * h.clone();
        let c = w.transpose() * matrix_to_factorize.clone();
        let d = w.clone() * h.clone() * h.transpose();
        let e = matrix_to_factorize.clone() * h.transpose();

        // Can parallelize these implementations
        let h = hadamard_division(&h.component_mul(&c), &b);
        let w = hadamard_division(&w.component_mul(&e), &d);
    }
    return (w, h);
}

fn hadamard_division(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a.zip_map(b, |a_elem, b_elem| a_elem / b_elem)
}