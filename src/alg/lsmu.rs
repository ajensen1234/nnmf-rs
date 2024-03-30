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

    println!("Orig W {}", w);
    let num_iterations = 1000;
    for _ in 0..num_iterations {
        let b = w.transpose() * w.clone() * h.clone();
        let c = w.transpose() * matrix_to_factorize.clone();
        let d = w.clone() * h.clone() * h.transpose();
        let e = matrix_to_factorize.clone() * h.transpose();

        // Can parallelize these implementations
        h.component_mul_assign(&c);
        h.component_div_assign(&b);
        w.component_mul_assign(&e);
        w.component_div_assign(&d);
    }
    return (w, h);
}
