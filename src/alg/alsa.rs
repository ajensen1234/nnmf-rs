use nalgebra::{DMatrix, QR};
pub fn alternating_leastsq_nnmf(
    matrix_to_factorize: DMatrix<f64>,
    num_synergies: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let m = matrix_to_factorize.nrows();
    let n = matrix_to_factorize.ncols();

    // Old method of initializing via identity matrices
    // let mut w = DMatrix::<f64>::identity(m, num_synergies);
    // let mut h = DMatrix::<f64>::identity(num_synergies, n).abs();

    // New method of initializing by random matrices.
    // Need to add `rand` feature into nalgebra to enable this function
    let mut w = DMatrix::<f64>::new_random(m, num_synergies).abs();
    let mut h = DMatrix::<f64>::new_random(num_synergies, n).abs();

    let num_iterations = 250;
    let epsilon = 1e-14;

    for _ in 0..num_iterations {
        let qr_w = QR::new(w.clone());
        let b_w = qr_w.q().transpose() * matrix_to_factorize.clone();
        let r_w = qr_w.r();

        for i in 0..n {
            let new_lst_sqr_col = r_w.clone().pseudo_inverse(epsilon).unwrap() * b_w.column(i);

            h.set_column(i, &new_lst_sqr_col);
        }

        let qr_h = QR::new(h.transpose());

        let b_h = (qr_h.q().transpose()) * (matrix_to_factorize.transpose());

        let r_h = qr_h.r();

        for j in 0..m {
            let new_lst_sqr_row =
                (r_h.clone().pseudo_inverse(epsilon).unwrap() * b_h.column(j)).transpose();

            w.set_row(j, &new_lst_sqr_row)
        }
    }

    (w, h)
}
