use nalgebra::{U2, U3, Dynamic, ArrayStorage, VecStorage, SMatrix, Vector, OMatrix, DefaultAllocator, Scalar, DMatrix};
use nalgebra::linalg::{QR};
use lstsq::{lstsq};
use std::cmp;
use nalgebra::allocator::Allocator;
use nalgebra::dimension::{Dim, DimMin, U1};


fn main() {
    let a = DMatrix::identity(15, 15);

    alternating_leastsq_nnmf(a, 2);
}

fn alternating_leastsq_nnmf(matrix_to_factorize: DMatrix<f64>, num_synergies: usize) -> (DMatrix<f64>, DMatrix<f64>)
{
    let m = matrix_to_factorize.nrows();
    let n = matrix_to_factorize.ncols();

    // TODO randomize
    let mut w = DMatrix::<f64>::identity(m, num_synergies);
    let mut h = DMatrix::<f64>::identity(num_synergies, n);

    println!("M: {}", m);
    println!("N: {}", n);

    let num_iterations = 3;
    let epsilon = 1e-14;

    for iter in 0..num_iterations {
        let qr_w = QR::new(w.clone());
        let b_w = qr_w.q().transpose() * matrix_to_factorize.clone();
        let r_w = qr_w.r();

        for i in 0..n {
            let new_lst_sqr_col = r_w.clone().pseudo_inverse(epsilon).unwrap() * b_w.column(i);;

            h.set_column(i, &new_lst_sqr_col);
        }

        let qr_h = QR::new(h.transpose());
        
        let b_h = (qr_h.q().transpose()) * (matrix_to_factorize.transpose());
        println!("A");
        
        let r_h = qr_h.r();

        println!("Number of rows B: {}", b_h.nrows());
        println!("Number of columns B: {}", b_h.ncols());

        println!("Number of rows R: {}", r_h.nrows());
        println!("Number of columns R: {}", r_h.ncols());

        for j in 0..m {
            let tmp_col_h = b_h.column(j);
            let new_lst_sqr_row = (r_h.clone().pseudo_inverse(epsilon).unwrap() * b_h.column(j)).transpose();

            w.set_row(j, &new_lst_sqr_row)
        }

        println!("Iteration: {}", iter);
        println!("H: {}", h);
        println!("W: {}", w);

    }

    (w, h)
}