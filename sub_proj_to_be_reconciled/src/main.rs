use nalgebra::{U2, U3, Dynamic, ArrayStorage, VecStorage, SMatrix, Vector, OMatrix, DefaultAllocator, Scalar};
use nalgebra::linalg::{QR};
use lstsq::{lstsq};
use std::cmp;
use nalgebra::allocator::Allocator;
use nalgebra::dimension::{Dim, DimMin, U1};


fn main() {
    type Matrix2x3f = SMatrix<f64, 4, 2>;
    let a = Matrix2x3f::new(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8);

    let m = a.nrows();
    let n = a.ncols();
    let k = 2;
    println!("K {}", k);

    // To randomize
    let mut w = OMatrix::<f64, Dynamic, Dynamic>::identity(m, k);
    let mut h = OMatrix::<f64, Dynamic, Dynamic>::identity(k, n);

    println!("M: {}", m);
    println!("N: {}", n);

    let num_iterations = 3;

    for iter in 0..num_iterations {
        let qr = QR::new(w.clone());
        let b = qr.q().transpose() * a;
        let r = qr.r();
        let epsilon = 1e-14;


        let tmp_col = b.column(0);
        let column_vector = tmp_col.clone_owned();

        for i in 0..n {
            let tmp_col = b.column(i);
            let tmp_col_owned = tmp_col.clone_owned();


            let lstq_result = lstsq(&r, &tmp_col_owned, epsilon);
            let new_col = lstq_result.unwrap().solution;


            h.set_column(i, &new_col);
        }

        let qr_h = QR::new(h.transpose());
        
        let b_h = (qr_h.q().transpose()) * (a.transpose());
        println!("A");
        
        let r_h = qr_h.r();

        println!("Number of rows B: {}", b_h.nrows());
        println!("Number of columns B: {}", b_h.ncols());

        println!("Number of rows R: {}", r_h.nrows());
        println!("Number of columns R: {}", r_h.ncols());

        for j in 0..m {
            let tmp_col_h = b_h.column(j);
            let tmp_col_owned_h = tmp_col_h.clone_owned();

            let lstq_result = lstsq(&r_h, &tmp_col_owned_h, epsilon);
            let new_row = lstq_result.unwrap().solution.transpose();

            w.set_row(j, &new_row)
        }

        println!("Iteration: {}", iter);
        println!("H: {}", h);
        println!("W: {}", w);

    }

    let prod = w * h;
    println!("Prod: {}", prod);

    alternating_leastsq_nnmf(a);
}

fn alternating_leastsq_nnmf<N, R, C>(matrix_to_factorize: OMatrix<N, R, C>) -> OMatrix<N, R, C> where
    N: Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>
{
    let m = matrix_to_factorize.nrows();
    let n = matrix_to_factorize.ncols();
    let k = 2;
    println!("K {}", k);

    // To randomize
    let mut w = OMatrix::<f64, Dynamic, Dynamic>::identity(m, k);
    let mut h = OMatrix::<f64, Dynamic, Dynamic>::identity(k, n);

    println!("M: {}", m);
    println!("N: {}", n);

    matrix_to_factorize
}