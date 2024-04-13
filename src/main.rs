pub mod alg;
pub mod gpu;
pub mod io;
pub mod synth_data;
use crate::io::load_csv;
use crate::io::write_csv::write_matrix_to_csv;

use approx::assert_relative_eq;
use approx::relative_eq;

use na::DMatrix;
use nalgebra as na;

use alg::alsa::alternating_leastsq_nnmf;
use alg::lsmu::lee_seung_multiplicative_update_rule;
use std::time;
fn main() {
    let path = "./data/YA04/YA04_EMG_L.csv";
    let matrix = load_csv::load_csv_matrix(path);

    // we want to create a test matrix
    let (W_test, h_test, EMG_test) =
        synth_data::generate_test_data::generate_test_data(8, 3000, 4, false);
    //println!("W_test: {:?}", W_test);
    //println!("h_test: {:?}", h_test);

    let now = time::Instant::now();
    let (w_est, h_est) = lee_seung_multiplicative_update_rule(matrix.clone(), 4);
    let elapsed_time = now.elapsed();
    let A_est = w_est * h_est;
    if relative_eq!(matrix, A_est.clone(), epsilon = 0.0001) {
        println!("yay");
        for i in 0..10 {
            println!("{:?}", matrix[i]);
            println!("{:?}", A_est[i]);
            println!("-----------")
        }
        println!("{:?}", matrix[1]);
        println!("{:?}", A_est[1]);
    } else {
        println!("Nay");
        for i in 0..10 {
            println!("{:?}", matrix[i]);
            println!("{:?}", A_est[i]);
            println!("-----------")
        }
    }
    println!(
        "Running lee_seung..() took {} seconds.",
        elapsed_time.as_millis() / 1000
    );
}
