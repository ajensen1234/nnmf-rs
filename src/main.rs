pub mod alg;
pub mod io;
pub mod synth_data;
pub mod gpu;
use crate::io::load_csv;
use crate::io::write_csv::write_matrix_to_csv;

use approx::relative_eq;
use approx::assert_relative_eq;

use nalgebra as na;
use na::DMatrix;

use alg::alsa::alternating_leastsq_nnmf;
use alg::lsmu::lee_seung_multiplicative_update_rule;

fn main() {
    let path = "./data/YA04/YA04_EMG_L.csv";
    let matrix = load_csv::load_csv_matrix(path);

    // we want to create a test matrix
    let (W_test, h_test, EMG_test) =
        synth_data::generate_test_data::generate_test_data(8, 8, 2, false);
    println!("W_test: {:?}", W_test);
    println!("h_test: {:?}", h_test);

    let (w_est, h_est) = lee_seung_multiplicative_update_rule(EMG_test.clone(), 2);

    assert_relative_eq!(EMG_test, w_est.clone() * h_est.clone(), epsilon = 0.001)
}
