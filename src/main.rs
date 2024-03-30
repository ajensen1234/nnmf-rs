pub mod alg;
pub mod io;
pub mod synth_data;
use crate::io::load_csv;

extern crate nalgebra as na;

use alg::alsa::alternating_leastsq_nnmf;

fn main() {
    let path = "./data/YA04/YA04_EMG_L.csv";
    let matrix = load_csv::load_csv_matrix(path);

    // we want to create a test matrix
    let (W_test, h_test, EMG_test) =
        synth_data::generate_test_data::generate_test_data(7, 20, 3, false);
    println!("W_test: {:?}", W_test);
    println!("h_test: {:?}", h_test);

    // testing out alsa
    let (w_est, h_est) = alternating_leastsq_nnmf(EMG_test, 3);
    assert_eq!(W_test, w_est);
}
