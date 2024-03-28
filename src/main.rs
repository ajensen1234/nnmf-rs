pub mod alg;
pub mod io;
use crate::alg::alsa;
use crate::io::load_csv;

extern crate nalgebra as na;

use alg::alsa::alternating_leastsq_nnmf;
use na::DMatrix;

fn main() {
    let path = "./data/YA04/YA04_EMG_L.csv";
    let matrix = load_csv::load_csv_matrix(path);
    let (w, h) = alternating_leastsq_nnmf(matrix.clone(), 4);

    println!("W : {:?}", w);
    println!("H : {:?}", h);

    // println!("W times H : {:?}", w * h);

    assert_eq!(matrix, w * h);
}
