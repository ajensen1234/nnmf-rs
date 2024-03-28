pub mod alg;
pub mod io;
use crate::io::load_csv;

extern crate nalgebra as na;

use alg::alsa::alternating_leastsq_nnmf;

fn main() {
    let path = "./data/YA04/YA04_EMG_L.csv";
    let matrix = load_csv::load_csv_matrix(path);
    let (w, _h) = alternating_leastsq_nnmf(matrix.clone(), 2);

    println!("W : {:?}", w);
    //println!("H : {:?}", h);

    // println!("W times H : {:?}", w * h);

    //assert_eq!(matrix, w * h);
}
