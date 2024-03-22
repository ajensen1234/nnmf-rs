pub mod io;

use crate::io::load_csv;

extern crate nalgebra as na;

use na::DMatrix;

fn main() {
    let my_mat = load_csv::print_string_contents("./data/YA04/val/W_6_R.csv");
}
