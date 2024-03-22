pub mod io;

use crate::io::load_csv;

extern crate nalgebra as na;

use na::DMatrix;

fn main() {
    let path = "./data/YA04/val/W_6_R.csv";
    match load_csv::print_string_contents(path) {
        Ok(matrix) => println!("The Matrix: {:?}", matrix),
        Err(e) => println!("The matrix didn't work :(: {:?}", e),
    }
}
