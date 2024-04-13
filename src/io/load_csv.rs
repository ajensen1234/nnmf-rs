use std::fs;
extern crate nalgebra as na;
extern crate ndarray as np;
use na::*;
pub fn load_csv_matrix(fp: &str) -> DMatrix<f32> {
    let contents = fs::read_to_string(fp).expect("We should have been able to read the file");
    let vec_of_vecs: Vec<Vec<_>> = contents
        .lines()
        .map(|line| {
            line.split([','])
                .map(|char| {
                    if char.parse::<f32>().unwrap() < 0.0 {
                        0.0
                    } else {
                        char.parse::<f32>().unwrap()
                    }
                })
                .collect()
        })
        .collect();

    let rows = vec_of_vecs.len();
    let cols = vec_of_vecs[0].len();

    let matrix = na::DMatrix::from_fn(rows, cols, |r, c| vec_of_vecs[r][c]);

    assert_eq!(vec_of_vecs[1][2], matrix[(1, 2)]);

    return matrix;
}
