use std::env;
use std::fs;

extern crate nalgebra as na;
extern crate ndarray as np;
use na::*;
pub fn print_string_contents(fp: &str) -> DMatrix<f64> {
    println!("We are in: {:?}", env::current_dir());
    println!("File Name {}", fp);
    let contents = fs::read_to_string(fp).expect("We should have been able to read the file");

    println!("Lines as vectors of floats: \n ======================================");
    for line in contents.lines() {
        let v: Vec<_> = line
            .split([','])
            .map(|char| char.parse::<f64>().unwrap())
            .collect();
        println!("{:?}", v);
    }
    let vec_of_vecs: Vec<Vec<_>> = contents
        .lines()
        .map(|line| {
            line.split([','])
                .map(|char| char.parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    println!("Vec of vecs: {:?}", vec_of_vecs);

    let rows = vec_of_vecs.len();
    let cols = vec_of_vecs[0].len();

    let matrix = na::DMatrix::from_fn(rows, cols, |r, c| vec_of_vecs[r][c]);

    println!("Our Matrix: {:?}", matrix);

    assert_eq!(vec_of_vecs[1][2], matrix[(1, 2)]);

    return matrix;
}
