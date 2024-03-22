use std::env;
use std::fs;

pub fn print_string_contents(fp: &str) {
    println!("We are in: {:?}", env::current_dir());
    println!("File Name {}", fp);
    let contents = fs::read_to_string(fp).expect("We should have been able to read the file");

    println!("Contents: {}", contents);

    for line in contents.lines() {
        let v: Vec<_> = line
            .split([','])
            .map(|char| char.parse::<f64>().unwrap())
            .collect();
        println!("{:?}", v[0]);
    }
}
