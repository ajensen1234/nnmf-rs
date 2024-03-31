use nalgebra::DMatrix;
use csv::Writer;
use std::error::Error;
use std::fs::File;

pub fn write_matrix_to_csv(matrix: &DMatrix<f64>, file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_writer(File::create(file_path)?);

    for row in matrix.row_iter() {
        // Collect the row's values into a Vec<f64>
        let row_vec: Vec<f64> = row.iter().copied().collect();
        // Write the row to the CSV
        wtr.serialize(row_vec)?;
    }

    wtr.flush()?;
    Ok(())
}