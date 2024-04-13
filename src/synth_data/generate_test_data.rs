use na::DMatrix;
use nalgebra as na;

pub fn generate_test_data(
    num_muscles: usize,
    num_time_points: usize,
    num_synergies: usize,
    add_noise: bool,
) -> (na::DMatrix<f32>, na::DMatrix<f32>, na::DMatrix<f32>) {
    // We need to generate some random matrices with the correct size
    // W (synergies) will be defining the muscles synergies based on the input muscles - num_muscles x num_synergies
    // h (activations) defines the activations of the synergies at each of the time points - num_synergies x num_time_points
    let W: DMatrix<f32> = na::DMatrix::new_random(num_muscles, num_synergies).abs();
    let h: DMatrix<f32> = na::DMatrix::new_random(num_synergies, num_time_points).abs();

    let mut EMG = W.clone() * h.clone();
    if add_noise {
        EMG = EMG + DMatrix::new_random(num_muscles, num_time_points);
    }

    return (W, h, EMG);
}
