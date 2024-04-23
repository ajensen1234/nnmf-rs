// use crate::gpu::component_wise_mul_div::{
//     gpu_component_wise_mul_div, prepare_gpu, run_compute_pipeline,
// };
use sandblast;

use approx::relative_eq;
use nalgebra::{DMatrix, DMatrixSlice, DVector, Scalar, QR};
use ndarray::{arr2, Array1, ArrayView1, ArrayView2};
pub fn lee_seung_multiplicative_update_rule(
    matrix_to_factorize: DMatrix<f32>,
    num_synergies: usize,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let m = matrix_to_factorize.nrows();
    let n = matrix_to_factorize.ncols();

    let old_way = false;
    let mut w = DMatrix::<f32>::new_random(m, num_synergies).abs();
    let mut h = DMatrix::<f32>::new_random(num_synergies, n).abs();

    // let mut i = 1;
    // while true {
    //     let b = w.transpose() * w.clone() * h.clone();
    //     let c = w.transpose() * matrix_to_factorize.clone();

    //     // Can parallelize these implementations
    //     h = pollster::block_on(gpu_component_wise_mul_div(h, c, b)).unwrap();

    //     let d = w.clone() * h.clone() * h.transpose();
    //     let e = matrix_to_factorize.clone() * h.transpose();

    //     let prev_w = w.clone();
    //     w = pollster::block_on(gpu_component_wise_mul_div(w, e, d)).unwrap();

    //     if relative_eq!(prev_w, w, epsilon = 0.00000000001) {
    //         println!("Convergence after {} iterations", i);
    //         break;
    //     }
    //     i += 1;
    //     if i % 10 == 0 {
    //         println!("Iteration: {}", i);
    //     }
    // }

    // while true {
    //     let b = w.transpose() * w.clone() * h.clone();
    //     let c = w.transpose() * matrix_to_factorize.clone();

    //     // prepare the data coming from the GPU
    //     let (dev1, queue1, pl1, bg1, in_buf1, out_buf1, buf_len1) =
    //         pollster::block_on(prepare_gpu(h, c, b));
    // }
    return (w, h);
}
