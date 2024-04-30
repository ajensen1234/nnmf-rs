use sandblast::buffer::GpuBuffer;
use sandblast::shader::ComputeShader;
use sandblast::matrix_serialization_utils::matrix_to_casted_array;
use sandblast::device::GpuDevice;
use bytemuck::{Pod, Zeroable, cast_slice};

use pollster;
use wgpu;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatrixMulMetadata {
    num_rows: u32,
    num_cols: u32,
    to_tranpose: u32
}
use approx::relative_eq;
use nalgebra::{DMatrix, DMatrixSlice, DVector, Scalar, QR};
use ndarray::{arr2, Array1, ArrayView1, ArrayView2};
pub fn lee_seung_multiplicative_update_rule(
    matrix_to_factorize: DMatrix<f32>,
    num_synergies: usize,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let device = pollster::block_on(GpuDevice::new());

    let num_rows = matrix_to_factorize.nrows();
    let num_cols = matrix_to_factorize.ncols();

    let old_way = false;
    let mut w = DMatrix::<f32>::new_random(num_rows, num_synergies).abs();
    let mut h = DMatrix::<f32>::new_random(num_synergies, num_cols).abs();
    println!("Matrix to Factorize {}", matrix_to_factorize);
    println!("H transpose {}", h.transpose());


    let zeros_1 = DMatrix::<f32>::zeros(num_rows, num_cols);
    let zeros_2 = DMatrix::<f32>::zeros(num_rows, num_cols);

    let matrix_a = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&matrix_to_factorize), false, false);
    let matrix_b = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&matrix_to_factorize), false, false);
    let matrix_c = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&matrix_to_factorize), false, false);
    let matrix_d = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros_1), false, false);
    let output_matrix = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros_2), false, true);

    let matrix_a_mul_metadata = [MatrixMulMetadata{num_rows: num_rows as u32, num_cols: num_cols as u32, to_tranpose: 0}];
    let matrix_a_mul_metadata_casted: &[f32] = cast_slice(&matrix_a_mul_metadata);
    let matrix_a_mul_metadata_gpu = GpuBuffer::<f32>::new(&device, matrix_a_mul_metadata_casted, true, false);

    let matrix_b_mul_metadata = [MatrixMulMetadata{num_rows: num_rows as u32, num_cols: num_cols as u32, to_tranpose: 0}];
    let matrix_b_mul_metadata_casted: &[f32] = cast_slice(&matrix_b_mul_metadata);
    let matrix_b_mul_metadata_gpu = GpuBuffer::<f32>::new(&device, matrix_b_mul_metadata_casted, true, false);

    let matrix_c_mul_metadata = [MatrixMulMetadata{num_rows: num_rows as u32, num_cols: num_cols as u32, to_tranpose: 0}];
    let matrix_c_mul_metadata_casted: &[f32] = cast_slice(&matrix_c_mul_metadata);
    let matrix_c_mul_metadata_gpu = GpuBuffer::<f32>::new(&device, matrix_c_mul_metadata_casted, true, false);

    let shader = ComputeShader::<f32>::new(device, "src/gpu/three_matrix_mul.wgsl");

    let output_data = pollster::block_on(shader.run(&[matrix_a, matrix_b, matrix_c, matrix_d, matrix_a_mul_metadata_gpu, matrix_b_mul_metadata_gpu, matrix_c_mul_metadata_gpu], (num_rows as u32, num_cols as u32, 1), &output_matrix));

    let reconstructed_matrix = DMatrix::from_fn(num_rows , num_cols, |r, c| output_data[c * num_rows + r]);

    assert_eq!(reconstructed_matrix, matrix_to_factorize.clone() * matrix_to_factorize.clone() * matrix_to_factorize.clone());

    // let buf_slice = matrix_c.buffer.slice(..);

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
