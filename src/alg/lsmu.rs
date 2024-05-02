use sandblast::buffer::GpuBuffer;
use sandblast::shader::ComputeShader;
use sandblast::matrix_serialization_utils::matrix_to_casted_array;
use sandblast::device::GpuDevice;
use bytemuck::{Pod, Zeroable, cast_slice};
use futures::join;
use approx::assert_relative_eq;
use std::time::{Instant, Duration};
use std::thread::sleep;



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
pub async fn lee_seung_multiplicative_update_rule(
    matrix_to_factorize: DMatrix<f32>,
    num_synergies: usize,
) -> (DMatrix<f32>, DMatrix<f32>) {

    let mut w = DMatrix::<f32>::new_random(matrix_to_factorize.nrows(), num_synergies).abs();
    let mut h = DMatrix::<f32>::new_random(num_synergies, matrix_to_factorize.ncols()).abs();
    //println!("W {}", w);
    //println!("H {}", h);

    let zeros = DMatrix::<f32>::zeros(matrix_to_factorize.nrows(), matrix_to_factorize.ncols());

    // Prepare Device
    let device = pollster::block_on(GpuDevice::new());

    // Prepare Shader
    let three_matrix_mul_shader = ComputeShader::<f32>::new(&device, "src/gpu/three_matrix_mul.wgsl");
    let two_matrix_mul_shader = ComputeShader::<f32>::new(&device, "src/gpu/matrix_mul_transpose.wgsl");
    let element_wise_mul_div_shader = ComputeShader::<f32>::new(&device, "src/gpu/element_mul_div.wgsl");

    // Instantiate GPU Buffers
    let gpu_w = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&w), false, false);
    let gpu_h = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&h), false, false);
    let gpu_a = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&matrix_to_factorize), false, false);
    let gpu_b = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, false);
    let gpu_c = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, false);
    let gpu_d = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, false);
    let gpu_e = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, false);    

    // Buffers with map_read properties that can be copied back to the CPU
    let gpu_output_0 = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, true);
    let gpu_output_1 = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, true);
    let gpu_output_2 = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, true);
    let gpu_output_3 = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, true);
    let gpu_output_4 = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, true);
    let gpu_output_5 = GpuBuffer::<f32>::new(&device, matrix_to_casted_array(&zeros.clone()), false, true);

    // Prepare metadata for multiplications
    let w_t_metadata = [MatrixMulMetadata{num_rows: w.nrows() as u32, num_cols: w.ncols() as u32, to_tranpose: 1}];
    let w_t_metadata_casted: &[f32] = cast_slice(&w_t_metadata);
    let w_t_metadata_gpu = GpuBuffer::<f32>::new(&device, w_t_metadata_casted, true, false);

    let w_metadata = [MatrixMulMetadata{num_rows: w.nrows() as u32, num_cols: w.ncols() as u32, to_tranpose: 0}];
    let w_metadata_casted: &[f32] = cast_slice(&w_metadata);
    let w_metadata_gpu = GpuBuffer::<f32>::new(&device, w_metadata_casted, true, false);

    let h_metadata = [MatrixMulMetadata{num_rows: h.nrows() as u32, num_cols: h.ncols() as u32, to_tranpose: 0}];
    let h_metadata_casted: &[f32] = cast_slice(&h_metadata);
    let h_metadata_gpu = GpuBuffer::<f32>::new(&device, h_metadata_casted, true, false);

    let h_t_metadata = [MatrixMulMetadata{num_rows: h.nrows() as u32, num_cols: h.ncols() as u32, to_tranpose: 1}];
    let h_t_metadata_casted: &[f32] = cast_slice(&h_t_metadata);
    let h_t_metadata_gpu = GpuBuffer::<f32>::new(&device, h_t_metadata_casted, true, false);

    let a_metadata = [MatrixMulMetadata{num_rows: matrix_to_factorize.nrows() as u32, num_cols: matrix_to_factorize.ncols() as u32, to_tranpose: 0}];
    let a_metadata_casted: &[f32] = cast_slice(&a_metadata);
    let a_metadata_gpu = GpuBuffer::<f32>::new(&device, a_metadata_casted, true, false);

    let b_mul_gpu_buffers = [&gpu_w, &gpu_w, &gpu_h, &gpu_b, &w_t_metadata_gpu, &w_metadata_gpu, &h_metadata_gpu];
    let b_mul_bind_group = three_matrix_mul_shader.bind_group_from_buffers(&b_mul_gpu_buffers);

    let c_mul_gpu_buffers = [&gpu_w, &gpu_a, &gpu_c, &w_t_metadata_gpu, &a_metadata_gpu];
    let c_mul_bind_group = two_matrix_mul_shader.bind_group_from_buffers(&c_mul_gpu_buffers);

    let d_mul_gpu_buffers = [&gpu_w, &gpu_h, &gpu_h, &gpu_d, &w_metadata_gpu, &h_metadata_gpu, &h_t_metadata_gpu];
    let d_mul_bind_group = three_matrix_mul_shader.bind_group_from_buffers(&d_mul_gpu_buffers);

    let e_mul_gpu_buffers = [&gpu_a, &gpu_h, &gpu_e, &a_metadata_gpu, &h_t_metadata_gpu];
    let e_mul_bind_group = two_matrix_mul_shader.bind_group_from_buffers(&e_mul_gpu_buffers);

    let h_mul_div_buffers = [&gpu_h, &gpu_c, &gpu_b];
    let h_mul_div_bind_group = element_wise_mul_div_shader.bind_group_from_buffers(&h_mul_div_buffers);

    let w_mul_div_buffers = [&gpu_w, &gpu_e, &gpu_d];
    let w_mul_div_bind_group = element_wise_mul_div_shader.bind_group_from_buffers(&w_mul_div_buffers);

    println!("Starting compute passes");
    let now = Instant::now();
    let num_iterations = 1000;
    for i in 0..num_iterations {
        println!("{}", i);
        let mut b_output_data = vec![];;
        let mut c_output_data = vec![];;

        
        let combined_future = async {
            (b_output_data, c_output_data) = join!(three_matrix_mul_shader.run(&b_mul_bind_group, &b_mul_gpu_buffers, (matrix_to_factorize.nrows() as u32, matrix_to_factorize.ncols() as u32, 1), None), two_matrix_mul_shader.run(&c_mul_bind_group, &c_mul_gpu_buffers, (matrix_to_factorize.nrows() as u32, matrix_to_factorize.ncols() as u32, 1), None));
        };

        pollster::block_on(combined_future);

        //let reconstructed_b = DMatrix::from_fn(w.ncols() , h.ncols(), |r, c| b_output_data[c * matrix_to_factorize.nrows() + r]);
        //let reconstructed_c = DMatrix::from_fn(w.ncols() , h.ncols(), |r, c| c_output_data[c * matrix_to_factorize.nrows() + r]);

        //assert_eq!(reconstructed_b, w.clone().transpose() * w.clone() * h.clone());
        //assert_eq!(reconstructed_c, w.clone().transpose() * matrix_to_factorize.clone());

        let output_data_h = pollster::block_on(element_wise_mul_div_shader.run(&h_mul_div_bind_group, &h_mul_div_buffers, (matrix_to_factorize.nrows() as u32, matrix_to_factorize.ncols() as u32, 1), None));

        //let reconstructed_h = DMatrix::from_fn(h.nrows() , h.ncols(), |r, c| output_data_h[c * matrix_to_factorize.nrows() + r]);

        //assert_relative_eq!(reconstructed_h, h.component_mul(&reconstructed_c).component_div(&reconstructed_b), epsilon = 0.0001);

        let mut d_output_data = vec![];;
        let mut e_output_data = vec![];;

        let combined_future = async {
            (d_output_data, e_output_data) = join!(three_matrix_mul_shader.run(&d_mul_bind_group, &d_mul_gpu_buffers, (matrix_to_factorize.nrows() as u32, matrix_to_factorize.ncols() as u32, 1), None), two_matrix_mul_shader.run(&e_mul_bind_group, &e_mul_gpu_buffers, (matrix_to_factorize.nrows() as u32, matrix_to_factorize.ncols() as u32, 1), None));
        };

        pollster::block_on(combined_future);

        //let reconstructed_d = DMatrix::from_fn(w.nrows() , w.ncols(), |r, c| d_output_data[c * matrix_to_factorize.nrows() + r]);
        //let reconstructed_e = DMatrix::from_fn(w.nrows() , w.ncols(), |r, c| e_output_data[c * matrix_to_factorize.nrows() + r]);

        //assert_eq!(reconstructed_d, w.clone() * reconstructed_h.clone() * reconstructed_h.clone().transpose());
        //assert_eq!(reconstructed_e, matrix_to_factorize.clone() * reconstructed_h.clone().transpose());

        let output_data_w = pollster::block_on(element_wise_mul_div_shader.run(&w_mul_div_bind_group, &w_mul_div_buffers, (matrix_to_factorize.nrows() as u32, matrix_to_factorize.ncols() as u32, 1), None));

        //let reconstructed_w = DMatrix::from_fn(w.nrows() , w.ncols(), |r, c| output_data_w[c * matrix_to_factorize.nrows() + r]);

        //assert_relative_eq!(reconstructed_w, w.component_mul(&reconstructed_e).component_div(&reconstructed_d), epsilon = 0.0001);
    }
    println!("{} iterations took {}", num_iterations, now.elapsed().as_millis());

    // let buf_slice = matrix_c.buffer.slice(..);

    // Binding info hardcode
    // Output buffer creation
    // GPUDevice Ownership
    // Copy output buffer hardcode
    // Dispatch size and general reliance on matrix num rows/cols
    // Abstract matrix creation/ops?
    
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
