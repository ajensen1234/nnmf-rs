use std::time::Instant;
use std::error::Error;

use wgpu::util::DeviceExt;

use bytemuck;

use na::DMatrix;
use nalgebra as na;

// TODO: Elaborate this this mutates matrix 0
pub async fn gpu_component_wise_mul(matrix_0: DMatrix<f32>, matrix_1: DMatrix<f32>) -> Result<DMatrix<f32>, Box<dyn Error>> {
    let num_rows = matrix_0.nrows();
    let num_cols = matrix_1.ncols();
    // TODO: Assert dimension equivalence

    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: features & wgpu::Features::TIMESTAMP_QUERY,
                limits: Default::default(),
            },
            None,
        )
        .await
        .unwrap();
    let query_set = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 2,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };

    let start_instant = Instant::now();
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        //source: wgpu::ShaderSource::SpirV(bytes_to_u32(include_bytes!("alu.spv")).into()),
        source: wgpu::ShaderSource::Wgsl(include_str!("element_multiplication.wgsl").into()),
    });
    // println!("shader compilation {:?}", start_instant.elapsed());
    let input_f_0: &[f32] = matrix_0.as_slice().try_into().unwrap();
    let input_f_1: &[f32] = matrix_1.as_slice().try_into().unwrap();


    let input_0: &[u8] = bytemuck::cast_slice(input_f_0);
    let input_1: &[u8] = bytemuck::cast_slice(input_f_1);

    let input_buf_0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: input_0,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let input_buf_1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: input_1,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input_0.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: input_buf_0.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: input_buf_1.as_entire_binding(),            
        }
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(input_f_0.len() as u32, 1, 1);
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }
    encoder.copy_buffer_to_buffer(&input_buf_0, 0, &output_buf, 0, input_0.len() as u64);
    queue.submit(Some(encoder.finish()));

    let buf_slice = output_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    // println!("pre-poll {:?}", std::time::Instant::now());
    device.poll(wgpu::Maintain::Wait);
    // println!("post-poll {:?}", std::time::Instant::now());
    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*buf_slice.get_mapped_range();
        let data: &[f32] = bytemuck::cast_slice(data_raw);
        // println!("data: {:?}", &*data);

        let reconstructed_matrix = na::DMatrix::from_fn(num_rows, num_cols, |r, c| data[c * num_rows + r]);
        // println!("{}", reconstructed_matrix);
        assert_eq!(reconstructed_matrix, matrix_0.component_mul(&matrix_1));

        return Ok(reconstructed_matrix)
    }
    return Err("Could not complete GPU component wise multiplication".into())
}