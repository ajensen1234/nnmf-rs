// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

struct DataBuf {
    data: array<f32>,
}

@group(0)
@binding(0)
var<storage, read_write> matrix_0: DataBuf;

@group(0)
@binding(1)
var<storage, read_write> matrix_1: DataBuf;

@group(0)
@binding(2)
var<storage, read_write> output_matrix: DataBuf;

@group(0)
@binding(3)
var<storage, read_write> num_rows_matrix_0: u32;

@group(0)
@binding(4)
var<storage, read_write> num_cols_matrix_0: u32;

@group(0)
@binding(5)
var<storage, read_write> num_cols_matrix_1: u32;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_row = global_id.x % num_rows;
    let output_col = global_id.x / num_rows;

    let new_val: f32 = 0;
    let current_matrix_0_col: u32 = 0;

    while (current_matrix_0_col < num_cols_matrix_0) {
        // Adds product of elements at (output_row, current_matrix_0_col) and (current_matrix_0_col, output_col) to new_val
        new_val += matrix_0[current_matrix_0_col * num_rows_matrix_0 + output_row] * matrix_1[output_col * num_cols_matrix_0 + current_matrix_0_col]
        current_matrix_0_col += 1;
    }

    output_matrix[global_id.x] = new_val;
}
