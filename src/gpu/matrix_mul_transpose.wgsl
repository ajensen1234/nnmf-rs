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

struct MatrixMulMetadata {
    num_rows: u32,
    num_cols: u32,
    to_transpose: u32,
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
var<uniform> matrix_0_metadata: MatrixMulMetadata;

@group(0)
@binding(4)
var<uniform> matrix_1_metadata: MatrixMulMetadata;


@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_row = global_id.x;
    let output_col = global_id.y;

    var new_val: f32 = 0;
    var current_matrix_0_col: i32 = 0;

    var matrix_0_col_boundary: i32;

    if (matrix_0_metadata.to_transpose == 1) {
        matrix_0_col_boundary = i32(matrix_0_metadata.num_rows);
    } else {
        matrix_0_col_boundary = i32(matrix_0_metadata.num_cols);
    };
    var matrix_0_row_elem_idx: i32;
    var matrix_1_col_elem_idx: i32;

    while (current_matrix_0_col < matrix_0_col_boundary) {
        // Adds product of elements at (output_row, current_matrix_0_col) and (current_matrix_0_col, output_col) to new_val
        if (matrix_0_metadata.to_transpose == 1) {
            matrix_0_row_elem_idx = i32(output_row * matrix_0_metadata.num_rows) + current_matrix_0_col;
        } else {
            matrix_0_row_elem_idx = current_matrix_0_col * i32(matrix_0_metadata.num_rows) + i32(output_row);
        };

        if (matrix_1_metadata.to_transpose == 1) {
            matrix_1_col_elem_idx = current_matrix_0_col * i32(matrix_1_metadata.num_rows) + i32(output_col);
        } else {
            matrix_1_col_elem_idx = i32(output_col * matrix_1_metadata.num_rows) + current_matrix_0_col;
        };

        new_val = new_val + matrix_0.data[matrix_0_row_elem_idx] * matrix_1.data[matrix_1_col_elem_idx];
        current_matrix_0_col += 1;
    }

    // Number of output rows is equal to number of rows of matrix 0
    var num_rows_after_transpose: u32;
    if (matrix_0_metadata.to_transpose) == 1 {
        num_rows_after_transpose = matrix_0_metadata.num_cols;
    } else {
        num_rows_after_transpose = matrix_0_metadata.num_rows;
    };

    let idx = global_id.x + num_rows_after_transpose * global_id.y;
    output_matrix.data[idx] = new_val;
}
