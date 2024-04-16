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
var<storage, read_write> v_indices: DataBuf;

@group(0)
@binding(1)
var<storage, read_write> v_indices_multiply: DataBuf;

@group(0)
@binding(2)
var<storage, read_write> v_indices_divide: DataBuf;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices.data[global_id.x] = v_indices.data[global_id.x] * v_indices_multiply.data[global_id.x] / v_indices_divide.data[global_id.x];
}
