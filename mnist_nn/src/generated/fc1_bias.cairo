use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 1892, sign: true });
    data.append(i32 { mag: 448, sign: false });
    data.append(i32 { mag: 1898, sign: false });
    data.append(i32 { mag: 7073, sign: false });
    data.append(i32 { mag: 1707, sign: true });
    data.append(i32 { mag: 4299, sign: false });
    data.append(i32 { mag: 5471, sign: false });
    data.append(i32 { mag: 1579, sign: true });
    data.append(i32 { mag: 3637, sign: false });
    data.append(i32 { mag: 2181, sign: true });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
