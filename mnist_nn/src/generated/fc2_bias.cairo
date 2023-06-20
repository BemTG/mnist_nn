use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 298, sign: true });
    data.append(i32 { mag: 344, sign: false });
    data.append(i32 { mag: 424, sign: false });
    data.append(i32 { mag: 1036, sign: true });
    data.append(i32 { mag: 46, sign: false });
    data.append(i32 { mag: 1567, sign: false });
    data.append(i32 { mag: 129, sign: true });
    data.append(i32 { mag: 206, sign: false });
    data.append(i32 { mag: 1635, sign: true });
    data.append(i32 { mag: 176, sign: false });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
