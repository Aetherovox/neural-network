mod neuron;
mod data;
use ndarray::prelude::*;
use crate::data::Data;
use crate::neuron::{Layer, Network,SoftMax,ReLu};
#[macro_use(s)]

fn main () {

    let data = Data::new();

    let layer1: Layer = Layer::new(28*28, 128);
    let activation: SoftMax = SoftMax{};
    let layer2: Layer = Layer::new(128, 64);
    let layer3: Layer = Layer::new(64,10);
    let out1 = layer1.forward(data.train_data.slice(s![0,..]));
        println!("raw1 {}",out1);
    let act1 = activation.forward(&out1);
        println!("activated1 {}",act1);
    let out2= layer2.forward(act1.slice(s![..]));
        println!("raw2 {}",out2);
    let act2 = activation.forward(&out2);
        println!("activated2 {}",act2);
    let out3= layer3.forward(act2.slice(s![..]));
        println!("raw3 {}",out3);
    let act3 = activation.forward(&out3);
        println!("activated3 {}",act3);

}

