use std::cmp::max;
use ndarray::prelude::*;
extern crate rand;
use rand::{distributions::Standard,Rng};
use ndarray::OwnedRepr;
use std::fmt::{Display, Formatter};

// this uses an array of 2 dimensions and an array of 1 dimensions
// we're going to need to use

pub struct Network {
    layers: Vec<Layer>,
    activation : Vec<ReLu>
}
impl Network {
    pub fn new(depth:usize,n_inputs:usize,n_neurons:usize) -> Self {
        let mut layers:Vec<Layer> = Vec::new();
        let mut activation: Vec<ReLu> = Vec::new();
        for i in 0..depth {
            layers.push(Layer::new(n_inputs,n_neurons));
            activation.push(ReLu{});
        }
        return Self {layers,activation}
    }
}

pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>
}
impl Layer {
    pub fn new(n_inputs:usize,n_neurons:usize) -> Self {
        let randomed: Vec<f32> = rand::thread_rng()
            .sample_iter(Standard)
            .take(n_inputs * n_neurons)
            // .map(|x:f32| x*0.1)
            .collect() ;
        let weights: Array2<f32> = Array::from(randomed)
            .into_shape((n_neurons,n_inputs))
            .unwrap();
        let biases: Array1<f32> = Array::zeros(n_neurons);
        return Self {weights,biases};
    }
    pub fn forward(&self,inputs:ArrayView1<f32>) -> Array1<f32>{
        // let inps =
        let sum = &self.weights.dot(&inputs) + &self.biases;
        return sum;
    }
}
impl Display for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.weights)
    }
}

pub struct ReLu {

}
impl ReLu {
    pub fn forward(&self,inputs:&Array1<f32>) -> Array1<f32> {
        let max:Array1<f32> = inputs.map(|x|x.max(0.0));
        return max;
    }
}

pub struct SoftMax {
    // inputs: Array1<f32>
}

impl SoftMax {
    // pub fn new(inputs:Array1<f32>) -> Self {
    //     return Self {inputs}
    // }
    pub fn forward(&self,inputs:&Array1<f32>) -> Array1<f32> {
        let max = inputs.fold(f32::NEG_INFINITY,|a,&b| a.max(b));
        println!("max: {}",max);
        let exponents = (inputs - max).map(|x| x.exp());
        let p = exponents / inputs.sum();
        return p;
    }
}