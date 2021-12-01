use mnist::*;
use ndarray::prelude::*;

pub struct Data {
    pub train_data:Array2<f32>,
    pub train_labels:Array2<f32>,
    pub test_data:Array2<f32>,
    pub test_labels:Array2<f32>

}
impl Data {
    pub fn new() -> Self {
        let NormalizedMnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize()
            .normalize();
        let image_num = 0;
// Can use an Array2 or Array3 here (Array3 for visualization)
        let train_data = Array2::from_shape_vec((50_000, 28*28), trn_img)
            .expect("Error converting training images to Array2 struct")
            .map(|x| *x as f32 / 256.0);
        println!("{:#.1?}\n", train_data.slice(s![ image_num, ..]));


// Convert the returned Mnist struct to Array2 format
        let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
            .expect("Error converting training labels to Array2 struct")
            .map(|x| *x as f32);
        println!("The first digit is a {:?}", train_labels.slice(s![image_num, ..]));

        let test_data = Array2::from_shape_vec((10_000, 28*28), tst_img)
            .expect("Error converting testing images to Array3 struct")
            .map(|x| *x as f32 / 256.);

        let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
            .expect("Error converting testing labels to Array2 struct")
            .map(|x| *x as f32);
        return Self {train_data,train_labels,test_data,test_labels}
    }
}