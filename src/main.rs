use std::fmt;

fn print_all(x: &Vec<f32>) -> fmt::Result {
    for e in x.iter() {
        print!("{}, ", e);
    }
    Ok(())
}

#[derive(Debug)]
struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    name: String,
}

impl Tensor<f32> {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        if data.len() != shape.iter().fold(1usize, |acc, &val| acc * val) {
            panic!("Shape dimension does not match Data dimension")
        }
        Self {
            data: data,
            name: "?".to_string(),
            shape: shape,
        }
    }
    pub fn size(&self) -> usize {
        self.shape.iter().fold(1usize, |acc, &val| acc * val)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let mut res_data = Vec::new();
        res_data.resize(shape.iter().fold(1usize, |accum, &v| accum * v), 1f32);

        Self {
            data: res_data,
            shape: shape,
            name: "?".to_string(),
        }
    }

    pub fn empty() -> Self {
        Self {
            data: vec![],
            shape: vec![],
            name: "?".to_string(),
        }
    }

    pub fn fill_like(like: &Self, initial: f32) -> Self {
        let mut result = Self {
            data: vec![],
            shape: like.shape.clone(),
            name: "?".to_string(),
        };
        result.data.resize(like.size(), initial);
        result
    }

    pub fn x(shape: Vec<usize>, x: f32) -> Self {
        let mut res_data = Vec::new();
        res_data.resize(shape.iter().fold(1usize, |accum, &v| accum * v), x);
        Self {
            data: res_data,
            shape: shape,

            name: "?".to_string(),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            Err("Shape Mismatch while performing add op".to_string())
        } else {
            let mut result = Self::fill_like(self, 0.);
            for i in 0..self.size() {
                result.data[i] += self.data[i] + other.data[i];
            }
            Ok(result)
        }
    }

    pub fn relu(&mut self) -> () {
        for i in self.data.iter_mut() {
            *i = if *i > 0. { *i } else { 0. };
        }
    }

    pub fn sigmoid(&mut self) -> () {
        for i in self.data.iter_mut() {
            *i = 1. / (1. + f32::exp(-*i));
        }
    }

    /// reluX
    /// ======
    pub fn relu_x(&mut self, upper_threshold: f32) -> () {
        for i in self.data.iter_mut() {
            let unclampped = if *i > 0f32 { *i } else { 0f32 };
            *i = if unclampped > upper_threshold {
                upper_threshold
            } else {
                unclampped
            }
        }
    }

    pub fn relu6(&mut self) -> () {
        self.relu_x(6.0f32)
    }
}

impl std::fmt::Display for Tensor<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.data.len() > 0 {
            true => {
                match write!(
                    f,
                    "[Tensor:size={},shape={}] ",
                    self.size(),
                    self.shape
                        .iter()
                        .fold(String::new(), |x: String, v: &usize| {
                            let mut y = String::new();
                            y.push_str(&x);
                            if x != String::new() {
                                y.push_str(", ");
                            }
                            y.push_str(&v.to_string());
                            y
                        })
                ) {
                    Ok(()) => {}
                    Err(_) => panic!(),
                };
                print_all(&self.data)
            }
            false => write!(f, "(Empty)"),
        }
    }
}

#[test]
fn relu_positive_test() {
    let mut tensor = Tensor::<f32>::ones(vec![23usize, 24usize]);
    tensor.relu();
    assert_eq!(tensor.data, vec![1f32; 23 * 24]);
}

#[test]
fn relu_negative_test() {
    let mut tensor = Tensor::x(vec![23usize, 24usize], -1.);
    tensor.relu();
    assert_eq!(tensor.data, vec![0.; 23 * 24]);
}

#[test]
fn relu6_negative_test() {
    let mut tensor = Tensor::x(vec![23usize, 24usize], -1.);
    tensor.relu6();
    assert_eq!(tensor.data, vec![0.; 23 * 24]);
}

#[test]
fn relu6_positive_test() {
    let mut tensor = Tensor::<f32>::ones(vec![23usize, 24usize]);
    tensor.relu6();
    assert_eq!(tensor.data, vec![1.; 23 * 24]);
}

#[test]
fn relu6_positive_exceeding_test() {
    let mut tensor = Tensor::x(vec![23usize, 24usize], 98.);
    tensor.relu6();
    assert_eq!(tensor.data, vec![6.; 23 * 24]);
}

fn main() -> () {
    let mut tensor_x = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2usize, 2usize, 3usize],
    );
    tensor_x.name = "x".to_string();
    let mut tensor_y = Tensor::fill_like(&tensor_x, 30.0f32);
    tensor_y.name = "y".to_string();

    println!("{}", tensor_x);
    println!("{}", tensor_y);

    match tensor_x.add(&tensor_y) {
        Ok(r) => println!("{}", r),
        Err(e) => panic!(e),
    }
}
