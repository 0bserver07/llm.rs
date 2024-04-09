use tch::{Tensor, Kind};

struct LayerNorm {
    eps: f64,
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn new(num_channels: i64) -> Self {
        let eps = 1e-5;
        let weight = Tensor::rand(&[num_channels], (Kind::Float, tch::Device::Cpu));
        let bias = Tensor::rand(&[num_channels], (Kind::Float, tch::Device::Cpu));
        Self { eps, weight, bias }
    }

    fn forward(&self, x: &Tensor) -> (Tensor, (Tensor, Tensor, Tensor, Tensor)) {
        let (batch_size, seq_len, num_channels) = x.size3().unwrap();
        let mean = x.sum_dim_intlist(&[-1], true, Kind::Float) / num_channels as f64;
        let xshift = x - &mean;
        let var = xshift.pow(2).sum_dim_intlist(&[-1], true, Kind::Float) / num_channels as f64;
        let rstd = (var + self.eps).pow(-0.5);
        let norm = &xshift * &rstd;
        let out = &norm * &self.weight + &self.bias;
        let cache = (x.copy(), self.weight.copy(), mean, rstd);
        (out, cache)
    }

    fn backward(&self, dout: &Tensor, cache: (Tensor, Tensor, Tensor, Tensor)) -> (Tensor, Tensor, Tensor) {
        let (x, w, mean, rstd) = cache;
        let norm = (&x - &mean) * &rstd;
        let db = dout.sum_dim_intlist(&[0, 1], false, Kind::Float);
        let dw = (dout * &norm).sum_dim_intlist(&[0, 1], false, Kind::Float);
        let dnorm = dout * &w;
        let dx = &dnorm - dnorm.mean_dim_intlist(&[-1], true, Kind::Float) - &norm * ((&dnorm * &norm).mean_dim_intlist(&[-1], true, Kind::Float));
        let dx = &dx * &rstd;
        (dx, dw, db)
    }
}