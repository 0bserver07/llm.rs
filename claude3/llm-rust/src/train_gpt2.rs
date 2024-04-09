use std::env;
use tch::{nn, Device, Kind, Tensor};
use tiktoken_rs::get_encoding;

// GPT Configuration
struct GPTConfig {
    block_size: i64,
    vocab_size: i64,
    n_layer: i64,
    n_head: i64,
    n_embd: i64,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            block_size: 1024,
            vocab_size: 50257,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
        }
    }
}

// Causal Self Attention
struct CausalSelfAttention {
    c_attn: nn::Linear,
    c_proj: nn::Linear,
    n_head: i64,
    n_embd: i64,
    bias: Tensor,
}

impl CausalSelfAttention {
    fn new(config: &GPTConfig) -> Self {
        let c_attn = nn::linear(config.n_embd, 3 * config.n_embd, Default::default());
        let c_proj = nn::linear(config.n_embd, config.n_embd, Default::default());
        let bias = Tensor::ones(&[config.block_size, config.block_size], (Kind::Float, Device::Cpu))
            .tril(0)
            .view([1, 1, config.block_size, config.block_size]);
        Self {
            c_attn,
            c_proj,
            n_head: config.n_head,
            n_embd: config.n_embd,
            bias,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let (batch_size, seq_len, _) = x.size3().unwrap();
        let qkv = self.c_attn.forward(x);
        let (q, k, v) = qkv.chunk(3, 2);
        let q = q.view([batch_size, seq_len, self.n_head, self.n_embd / self.n_head]).transpose(1, 2);
        let k = k.view([batch_size, seq_len, self.n_head, self.n_embd / self.n_head]).transpose(1, 2);
        let v = v.view([batch_size, seq_len, self.n_head, self.n_embd / self.n_head]).transpose(1, 2);
        let att = q.matmul(&k.transpose(-2, -1)) * (1.0 / (k.size().last().unwrap() as f64).sqrt());
        let att = att.masked_fill(&self.bias.slice(0, 0, seq_len, 1).slice(1, 0, seq_len, 1).eq(0.), std::f64::NEG_INFINITY);
        let att = att.softmax(-1, Kind::Float);
        let y = att.matmul(&v);
        let y = y.transpose(1, 2).contiguous().view([batch_size, seq_len, self.n_embd]);
        self.c_proj.forward(&y)
    }
}

// MLP
struct MLP {
    c_fc: nn::Linear,
    c_proj: nn::Linear,
}

impl MLP {
    fn new(config: &GPTConfig) -> Self {
        let c_fc = nn::linear(config.n_embd, 4 * config.n_embd, Default::default());
        let c_proj = nn::linear(4 * config.n_embd, config.n_embd, Default::default());
        Self { c_fc, c_proj }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.c_fc.forward(x);
        let x = x.gelu("none");
        self.c_proj.forward(&x)
    }
}

// Block
struct Block {
    ln_1: nn::LayerNorm,
    attn: CausalSelfAttention,
    ln_2: nn::LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(config: &GPTConfig) -> Self {
        let ln_1 = nn::layer_norm(config.n_embd, Default::default());
        let attn = CausalSelfAttention::new(config);
        let ln_2 = nn::layer_norm(config.n_embd, Default::default());
        let mlp = MLP::new(config);
        Self { ln_1, attn, ln_2, mlp }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x + self.attn.forward(&self.ln_1.forward(x));
        x + self.mlp.forward(&self.ln_2.forward(&x))
    }
}

// GPT Model
struct GPT {
    config: GPTConfig,
    transformer: nn::Sequential,
    lm_head: nn::Linear,
}

impl GPT {
    fn new(config: &GPTConfig) -> Self {
        let mut transformer = nn::sequential();
        transformer.add(nn::embedding(config.vocab_size, config.n_embd, Default::default()).alias("wte"));
        transformer.add(nn::embedding(config.block_size, config.n_embd, Default::default()).alias("wpe"));
        for _ in 0..config.n_layer {
            transformer.add(Block::new(config));
        }
        transformer.add(nn::layer_norm(config.n_embd, Default::default()).alias("ln_f"));

        let lm_head = nn::linear(config.n_embd, config.vocab_size, Default::default());

        let mut model = Self {
            config: config.clone(),
            transformer,
            lm_head,
        };

        model.reset_parameters();
        model
    }

    fn reset_parameters(&mut self) {
        // Initialize weights
        for name, param in self.named_parameters() {
            if name.ends_with(".bias") {
                nn::init::zeros_(param);
            } else if name.ends_with(".weight") {
                if name.starts_with("transformer.wte") || name.starts_with("transformer.wpe") {
                    nn::init::normal_(param, 0.0, 0.02);
                } else {
                    nn::init::xavier_normal_(param);
                }
            }
        }
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let (batch_size, seq_len) = idx.size2().unwrap();
        assert!(seq_len <= self.config.block_size, "Cannot forward sequence of length {}, block size is only {}", seq_len, self.config.block_size);

        let pos = Tensor::arange(seq_len, (Kind::Int64, idx.device())).unsqueeze(0).repeat(&[batch_size, 1]);

        let tok_emb = self.transformer.get_submodule("wte").unwrap().forward(idx);
        let pos_emb = self.transformer.get_submodule("wpe").unwrap().forward(&pos);
        let mut x = tok_emb + pos_emb;

        x = self.transformer.forward_to_submodule(&x, "ln_f");
        x = self.transformer.get_submodule("ln_f").unwrap().forward(&x);

        let logits = self.lm_head.forward(&x);

        let loss = match targets {
            Some(targets) => {
                let loss = nn::cross_entropy_loss(
                    &logits.view([-1, logits.size().last().unwrap()]),
                    &targets.view([-1]),
                    nn::Reduction::Mean,
                    -1,
                    &Tensor::of_slice(&[]),
                );
                Some(loss)
            }
            None => None,
        };

        (logits, loss)
    }


    fn generate(&self, idx: &Tensor, max_new_tokens: i64, temperature: f64, top_k: Option<i64>) -> Tensor {
        // Text generation
        let mut generated = idx.clone();
        let device = idx.device();

        for _ in 0..max_new_tokens {
            let idx_cond = if generated.size()[1] <= self.config.block_size {
                generated.clone()
            } else {
                generated.slice(1, generated.size()[1] - self.config.block_size, generated.size()[1], 1)
            };

            let (logits, _) = self.forward_t(&idx_cond, None, false).unwrap();
            let logits = logits.select(1, -1).squeeze() / temperature;

            let probs = match top_k {
                Some(k) => {
                    let (top_k_values, top_k_indices) = logits.topk(k, -1, true, false);
                    let mut probs = nn::functional::softmax(&top_k_values, -1, Kind::Float);
                    let mut indices = Vec::with_capacity(probs.size()[0]);
                    for i in 0..probs.size()[0] {
                        let p = probs.get(i).double_value(&[]);
                        let idx = top_k_indices.get(i).int64_value(&[]);
                        indices.push((idx, p));
                    }
                    indices
                }
                None => {
                    let probs = nn::functional::softmax(&logits, -1, Kind::Float);
                    let mut indices = Vec::with_capacity(probs.size()[0]);
                    for (i, p) in probs.iter().enumerate() {
                        indices.push((i as i64, p.double_value(&[])));
                    }
                    indices
                }
            };

            let next_token = Tensor::of_slice(&[rand::distributions::WeightedIndex::new(&probs).sample(&mut rand::thread_rng())]).to(device);
            generated = Tensor::cat(&[generated, next_token.unsqueeze(0)], 1);
        }

        generated
    }

}

fn main() {
    let device = Device::cuda_if_available();

    let encoding = get_encoding("gpt2").unwrap();

    let config = GPTConfig::default();
    let model = GPT::new(&config);

    // Load the pre-trained weights
    // Load the pre-trained weights
    let model_type = "gpt2";
    let weights_file = format!("{}_124M.bin", model_type);
    let weights = std::fs::read(weights_file).unwrap();
    model.load_state_dict(&weights).unwrap();

    // Prepare the data
    // Prepare the data
    let data_file = "path/to/data/file.txt";
    let text = std::fs::read_to_string(data_file).unwrap();
    let tokens: Vec<i64> = encoding.encode_with_special_tokens(&text);
    let token_tensor = Tensor::of_slice(&tokens).to(device);

    let batch_size = 4;
    let seq_len = 64;
    let num_batches = tokens.len() / (batch_size * seq_len);

    let data_loader = token_tensor
        .chunk(batch_size * seq_len, 0)
        .take(num_batches as usize)
        .map(|chunk| chunk.view([batch_size, seq_len]));

    // Training loop
    // Training loop
    let num_epochs = 10;
    let optimizer = nn::Adam::default().build(&model.parameters()).unwrap();

    for epoch in 0..num_epochs {
        for (batch_idx, batch) in data_loader.enumerate() {
            let x = batch;
            let y = batch.slice(1, 1, seq_len + 1, 1);

            let (logits, loss) = model.forward_t(&x, Some(&y), false).unwrap();
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if (batch_idx + 1) % 100 == 0 {
                println!("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4}", epoch + 1, num_epochs, batch_idx + 1, num_batches, loss.double_value(&[]));
            }
        }
    }

    // Inference
    // Inference
    let start_text = "<|endoftext|>";
    let start_ids = encoding.encode_with_special_tokens(start_text);
    let start_tensor = Tensor::of_slice(&start_ids).to(device).unsqueeze(0);

    let max_new_tokens = 16;
    let temperature = 1.0;
    let top_k = 40;

    model.eval();
    let generated = model.generate(&start_tensor, max_new_tokens, temperature, Some(top_k));
    let generated_text = encoding.decode(generated.view(-1).to_vec());
    println!("Generated text: {}", generated_text);

    // Save the model and state
    // Save the model and state
    let model_file = "gpt2_124M.bin";
    let state_file = "gpt2_124M_debug_state.bin";

    let model_bytes = model.save_state_dict().unwrap();
    std::fs::write(model_file, model_bytes).unwrap();

    let state_bytes = model.save_state().unwrap();
    std::fs::write(state_file, state_bytes).unwrap();


}