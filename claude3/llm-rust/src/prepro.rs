use std::io::{BufRead, BufReader, Write};
use tiktoken_rs::models::p50k_base;

pub fn preprocess_tinyshakespeare() {
    let input_file = "data/tinyshakespeare.txt";
    let output_file_train = "data/tinyshakespeare_train.bin";
    let output_file_val = "data/tinyshakespeare_val.bin";

    let bpe = p50k_base().unwrap();

    let input = std::fs::File::open(input_file).expect("Failed to open input file");
    let reader = BufReader::new(input);

    let mut train_tokens = Vec::new();
    let mut val_tokens = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let tokens = bpe.encode_with_special_tokens(&line);

        if train_tokens.len() < 32768 {
            val_tokens.extend_from_slice(&tokens);
        } else {
            train_tokens.extend_from_slice(&tokens);
        }
    }

    let mut output_train = std::fs::File::create(output_file_train).expect("Failed to create train output file");
    let mut output_val = std::fs::File::create(output_file_val).expect("Failed to create val output file");

    output_train.write_all(train_tokens.as_slice()).expect("Failed to write train tokens");
    output_val.write_all(val_tokens.as_slice()).expect("Failed to write val tokens");

    println!("Saved {} tokens to {}", val_tokens.len(), output_file_val);
    println!("Saved {} tokens to {}", train_tokens.len(), output_file_train);
}


pub fn preprocess_tinystories() {


    let input_file = "data/tinystories.json";
    let output_file_train = "data/tinystories_train.bin";
    let output_file_val = "data/tinystories_val.bin";

    let bpe = get_bpe_from_model("gpt2").unwrap();

    let input = File::open(input_file).expect("Failed to open input file");
    let reader = BufReader::new(input);

    let mut train_tokens = Vec::new();
    let mut val_tokens = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let json: serde_json::Value = serde_json::from_str(&line).expect("Failed to parse JSON");

        let story = json["content"].as_str().expect("Missing 'content' field in JSON");
        let tokens = bpe.encode_with_special_tokens(story);

        if train_tokens.len() < 32768 {
            val_tokens.extend_from_slice(&tokens);
        } else {
            train_tokens.extend_from_slice(&tokens);
        }
    }

    let mut output_train = File::create(output_file_train).expect("Failed to create train output file");
    let mut output_val = File::create(output_file_val).expect("Failed to create val output file");

    output_train.write_all(train_tokens.as_slice()).expect("Failed to write train tokens");
    output_val.write_all(val_tokens.as_slice()).expect("Failed to write val tokens");

    println!("Saved {} tokens to {}", val_tokens.len(), output_file_val);
    println!("Saved {} tokens to {}", train_tokens.len(), output_file_train);
}