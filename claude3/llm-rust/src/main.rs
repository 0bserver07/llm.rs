use std::env;
use llm_rust::prepro;
use llm_rust::train_gpt2;
use llm_rust::test_gpt2;

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // Preprocess the dataset
    if args.len() > 1 && args[1] == "prepro" {
        let dataset = &args[2];
        match dataset.as_str() {
            "tinyshakespeare" => prepro::preprocess_tinyshakespeare(),
            "tinystories" => prepro::preprocess_tinystories(),
            _ => println!("Invalid dataset. Available options: tinyshakespeare, tinystories"),
        }
    }
    // Train the GPT-2 model
    else if args.len() > 1 && args[1] == "train" {
        let dataset = &args[2];
        let num_epochs = if args.len() > 3 { args[3].parse().unwrap() } else { 10 };
        train_gpt2::train(dataset, num_epochs);
    }
    // Test the GPT-2 model
    else if args.len() > 1 && args[1] == "test" {
        test_gpt2::test();
    }
    // Display usage instructions
    else {
        println!("Usage:");
        println!("  cargo run prepro [dataset]");
        println!("  cargo run train [dataset] [num_epochs]");
        println!("  cargo run test");
    }
}