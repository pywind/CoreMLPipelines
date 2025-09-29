# CoreMLPipelines

<p align="center">
    <img src="https://img.shields.io/badge/Swift-6.0-orange" alt="Swift 6.0" />
    <img src="https://img.shields.io/badge/macOS-15.0-blue" alt="macOS 15.0+" />
    <img src="https://img.shields.io/badge/iOS-18.0-blue" alt="iOS 18.0+" />
    <img src="https://img.shields.io/badge/License-CC0--1.0-lightgrey" alt="CC0 1.0 License" />
    <a href="https://github.com/apple/swift-package-manager"><img src="https://img.shields.io/badge/Swift%20Package%20Manager-compatible-brightgreen" alt="Swift Package Manager" /></a>
</p>

<p align="center">
    <strong>Run Large Language Models on Apple Silicon with Core ML</strong>
</p>

CoreMLPipelines is an experimental Swift library for running pretrained [Core ML](https://developer.apple.com/documentation/coreml) models to perform different AI tasks. It provides high-performance inference on Apple Silicon devices with minimal memory usage.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Architecture](#architecture)
- [CLI Usage](#cli-usage)
- [Model Conversion](#model-conversion)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Features

- 🚀 **High Performance**: Optimized Core ML inference on Apple Silicon
- 💾 **Memory Efficient**: 4-bit and 8-bit quantization support
- 🔄 **Streaming**: Real-time text generation with async streams
- 🛠️ **CLI Tools**: Command-line interface for text generation and chat
- 🔧 **Model Conversion**: Python tools to convert Hugging Face models to Core ML
- 📱 **Cross-Platform**: Supports iOS 18+ and macOS 15+

### Supported Pipelines

- **Text Generation**: Generate text using causal language models
- **Chat**: Interactive conversational AI

## Installation

### Swift Package

Add it to your Xcode project via File > Add Package Dependencies.

Or clone and build locally:

```bash
git clone https://github.com/pywind/CoreMLPipelines.git
cd CoreMLPipelines
swift build -c release
cp .build/release/coremlpipelines-cli /usr/local/bin/
```

### Python Tools

The Python conversion tools require Python 3.11+ and can be installed using uv:

```bash
cd coremlpipelinestools
uv sync
```

If you want to upload model to your HuggingFace Hub, please create `.env` file and put:

```bash
HF_TOKEN=hf_your_token
```

Please read this [coreml README](/coremlpipelinestools/README.md) for more details.

## Quick Start

### Basic Text Generation

```swift
import CoreMLPipelines

// Create a text generation pipeline
let pipeline = try await TextGenerationPipeline(model: .llama_3_2_1B_Instruct_4bit)

// Generate text with streaming
let stream = pipeline(
    messages: [[
        "role": "user",
        "content": "Write a haiku about programming"
    ]]
)

for try await text in stream {
    print(text, terminator: "")
}
```

### Advanced Usage

```swift
import CoreMLPipelines

let pipeline = try await TextGenerationPipeline(model: .qwen2_5_0_5B_Instruct_4bit)

// Configure generation parameters
let config = GenerationConfig(
    maxNewTokens: 100,
    temperature: 0.7,
    topP: 0.9,
    repetitionPenalty: 1.1
)

let stream = pipeline(
    messages: [["role": "user", "content": "Explain quantum computing simply"]],
    generationConfig: config
)

var fullResponse = ""
for try await text in stream {
    fullResponse += text
    print(text, terminator: "")
}
```

### Custom Model

```swift
import CoreMLPipelines

// Use any Hugging Face model (must be converted to Core ML first)
let pipeline = try await TextGenerationPipeline(
    model: "your-username/your-coreml-model"
)
```

## Supported Models

CoreMLPipelines supports various quantized language models optimized for Apple Silicon:

### Llama Models
- `llama_3_2_1B_Instruct_4bit` - Meta's Llama 3.2 1B parameter model (4-bit quantized)

### Qwen Models
- `qwen2_5_0_5B_Instruct_4bit` - Alibaba's Qwen2.5 0.5B model (4-bit quantized)
- `qwen2_5_Coder_0_5B_Instruct_4bit` - Qwen2.5 Coder 0.5B for code generation (4-bit quantized)

### SmolLM Models
- `smolLM2_135M_Instruct_4bit` - SmolLM2 135M model (4-bit quantized)
- `smolLM2_135M_Instruct_8bit` - SmolLM2 135M model (8-bit quantized)

### EXAONE Models
- `lgai_exaone_4_0_1_2B_4bit` - LG AI's EXAONE 4.0 1.2B model (4-bit quantized)

> **Note**: Models/tokenizers and chat_template are automatically downloaded from Hugging Face on first use. Ensure you have a stable internet connection.

## Architecture

### Core Components

```text
CoreMLPipelines/
├── Models/           # Model definitions and configurations
├── Pipelines/        # Pipeline implementations
│   ├── TextGenerationPipeline.swift
│   └── TextGenerationPipeline+Models.swift
├── Samplers/         # Token sampling strategies
│   ├── GreedySampler.swift
│   └── Sampler.swift
└── Extensions/       # Core ML tensor utilities
```

### Key Features

- **Unified API**: Consistent interface across different model architectures
- **Memory Management**: Efficient memory usage with Core ML's MLModel
- **Async/Await**: Modern Swift concurrency support
- **Streaming**: Real-time token generation with AsyncSequence
- **Type Safety**: Strong typing with Swift's type system

## CLI Usage

The command-line interface provides convenient tools for testing and development.

![CLI Demo](https://github.com/user-attachments/assets/4b72fa50-7e47-4171-9d98-791661d25dcc "CoreMLPipelines CLI Demo")

### Generate Text

```bash
coremlpipelines-cli generate-text --model finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit "Hello, world!" --max-new-tokens 50
```

**Options:**
- `--model <model>`: Hugging Face model repository ID
- `--max-new-tokens <int>`: Maximum number of tokens to generate (default: 100)
- `<prompt>`: Text prompt (default: "Hello")

### Interactive Chat

```bash
coremlpipelines-cli chat --model finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit
```

Start an interactive chat session with the specified model.

### Profiling

```bash
coremlpipelines-cli profile --model finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit
```

Profile model performance and memory usage.

## Model Conversion

Convert Hugging Face models to Core ML format using the Python tools:

```bash
cd coremlpipelinestools
uv run convert_causal_llm.py --model microsoft/DialoGPT-medium --quantize --half --compile --context-size 512 --batch-size 1
```

**Key Options:**

- `--model`: Hugging Face model ID
- `--quantize`: Apply 4-bit linear quantization
- `--half`: Load model in float16 precision
- `--compile`: Save as optimized .mlmodelc format
- `--context-size`: Maximum context length
- `--batch-size`: Batch size for inference
- `--upload`: Upload converted model to Hugging Face

### Example Conversions

```bash
# Convert Llama model with 4-bit quantization
uv run convert_causal_llm.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --quantize \
  --half \
  --compile \
  --context-size 2048 \
  --batch-size 1 \
  --upload

# Convert SmolLM model with 8-bit quantization
uv run convert_causal_llm.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --half \
  --compile \
  --context-size 1024 \
  --batch-size 1
```

## Requirements

### System Requirements

- **macOS**: 15.0 or later
- **iOS**: 18.0 or later
- **Xcode**: 16.0 or later
- **Swift**: 6.0 or later

### Dependencies

- [swift-transformers](https://github.com/huggingface/swift-transformers)
- [swift-argument-parser](https://github.com/apple/swift-argument-parser)
- [SwiftTUI](https://github.com/finnvoor/SwiftTUI)

### Python Tools Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Core ML Tools
- Transformers library

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/pywind/CoreMLPipelines.git
   cd CoreMLPipelines
   ```

2. Open in Xcode:

   ```bash
   open Package.swift
   ```

3. Run tests:

   ```bash
   swift test
   ```

4. Build CLI tool:

   ```bash
   swift build -c release --product coremlpipelines-cli
   ```

### Code Style

This project follows Swift's official style guidelines. Use `swiftformat` to format code:

```bash
swiftformat .
```

## License

This project is licensed under the CC0 1.0 Universal License - see the [LICENSE](LICENSE) file for details.

---
This project come from this repo: [finnvoor](https://github.com/finnvoor)
<p align="center">
    <strong>Become a sponsor to https://github.com/finnvoor</strong>
</p>
