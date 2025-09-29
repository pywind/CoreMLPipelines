import CoreML
import Foundation
@preconcurrency import Hub
import Tokenizers

// MARK: - TextGenerationPipeline

public class TextGenerationPipeline {
    // MARK: Lifecycle

    public convenience init(
        model: Model,
        prewarm: Bool = true,
        hubAPI: HubApi = .shared
    ) async throws {
        try await self.init(
            modelName: model.rawValue,
            prewarm: prewarm,
            hubAPI: hubAPI
        )
    }

    public init(
        modelName: String,
        prewarm: Bool = true,
        hubAPI: HubApi = .shared
    ) async throws {
        async let _tokenizer = try await AutoTokenizer.from(pretrained: modelName, hubApi: hubAPI)
        async let _model = try await CausalLMModel.from(pretrained: modelName, hubAPI: hubAPI)

        (tokenizer, model) = try await (_tokenizer, _model)
        kvCache = model.makeKVCache()

        sampler = GreedySampler()

        if prewarm { try await self.prewarm() }
    }

    public init(
        modelURL: URL,
        tokenizerName: String,
        prewarm: Bool = true,
        hubAPI: HubApi = .shared
    ) async throws {
        async let _tokenizer = try await AutoTokenizer.from(pretrained: tokenizerName, hubApi: hubAPI)
        async let _model = try await CausalLMModel.load(contentsOf: modelURL)

        (tokenizer, model) = try await (_tokenizer, _model)
        kvCache = model.makeKVCache()

        sampler = GreedySampler()

        if prewarm { try await self.prewarm() }
    }

    // MARK: Public

    public func prewarm() async throws {
        _ = try await model(
            inputIDs: MLShapedArray<Int32>(repeating: 0, shape: [1, 2]),
            causalMask: MLShapedArray<Float16>(repeating: 0, shape: [1, 1, 1, 2]),
            kvCache: model.makeKVCache()
        )
    }

    // TODO: - assert/check prompt starts with last prompt
    public func generate(
        prompt: String,
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            prompt: .text(prompt),
            maxNewTokens: maxNewTokens
        )
    }

    // TODO: - assert/check messages starts with last messages
    public func generate(
        messages: [[String: String]],
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            prompt: .messages(messages),
            maxNewTokens: maxNewTokens
        )
    }

    public func callAsFunction(
        prompt: String,
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            prompt: prompt,
            maxNewTokens: maxNewTokens
        )
    }

    public func callAsFunction(
        messages: [[String: String]],
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            messages: messages,
            maxNewTokens: maxNewTokens
        )
    }

    // MARK: Private

    private let tokenizer: any Tokenizer
    private let model: CausalLMModel
    private let sampler: Sampler
    private let kvCache: CausalLMModel.KVCache
}

private extension TextGenerationPipeline {
    enum Prompt {
        case text(String)
        case messages([[String: String]])
    }

    func generate(
        prompt: Prompt,
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        struct ModelContainer: @unchecked Sendable {
            let model: CausalLMModel
            let kvCache: CausalLMModel.KVCache
            let tokenizer: Tokenizer
            let sampler: Sampler
        }
        let modelContainer = ModelContainer(
            model: model,
            kvCache: kvCache,
            tokenizer: tokenizer,
            sampler: sampler
        )
        return AsyncThrowingStream<String, Error> { continuation in
            Task {
                do {
                    let (model, kvCache, tokenizer, sampler) = (
                        modelContainer.model,
                        modelContainer.kvCache,
                        modelContainer.tokenizer,
                        modelContainer.sampler
                    )
                    var tokens: [Int] = []
                    var tokensGenerated = 0
                    while true {
                        guard tokensGenerated < maxNewTokens ?? .max,
                              tokens.last == nil || tokens.last != tokenizer.eosTokenId else {
                            continuation.finish()
                            return
                        }

                        try Task.checkCancellation()

                        if tokens.isEmpty {
                            tokens = try {
                                switch prompt {
                                case let .text(text):
                                    tokenizer.encode(text: text)
                                case let .messages(messages):
                                    try tokenizer.applyChatTemplate(
                                        messages: messages
                                    )
                                }
                            }()

                            let inputIDs = MLShapedArray(scalars: tokens.map(Int32.init), shape: [1, tokens.count])

                            let causalMask = await MLTensor.causalMask(size: tokens.count)
                                .shapedArray(of: Float16.self)

                            let logits = try await model(
                                inputIDs: inputIDs,
                                causalMask: causalMask,
                                kvCache: kvCache
                            )

                            let predictedToken = await sampler.sample(logits[0, logits.shape[1] - 1])

                            tokens.append(predictedToken)
                            tokensGenerated += 1

                            let decodedText = tokenizer.decode(tokens: [predictedToken], skipSpecialTokens: true)
                            continuation.yield(decodedText)
                        } else {
                            let inputIDs = MLShapedArray(scalars: [Int32(tokens.last!)], shape: [1, 1])
                            let causalMask = MLShapedArray<Float16>(
                                repeating: 0,
                                shape: [1, 1, 1, tokens.count]
                            )

                            let logits = try await model(
                                inputIDs: inputIDs,
                                causalMask: causalMask,
                                kvCache: kvCache
                            )

                            let predictedToken = await sampler.sample(logits[0, logits.shape[1] - 1])

                            tokens.append(predictedToken)
                            tokensGenerated += 1

                            let decodedText = tokenizer.decode(tokens: [predictedToken], skipSpecialTokens: true)
                            continuation.yield(decodedText)
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
