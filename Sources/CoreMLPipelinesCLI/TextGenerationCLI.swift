import ArgumentParser
import CoreMLPipelines
import Foundation
import Hub
import SwiftTUI

// MARK: - TextGenerationCLI

struct TextGenerationCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate-text",
        abstract: "Generate text using CoreMLPipelines"
    )

    @Option var model: String = TextGenerationPipeline.Model.lgai_exaone_4_0_1_2B_4bit.rawValue
    @Option var local: String?
    @Argument var prompt: String = "Hello"
    @Option var maxNewTokens: Int?

    func run() async throws {
        ActivityIndicator.start(title: "Loading \(model.brightGreen)…")
        let pipeline: TextGenerationPipeline
        if let localPath = local {
            let modelURL = URL(fileURLWithPath: localPath)
            pipeline = try await TextGenerationPipeline(modelURL: modelURL, tokenizerName: model)
        } else {
            pipeline = try await TextGenerationPipeline(modelName: model)
        }
        ActivityIndicator.stop()
        CommandLine.success("Model loaded!")

        let stream = pipeline(
            messages: [[
                "role": "user",
                "content": prompt
            ]],
            maxNewTokens: maxNewTokens
        )

        // Performance metrics tracking
        let startTime = CFAbsoluteTimeGetCurrent()
        var tokenCount = 0
        var firstTokenTime: CFAbsoluteTime?

        for try await text in stream {
            print(text, terminator: "")
            fflush(stdout)
            tokenCount += 1

            // Track time to first token
            if firstTokenTime == nil {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
            }
        }
        print("")

        let endTime = CFAbsoluteTimeGetCurrent()
        let processingTime = endTime - startTime
        let tokensPerSecond = Double(tokenCount) / processingTime
        let timeToFirstToken = firstTokenTime.map { $0 - startTime }

        // Display performance metrics
        print("")
        print("Performance Metrics:")
        print("  Tokens generated: \(tokenCount)")
        print("  Processing time: \(String(format: "%.3f", processingTime)) seconds")
        print("  Tokens per second: \(String(format: "%.2f", tokensPerSecond))")
        if let ttft = timeToFirstToken {
            print("  Time to first token: \(String(format: "%.3f", ttft)) seconds")
        }
    }
}
