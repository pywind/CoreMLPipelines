import ArgumentParser
import CoreMLPipelines
import Foundation
import Hub
import SwiftTUI

// MARK: - ChatCLI

struct ChatCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Chat using CoreMLPipelines"
    )

    @Option var model: String = TextGenerationPipeline.Model.lgai_exaone_4_0_1_2B_4bit.rawValue
    @Option var local: String?

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

        var messages: [[String: String]] = [
            [
                "role": "system",
                "content": "You are a helpful assistant."
            ]
        ]
        while true {
            let prompt = CommandLine.askForText()
            guard prompt.localizedCaseInsensitiveCompare("exit") != .orderedSame,
                  prompt.localizedCaseInsensitiveCompare("quit") != .orderedSame else { break }
            messages.append([
                "role": "user",
                "content": prompt
            ])
            let stream = pipeline(messages: messages)
            var response = ""
            for try await text in stream {
                response += text
                print(text, terminator: "")
                fflush(stdout)
            }
            messages.append([
                "role": "assistant",
                "content": response
            ])
            print("")
        }
    }
}
