import ArgumentParser
import CoreMLPipelines
import Foundation
import Hub

// MARK: - ProfileCLI

struct ProfileCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile",
        abstract: "Profile CoreMLPipelines"
    )

    func run() async throws {
        let pipeline = try await TextGenerationPipeline(model: .lgai_exaone_4_0_1_2B_4bit)
        var messages: [[String: String]] = []
        messages.append([
            "role": "user",
            "content": "How do I make pizza?"
        ])
        let stream1 = pipeline(messages: messages)
        let response = try await stream1.reduce("", +)
        messages.append([
            "role": "assistant",
            "content": response
        ])
        messages.append([
            "role": "user",
            "content": "What are some good toppings?"
        ])
        let stream2 = pipeline(messages: messages)
        _ = try await stream2.reduce("", +)
    }
}
