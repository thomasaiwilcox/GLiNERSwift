import Foundation

/// Fixture data structures matching Python output
struct TestFixtures: Codable {
    let metadata: Metadata
    let testCases: [TestCase]

    enum CodingKeys: String, CodingKey {
        case metadata
        case testCases = "test_cases"
    }
    
    struct Metadata: Codable {
        let model: String
        let tolerance: Tolerance
        let generatorVersion: String

        enum CodingKeys: String, CodingKey {
            case model
            case tolerance
            case generatorVersion = "generator_version"
        }
        
        struct Tolerance: Codable {
            let tokenIds: Int
            let attentionMask: Int
            let hiddenStates: Float
            let entityScores: Float
            
            enum CodingKeys: String, CodingKey {
                case tokenIds = "token_ids"
                case attentionMask = "attention_mask"
                case hiddenStates = "hidden_states"
                case entityScores = "entity_scores"
            }
        }
    }
    
    struct TestCase: Codable {
        let id: String
        let text: String
        let labels: [String]
        let threshold: Float
        let tokenizerOutput: TokenizerOutput
        let encoderOutput: EncoderOutput
        let entities: [EntityFixture]
        
        enum CodingKeys: String, CodingKey {
            case id, text, labels, threshold
            case tokenizerOutput = "tokenizer_output"
            case encoderOutput = "encoder_output"
            case entities
        }
    }
    
    struct TokenizerOutput: Codable {
        let inputIds: [Int32]
        let attentionMask: [Int32]
        let tokens: [String]
        
        enum CodingKeys: String, CodingKey {
            case inputIds = "input_ids"
            case attentionMask = "attention_mask"
            case tokens
        }
    }
    
    struct EncoderOutput: Codable {
        let hiddenStates: [[Float]]?
        let shape: [Int]?
        
        enum CodingKeys: String, CodingKey {
            case hiddenStates = "hidden_states"
            case shape
        }
    }
    
    struct EntityFixture: Codable {
        let text: String
        let label: String
        let score: Float
        let start: Int
        let end: Int
    }
    
    static func load() throws -> TestFixtures {
        guard let fixtureURL = Bundle.module.url(
            forResource: "python_outputs",
            withExtension: "json",
            subdirectory: "Fixtures"
        ) else {
            throw TestError.fixturesNotFound(
                "python_outputs.json not found. Run Scripts/generate_test_fixtures.py to generate it."
            )
        }
        
        let data = try Data(contentsOf: fixtureURL)
        let decoder = JSONDecoder()
        return try decoder.decode(TestFixtures.self, from: data)
    }
}

enum TestError: LocalizedError {
    case fixturesNotFound(String)
    
    var errorDescription: String? {
        switch self {
        case .fixturesNotFound(let message):
            return message
        }
    }
}
