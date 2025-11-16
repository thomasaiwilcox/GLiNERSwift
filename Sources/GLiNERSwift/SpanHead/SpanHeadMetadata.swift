import Foundation

struct SpanHeadMetadata: Decodable {
    struct LinearInfo: Decodable {
        let weight: String
        let bias: String
        let inFeatures: Int
        let outFeatures: Int

        private enum CodingKeys: String, CodingKey {
            case weight, bias
            case inFeatures = "in_features"
            case outFeatures = "out_features"
        }
    }

    struct ProjectionInfo: Decodable {
        let fc1: LinearInfo
        let fc2: LinearInfo
    }

    struct LayerGroup: Decodable {
        let projectStart: ProjectionInfo
        let projectEnd: ProjectionInfo
        let outProject: ProjectionInfo
        let promptProjection: ProjectionInfo

        private enum CodingKeys: String, CodingKey {
            case projectStart = "project_start"
            case projectEnd = "project_end"
            case outProject = "out_project"
            case promptProjection = "prompt_projection"
        }
    }

    struct LSTMWeights: Decodable {
        let weightIH: String
        let weightHH: String
        let bias: String

        private enum CodingKeys: String, CodingKey {
            case weightIH = "weight_ih"
            case weightHH = "weight_hh"
            case bias
        }
    }

    struct RNNInfo: Decodable {
        let inputSize: Int
        let hiddenSize: Int
        let numLayers: Int
        let bidirectional: Bool
        let forward: LSTMWeights
        let backward: LSTMWeights?

        private enum CodingKeys: String, CodingKey {
            case inputSize = "input_size"
            case hiddenSize = "hidden_size"
            case numLayers = "num_layers"
            case bidirectional
            case forward
            case backward
        }
    }

    struct SpecialTokens: Decodable {
        let flertToken: String
        let flertTokenIndex: Int
        let entTokenIndex: Int
        let sepTokenIndex: Int
        let maskTokenIndex: Int
        let clsTokenIndex: Int
        let baseSepTokenIndex: Int
        let padTokenIndex: Int

        private enum CodingKeys: String, CodingKey {
            case flertToken = "flert_token"
            case flertTokenIndex = "flert_token_index"
            case entTokenIndex = "ent_token_index"
            case sepTokenIndex = "sep_token_index"
            case maskTokenIndex = "mask_token_index"
            case clsTokenIndex = "cls_token_index"
            case baseSepTokenIndex = "base_sep_token_index"
            case padTokenIndex = "pad_token_index"
        }
    }

    let model: String
    let hiddenSize: Int
    let maxWidth: Int
    let classTokenIndex: Int
    let entToken: String
    let sepToken: String
    let specialTokens: SpecialTokens
    let layers: LayerGroup
    let rnn: RNNInfo

    private enum CodingKeys: String, CodingKey {
        case model
        case hiddenSize = "hidden_size"
        case maxWidth = "max_width"
        case classTokenIndex = "class_token_index"
        case entToken = "ent_token"
        case sepToken = "sep_token"
        case specialTokens = "special_tokens"
        case layers
        case rnn
    }
}
