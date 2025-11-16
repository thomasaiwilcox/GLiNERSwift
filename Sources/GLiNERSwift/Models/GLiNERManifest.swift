import Foundation

/// Loads metadata emitted by `Scripts/convert_to_coreml.py` so the Swift runtime can locate
/// GLiNER2 Core ML packages, tokenizer files, and shape constraints.
struct GLiNERManifest: Decodable {
    struct ArtifactPaths: Decodable {
        let encoder: String
        let spanRep: String
        let classifier: String
        let countPredictor: String
        let countEmbed: String

        enum CodingKeys: String, CodingKey {
            case encoder
            case spanRep = "span_rep"
            case classifier
            case countPredictor = "count_predictor"
            case countEmbed = "count_embed"
        }
    }

    struct ResolvedResources {
        let encoderURL: URL
        let spanRepURL: URL
        let classifierURL: URL
        let countPredictorURL: URL
        let countEmbedURL: URL
        let tokenizerURL: URL?
    }

    let modelId: String
    let maxSeqLen: Int
    let maxSchemaTokens: Int
    let maxWidth: Int
    let hiddenSize: Int
    let countingLayer: String
    let maxCount: Int
    let precision: String
    let artifacts: ArtifactPaths
    let tokenizerDir: String?

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case maxSeqLen = "max_seq_len"
        case maxSchemaTokens = "max_schema_tokens"
        case maxWidth = "max_width"
        case hiddenSize = "hidden_size"
        case countingLayer = "counting_layer"
        case maxCount = "max_count"
        case precision
        case artifacts
        case tokenizerDir = "tokenizer_dir"
    }
}

extension GLiNERManifest {
    static func load(from url: URL) throws -> GLiNERManifest {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .useDefaultKeys
        return try decoder.decode(GLiNERManifest.self, from: data)
    }

    func resolveResources(relativeTo manifestURL: URL) throws -> ResolvedResources {
        let baseDirectory = manifestURL.deletingLastPathComponent()
        func resolve(_ path: String) -> URL {
            if path.hasPrefix("/") {
                return URL(fileURLWithPath: path).standardizedFileURL
            }
            return baseDirectory.appendingPathComponent(path).standardizedFileURL
        }

        let encoderURL = resolve(artifacts.encoder)
        let spanRepURL = resolve(artifacts.spanRep)
        let classifierURL = resolve(artifacts.classifier)
        let countPredictorURL = resolve(artifacts.countPredictor)
        let countEmbedURL = resolve(artifacts.countEmbed)
        let tokenizerURL = tokenizerDir.map(resolve)

        return ResolvedResources(
            encoderURL: encoderURL,
            spanRepURL: spanRepURL,
            classifierURL: classifierURL,
            countPredictorURL: countPredictorURL,
            countEmbedURL: countEmbedURL,
            tokenizerURL: tokenizerURL
        )
    }
}
