import Foundation

/// Represents the combined schema/text encoding used by the GLiNER2 Core ML pipeline.
public struct GLiNER2SchemaEncoding {
    public enum Segment: String {
        case schema
        case separator
        case text
    }

    public struct Mapping {
        public let segment: Segment
        public let originalIndex: Int
        public let schemaIndex: Int
    }

    public struct PromptLocation {
        public enum Kind {
            case prompt
            case entity
            case relation
            case classification
            case list
        }

        public let kind: Kind
        public let schemaIndex: Int
        public let originalIndex: Int
        public let subwordRange: Range<Int>
    }

    public let schemaTokensList: [[String]]
    public let textTokens: [String]
    public let inputIds: [Int32]
    public let attentionMask: [Int32]
    public let mappedIndices: [Mapping]
    public let subwordTokens: [String]
    public let textWordRanges: [Range<String.Index>]
    public let startTokenCharacterMap: [Int]
    public let endTokenCharacterMap: [Int]
    public let maxSpanWidth: Int
    public let spanIndices: [[Int32]]
    public let spanMask: [[Float]]
    public let promptLocations: [PromptLocation]
    public let entityLabels: [String]
    public let sequenceCapacity: Int
}

extension GLiNER2SchemaEncoding.PromptLocation.Kind {
    init?(token: String, configuration: GLiNER2PromptConfiguration) {
        switch token {
        case configuration.promptToken:
            self = .prompt
        case configuration.entityToken:
            self = .entity
        case configuration.relationToken:
            self = .relation
        case configuration.classificationToken:
            self = .classification
        case configuration.listToken:
            self = .list
        default:
            return nil
        }
    }
}

extension GLiNER2SchemaEncoding {
    var textStartOriginalIndex: Int {
        return schemaTokenCount + 1
    }

    private var schemaTokenCount: Int {
        let schemaTokens = schemaTokensList.reduce(0) { $0 + $1.count }
        let separatorTokens = max(0, schemaTokensList.count - 1)
        return schemaTokens + separatorTokens
    }
}
