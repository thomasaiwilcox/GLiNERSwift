import Foundation

/// Represents a named entity extracted from text.
public struct Entity: Equatable, Hashable {
    /// The text content of the entity
    public let text: String
    
    /// The label/type of the entity (e.g., "person", "organization")
    public let label: String
    
    /// Confidence score (0.0 to 1.0)
    public let score: Float
    
    /// Start character offset
    public let start: Int
    
    /// End character offset
    public let end: Int
    
    public init(text: String, label: String, score: Float, start: Int, end: Int) {
        self.text = text
        self.label = label
        self.score = score
        self.start = start
        self.end = end
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(text)
        hasher.combine(label)
        hasher.combine(start)
        hasher.combine(end)
    }
}

extension Entity: CustomStringConvertible {
    public var description: String {
        "\(label): \"\(text)\" [\(start)-\(end)] (score: \(score))"
    }
}
