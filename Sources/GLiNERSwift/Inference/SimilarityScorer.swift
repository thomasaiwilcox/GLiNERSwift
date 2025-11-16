import Foundation
import Accelerate

/// Computes similarity between span and label embeddings
public class SimilarityScorer {
    private let config: Configuration
    
    public init(config: Configuration = .default) {
        self.config = config
    }
    
    /// Compute similarity between a span embedding and a label embedding
    /// - Parameters:
    ///   - spanEmbedding: Span embedding vector
    ///   - labelEmbedding: Label embedding vector
    /// - Returns: Similarity score (0.0 to 1.0)
    public func computeSimilarity(
        spanEmbedding: [Float],
        labelEmbedding: [Float]
    ) -> Float {
        guard spanEmbedding.count == labelEmbedding.count else {
            return 0.0
        }
        
        switch config.similarityMetric {
        case .cosine:
            return cosineSimilarity(spanEmbedding, labelEmbedding)
        case .dotProduct:
            return dotProduct(spanEmbedding, labelEmbedding)
        }
    }
    
    /// Compute similarities between multiple spans and a single label
    /// - Parameters:
    ///   - spanEmbeddings: Array of span embeddings
    ///   - labelEmbedding: Label embedding
    /// - Returns: Array of similarity scores
    public func computeSimilarities(
        spanEmbeddings: [[Float]],
        labelEmbedding: [Float]
    ) -> [Float] {
        spanEmbeddings.map { computeSimilarity(spanEmbedding: $0, labelEmbedding: labelEmbedding) }
    }
    
    /// Score all spans against all labels
    /// - Parameters:
    ///   - spans: Array of spans
    ///   - labelEmbeddings: Dictionary of label embeddings
    /// - Returns: Dictionary mapping (spanIndex, label) to score
    public func scoreSpans(
        spans: [Span],
        labelEmbeddings: [String: [Float]]
    ) -> [SpanScore] {
        var scores: [SpanScore] = []
        
        for (spanIdx, span) in spans.enumerated() {
            for (label, labelEmbedding) in labelEmbeddings {
                let score = computeSimilarity(
                    spanEmbedding: span.embedding,
                    labelEmbedding: labelEmbedding
                )
                
                scores.append(SpanScore(
                    spanIndex: spanIdx,
                    label: label,
                    score: score,
                    span: span
                ))
            }
        }
        
        return scores
    }
    
    // MARK: - Private Methods
    
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let dim = vDSP_Length(a.count)
        
        // Compute dot product
        var dotProduct: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, dim)
        
        // Compute magnitude of a
        var magnitudeA: Float = 0
        vDSP_dotpr(a, 1, a, 1, &magnitudeA, dim)
        magnitudeA = sqrt(magnitudeA)
        
        // Compute magnitude of b
        var magnitudeB: Float = 0
        vDSP_dotpr(b, 1, b, 1, &magnitudeB, dim)
        magnitudeB = sqrt(magnitudeB)
        
        // Avoid division by zero
        guard magnitudeA > 0, magnitudeB > 0 else {
            return 0.0
        }
        
        return dotProduct / (magnitudeA * magnitudeB)
    }
    
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        let dim = vDSP_Length(a.count)
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, dim)
        
        // Normalize to 0-1 range using sigmoid-like scaling
        // (This is a simplification - actual implementation may vary)
        return max(0, min(1, result))
    }
}

/// Represents a scored span-label pair
public struct SpanScore {
    let spanIndex: Int
    let label: String
    let score: Float
    let span: Span
}
