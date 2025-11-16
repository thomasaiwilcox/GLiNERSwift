import Foundation
import Accelerate

/// Computes similarity between span and label embeddings
public class SimilarityScorer {
    private let config: Configuration
    // Cache for label embedding norms to avoid recomputation
    private var labelNormCache: [String: Float] = [:]
    
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
        guard spanEmbedding.count == labelEmbedding.count, !spanEmbedding.isEmpty else {
            return 0.0
        }
        
        switch config.similarityMetric {
        case .cosine:
            return cosineSimilarity(spanEmbedding, labelEmbedding)
        case .dotProduct:
            return dotProduct(spanEmbedding, labelEmbedding)
        }
    }
    
    /// Compute similarities between multiple spans and a single label (vectorized)
    /// - Parameters:
    ///   - spanEmbeddings: Array of span embeddings
    ///   - labelEmbedding: Label embedding
    /// - Returns: Array of similarity scores
    public func computeSimilarities(
        spanEmbeddings: [[Float]],
        labelEmbedding: [Float]
    ) -> [Float] {
        guard !spanEmbeddings.isEmpty, !labelEmbedding.isEmpty else {
            return []
        }
        
        // For batch processing, flatten and use BLAS matrix-vector multiplication
        let numSpans = spanEmbeddings.count
        let embeddingDim = labelEmbedding.count
        
        // Verify all embeddings have the same dimension
        guard spanEmbeddings.allSatisfy({ $0.count == embeddingDim }) else {
            // Fallback to sequential processing if dimensions don't match
            return spanEmbeddings.map { computeSimilarity(spanEmbedding: $0, labelEmbedding: labelEmbedding) }
        }
        
        switch config.similarityMetric {
        case .cosine:
            return batchCosineSimilarity(spanEmbeddings: spanEmbeddings, labelEmbedding: labelEmbedding)
        case .dotProduct:
            return batchDotProduct(spanEmbeddings: spanEmbeddings, labelEmbedding: labelEmbedding)
        }
    }
    
    /// Score all spans against all labels with optimized batch processing
    /// - Parameters:
    ///   - spans: Array of spans
    ///   - labelEmbeddings: Dictionary of label embeddings
    /// - Returns: Array of scored span-label pairs
    public func scoreSpans(
        spans: [Span],
        labelEmbeddings: [String: [Float]]
    ) -> [SpanScore] {
        guard !spans.isEmpty, !labelEmbeddings.isEmpty else {
            return []
        }
        
        // Pre-compute label norms for cosine similarity
        if config.similarityMetric == .cosine {
            for (label, embedding) in labelEmbeddings {
                if labelNormCache[label] == nil {
                    labelNormCache[label] = computeNorm(embedding)
                }
            }
        }
        
        var scores: [SpanScore] = []
        scores.reserveCapacity(spans.count * labelEmbeddings.count)
        
        // Process each label with all spans in batch
        for (label, labelEmbedding) in labelEmbeddings {
            let spanEmbeddings = spans.map { $0.embedding }
            let batchScores = computeSimilarities(spanEmbeddings: spanEmbeddings, labelEmbedding: labelEmbedding)
            
            for (spanIdx, score) in batchScores.enumerated() {
                scores.append(SpanScore(
                    spanIndex: spanIdx,
                    label: label,
                    score: score,
                    span: spans[spanIdx]
                ))
            }
        }
        
        return scores
    }
    
    // MARK: - Private Methods
    
    private func computeNorm(_ vector: [Float]) -> Float {
        let dim = vDSP_Length(vector.count)
        var magnitude: Float = 0
        vDSP_dotpr(vector, 1, vector, 1, &magnitude, dim)
        return sqrt(magnitude)
    }
    
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
    
    /// Batch compute dot products between multiple spans and one label
    private func batchDotProduct(spanEmbeddings: [[Float]], labelEmbedding: [Float]) -> [Float] {
        let numSpans = spanEmbeddings.count
        let embeddingDim = labelEmbedding.count
        
        var scores = [Float](repeating: 0, count: numSpans)
        let flatSpans = spanEmbeddings.flatMap { $0 }
        
        // Matrix-vector multiplication: scores = spanMatrix * labelVector
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            Int32(numSpans),
            Int32(embeddingDim),
            1.0,
            flatSpans,
            Int32(embeddingDim),
            labelEmbedding,
            1,
            0.0,
            &scores,
            1
        )
        
        // Normalize scores to 0-1 range
        for i in 0..<numSpans {
            scores[i] = max(0, min(1, scores[i]))
        }
        
        return scores
    }
    
    /// Batch compute cosine similarities between multiple spans and one label
    private func batchCosineSimilarity(spanEmbeddings: [[Float]], labelEmbedding: [Float]) -> [Float] {
        let numSpans = spanEmbeddings.count
        let embeddingDim = labelEmbedding.count
        
        // Pre-compute label norm
        let labelNorm = computeNorm(labelEmbedding)
        guard labelNorm > 0 else {
            return [Float](repeating: 0, count: numSpans)
        }
        
        var scores = [Float](repeating: 0, count: numSpans)
        let flatSpans = spanEmbeddings.flatMap { $0 }
        
        // Compute dot products in batch
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            Int32(numSpans),
            Int32(embeddingDim),
            1.0,
            flatSpans,
            Int32(embeddingDim),
            labelEmbedding,
            1,
            0.0,
            &scores,
            1
        )
        
        // Compute span norms and normalize
        for i in 0..<numSpans {
            let spanNorm = computeNorm(spanEmbeddings[i])
            if spanNorm > 0 {
                scores[i] = scores[i] / (spanNorm * labelNorm)
            } else {
                scores[i] = 0.0
            }
        }
        
        return scores
    }
}

/// Represents a scored span-label pair
public struct SpanScore {
    let spanIndex: Int
    let label: String
    let score: Float
    let span: Span
}
