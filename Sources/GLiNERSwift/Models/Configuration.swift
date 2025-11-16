import Foundation

/// Configuration for GLiNER entity extraction.
public struct Configuration {
    /// Method for pooling token embeddings into span representations
    public enum PoolingMethod {
        case mean       // Average of token embeddings
        case max        // Max pooling of token embeddings
        case concat     // Concatenate start, end, and pooled embeddings
    }
    
    /// Similarity metric for scoring span-label compatibility
    public enum SimilarityMetric {
        case cosine     // Cosine similarity
        case dotProduct // Dot product
    }
    
    /// Pooling method for span representations
    public var poolingMethod: PoolingMethod
    
    /// Similarity metric for scoring
    public var similarityMetric: SimilarityMetric
    
    /// Confidence threshold for entity extraction
    public var threshold: Float
    
    /// Maximum sequence length in tokens
    public var maxSequenceLength: Int
    
    /// Stride length for overlapping windows
    public var strideLength: Int
    
    /// Maximum span length to consider
    public var maxSpanLength: Int
    
    /// Hidden dimension size from the encoder
    public var hiddenDim: Int
    
    /// NMS threshold for deduplication
    public var nmsThreshold: Float

    /// Enable the GLiNER2 Core ML span/classifier/count pipeline (manifest builds only)
    public var useGLiNER2Pipeline: Bool
    
    public init(
        poolingMethod: PoolingMethod = .mean,
        similarityMetric: SimilarityMetric = .cosine,
        threshold: Float = 0.3,
        maxSequenceLength: Int = 384,
        strideLength: Int = 192,
        maxSpanLength: Int = 8,
        hiddenDim: Int = 512,
        nmsThreshold: Float = 0.5,
        useGLiNER2Pipeline: Bool = true
    ) {
        self.poolingMethod = poolingMethod
        self.similarityMetric = similarityMetric
        self.threshold = threshold
        self.maxSequenceLength = maxSequenceLength
        self.strideLength = strideLength
        self.maxSpanLength = maxSpanLength
        self.hiddenDim = hiddenDim
        self.nmsThreshold = nmsThreshold
        self.useGLiNER2Pipeline = useGLiNER2Pipeline
    }
    
    /// Default configuration
    public static let `default` = Configuration()
}
