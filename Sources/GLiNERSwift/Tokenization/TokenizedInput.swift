import Foundation

/// Result of tokenization
public struct TokenizedInput {
    /// Token IDs
    public let inputIds: [Int32]
    
    /// Attention mask (1 for real tokens, 0 for padding)
    public let attentionMask: [Int32]
    
    /// Original tokens (for debugging)
    public let tokens: [String]
    
    /// Mapping from token index to character range in original text
    public let tokenToCharMap: [Range<String.Index>]?
    
    public init(
        inputIds: [Int32],
        attentionMask: [Int32],
        tokens: [String],
        tokenToCharMap: [Range<String.Index>]? = nil
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.tokens = tokens
        self.tokenToCharMap = tokenToCharMap
    }
}

/// A chunk of tokenized text (for sliding window processing)
public struct TokenizedChunk {
    /// Token IDs for this chunk
    public let inputIds: [Int32]
    
    /// Attention mask for this chunk
    public let attentionMask: [Int32]
    
    /// Original tokens
    public let tokens: [String]
    
    /// Character range in the original text that this chunk covers
    public let textRange: Range<String.Index>
    
    /// Token offset in the original sequence (for merging results)
    public let tokenOffset: Int
    
    public init(
        inputIds: [Int32],
        attentionMask: [Int32],
        tokens: [String],
        textRange: Range<String.Index>,
        tokenOffset: Int
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.tokens = tokens
        self.textRange = textRange
        self.tokenOffset = tokenOffset
    }
}
