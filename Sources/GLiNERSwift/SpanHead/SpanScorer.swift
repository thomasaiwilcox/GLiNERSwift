import Foundation

struct SpanScores {
    let values: [[[Float]]]
}

/// Prepares span head inputs and delegates scoring to the Core ML model.
final class SpanScorer {
    private let metadata: SpanHeadMetadata
    private let scoringModel: GLiNERSpanScoringModel

    init(metadata: SpanHeadMetadata, scoringModel: GLiNERSpanScoringModel) {
        self.metadata = metadata
        self.scoringModel = scoringModel
    }

    func score(hiddenStates: [[Float]], encoding: PromptEncoding, labelCount: Int) async throws -> SpanScores {
        guard hiddenStates.count == encoding.sequenceLength else {
            throw GLiNERError.encodingError("Hidden state count \(hiddenStates.count) does not match token count \(encoding.sequenceLength)")
        }
        guard labelCount > 0 else {
            return SpanScores(values: [])
        }
        guard !encoding.textWordRanges.isEmpty else {
            return SpanScores(values: [])
        }
        guard encoding.wordMask.count == encoding.sequenceLength else {
            throw GLiNERError.encodingError("Word mask length \(encoding.wordMask.count) does not align with input length \(encoding.sequenceLength)")
        }
        guard encoding.classTokenPositions.count >= labelCount else {
            throw GLiNERError.encodingError("Expected at least \(labelCount) class tokens but found \(encoding.classTokenPositions.count)")
        }

        let promptEmbeddings = try gatherPromptEmbeddings(from: hiddenStates, classPositions: encoding.classTokenPositions, labelCount: labelCount)
        let wordEmbeddings = try gatherWordEmbeddings(from: hiddenStates, encoding: encoding)
        guard !wordEmbeddings.isEmpty else {
            return SpanScores(values: [])
        }

        let spanInputs = buildSpanInputs(wordCount: wordEmbeddings.count)

        let scores = try await scoringModel.score(
            wordEmbeddings: wordEmbeddings,
            promptEmbeddings: promptEmbeddings,
            spanIndex: spanInputs.indices,
            spanMask: spanInputs.mask,
            labelCount: labelCount,
            maxWidth: metadata.maxWidth
        )

        return SpanScores(values: scores)
    }

    var maxWordCount: Int {
        scoringModel.limits.maxWordCount
    }

    private func gatherPromptEmbeddings(from hiddenStates: [[Float]], classPositions: [Int], labelCount: Int) throws -> [[Float]] {
        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(labelCount)
        for idx in 0..<labelCount {
            let tokenPosition = classPositions[idx]
            guard tokenPosition < hiddenStates.count else {
                throw GLiNERError.encodingError("Class token index \(tokenPosition) exceeds hidden state length \(hiddenStates.count)")
            }
            embeddings.append(hiddenStates[tokenPosition])
        }
        return embeddings
    }

    private func gatherWordEmbeddings(from hiddenStates: [[Float]], encoding: PromptEncoding) throws -> [[Float]] {
        let wordCount = encoding.textWordCount
        guard wordCount > 0 else { return [] }
        
        // Pre-allocate with correct capacity
        var wordEmbeddings: [[Float]] = []
        wordEmbeddings.reserveCapacity(wordCount)
        
        // Initialize with empty arrays
        for _ in 0..<wordCount {
            wordEmbeddings.append([])
        }
        
        // Populate embeddings in a single pass
        var seen = Array(repeating: false, count: wordCount)
        
        for (tokenIndex, marker) in encoding.wordMask.enumerated() where marker > 0 {
            let wordIndex = marker - 1
            guard wordIndex >= 0 && wordIndex < wordCount else { continue }
            
            if !seen[wordIndex] {
                wordEmbeddings[wordIndex] = hiddenStates[tokenIndex]
                seen[wordIndex] = true
            }
        }
        
        // Verify all words have embeddings
        guard seen.allSatisfy({ $0 }) else {
            let missingCount = seen.filter { !$0 }.count
            throw GLiNERError.encodingError("Word mask missing embeddings for \(missingCount) words (only \(wordCount - missingCount) / \(wordCount) available)")
        }
        
        return wordEmbeddings
    }

    private func buildSpanInputs(wordCount: Int) -> (indices: [[Float]], mask: [[Float]]) {
        guard wordCount > 0 else { return ([], []) }
        
        // Pre-allocate arrays with exact size
        let totalIndices = wordCount * metadata.maxWidth
        var indices: [[Float]] = []
        indices.reserveCapacity(totalIndices)
        
        // Pre-allocate mask array
        var mask = Array(repeating: Array(repeating: Float(0), count: metadata.maxWidth), count: wordCount)
        
        // Build indices and mask in a single pass with optimized loop
        for start in 0..<wordCount {
            let maxEnd = min(start + metadata.maxWidth, wordCount)
            
            for widthIndex in 0..<metadata.maxWidth {
                let end = start + widthIndex
                if end < maxEnd {
                    mask[start][widthIndex] = 1
                    indices.append([Float(start), Float(end)])
                } else {
                    // Use static zero array to reduce allocations
                    indices.append([0, 0])
                }
            }
        }
        
        return (indices, mask)
    }
}
