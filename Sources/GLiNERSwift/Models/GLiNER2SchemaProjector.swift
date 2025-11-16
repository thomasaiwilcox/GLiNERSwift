import Foundation

struct GLiNER2SchemaProjection {
    struct PromptEmbedding {
        let kind: GLiNER2SchemaEncoding.PromptLocation.Kind
        let vector: [Float]
    }

    let wordEmbeddings: [[Float]]
    let schemaPromptEmbeddings: [[PromptEmbedding]]
}

enum GLiNER2SchemaProjector {
    static func project(
        hiddenStates: [[Float]],
        encoding: GLiNER2SchemaEncoding
    ) throws -> GLiNER2SchemaProjection {
        guard let hiddenSize = hiddenStates.first?.count, hiddenSize > 0 else {
            throw GLiNERError.encodingError("Hidden states must include at least one dimension")
        }
        guard hiddenStates.count == encoding.mappedIndices.count else {
            throw GLiNERError.encodingError("Hidden state count (\(hiddenStates.count)) does not match mapped indices (\(encoding.mappedIndices.count))")
        }

        let wordEmbeddings = try buildWordEmbeddings(
            hiddenStates: hiddenStates,
            encoding: encoding,
            hiddenSize: hiddenSize
        )
        let promptEmbeddings = try buildPromptEmbeddings(
            hiddenStates: hiddenStates,
            encoding: encoding,
            hiddenSize: hiddenSize
        )

        return GLiNER2SchemaProjection(
            wordEmbeddings: wordEmbeddings,
            schemaPromptEmbeddings: promptEmbeddings
        )
    }
}

private extension GLiNER2SchemaProjector {
    static func buildWordEmbeddings(
        hiddenStates: [[Float]],
        encoding: GLiNER2SchemaEncoding,
        hiddenSize: Int
    ) throws -> [[Float]] {
        let wordCount = encoding.textTokens.count
        guard wordCount > 0 else {
            return []
        }
        var embeddings = Array(repeating: [Float](), count: wordCount)
        var seen = Array(repeating: false, count: wordCount)
        let textStart = encoding.textStartOriginalIndex

        for (index, mapping) in encoding.mappedIndices.enumerated() where mapping.segment == .text {
            let wordIndex = mapping.originalIndex - textStart
            guard wordIndex >= 0 && wordIndex < wordCount else { continue }
            guard !seen[wordIndex] else { continue }
            embeddings[wordIndex] = hiddenStates[index]
            seen[wordIndex] = true
        }

        guard seen.allSatisfy({ $0 }) else {
            throw GLiNERError.encodingError("Missing subword embeddings for one or more words in GLiNER2 schema encoding")
        }
        return embeddings
    }

    static func buildPromptEmbeddings(
        hiddenStates: [[Float]],
        encoding: GLiNER2SchemaEncoding,
        hiddenSize: Int
    ) throws -> [[GLiNER2SchemaProjection.PromptEmbedding]] {
        let schemaCount = encoding.schemaTokensList.count
        guard schemaCount > 0 else {
            throw GLiNERError.encodingError("GLiNER2 schema encoding does not contain any schemas")
        }
        var perSchema = Array(repeating: [GLiNER2SchemaProjection.PromptEmbedding](), count: schemaCount)
        for location in encoding.promptLocations {
            guard location.schemaIndex < schemaCount else { continue }
            let vector = try averageHiddenStates(
                hiddenStates: hiddenStates,
                range: location.subwordRange,
                hiddenSize: hiddenSize
            )
            let embedding = GLiNER2SchemaProjection.PromptEmbedding(
                kind: location.kind,
                vector: vector
            )
            perSchema[location.schemaIndex].append(embedding)
        }
        return perSchema
    }

    static func averageHiddenStates(
        hiddenStates: [[Float]],
        range: Range<Int>,
        hiddenSize: Int
    ) throws -> [Float] {
        guard !range.isEmpty else {
            throw GLiNERError.encodingError("Prompt location produced an empty subword range")
        }
        guard range.lowerBound >= 0, range.upperBound <= hiddenStates.count else {
            throw GLiNERError.encodingError("Prompt location range \(range) exceeds hidden state length \(hiddenStates.count)")
        }
        var accumulator = [Float](repeating: 0, count: hiddenSize)
        for index in range {
            try accumulate(source: hiddenStates[index], into: &accumulator)
        }
        let scale = 1 / Float(range.count)
        scaleVector(&accumulator, scale: scale)
        return accumulator
    }

    static func accumulate(source: [Float], into destination: inout [Float]) throws {
        guard source.count == destination.count else {
            throw GLiNERError.encodingError("Hidden state width mismatch while accumulating prompt/word embeddings")
        }
        for idx in 0..<destination.count {
            destination[idx] += source[idx]
        }
    }

    static func scaleVector(_ vector: inout [Float], scale: Float) {
        guard scale != 1 else { return }
        for idx in 0..<vector.count {
            vector[idx] *= scale
        }
    }
}
