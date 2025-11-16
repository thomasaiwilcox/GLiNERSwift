import Foundation

/// Tokenizer that mirrors the Hugging Face DeBERTa SentencePiece tokenizer used by GLiNER.
public class GLiNERTokenizer {
    private static var cachedModels: [URL: SentencePieceUnigramModel] = [:]
    private static let modelLock = NSLock()
    
    private let model: SentencePieceUnigramModel
    private let metadata: SpanHeadMetadata
    private let maxLength: Int
    private let wordSplitter = WordSplitter()
    private let scratchStorageKey: NSString
    private let modelAccessLock = NSLock()
    private let gliner2Config: GLiNER2PromptConfiguration?
    private let gliner2SpecialTokenIds: [String: Int32]
    
    private enum PromptEntry {
        case special(String)
        case label(String)
        case text(WordToken)
    }

    private struct ScratchBuffers {
        var pieceIds: [Int32] = []
        var pieceTokens: [String] = []
        var tokens: [String] = []
        var inputIds: [Int32] = []
        var attentionMask: [Int32] = []
        var wordMask: [Int] = []
        var entries: [PromptEntry] = []
        var textWordRanges: [Range<String.Index>] = []
        var classPositions: [Int] = []

        mutating func resetBaseOutputBuffers() {
            tokens.removeAll(keepingCapacity: true)
            inputIds.removeAll(keepingCapacity: true)
            attentionMask.removeAll(keepingCapacity: true)
        }

        mutating func resetPromptBuffers() {
            resetBaseOutputBuffers()
            wordMask.removeAll(keepingCapacity: true)
            textWordRanges.removeAll(keepingCapacity: true)
            classPositions.removeAll(keepingCapacity: true)
        }

        mutating func resetPieceBuffers() {
            pieceIds.removeAll(keepingCapacity: true)
            pieceTokens.removeAll(keepingCapacity: true)
        }
    }
    
    private let entToken: String
    private let sepToken: String
    private let entTokenId: Int32
    private let tokenizerDirectory: URL?
    
    public init(maxLength: Int = 512, tokenizerDirectory: URL? = nil, gliner2Config: GLiNER2PromptConfiguration? = nil) throws {
        guard maxLength >= 2 else {
            throw GLiNERError.tokenizerError("maxLength must be at least 2 to accommodate special tokens")
        }
        self.maxLength = maxLength
        self.gliner2Config = gliner2Config
        self.metadata = try SpanHeadMetadataProvider.metadata()
        self.model = try GLiNERTokenizer.loadModel(metadata: metadata, tokenizerDirectory: tokenizerDirectory)
        self.entToken = metadata.entToken
        self.sepToken = metadata.sepToken
        self.entTokenId = Int32(metadata.specialTokens.entTokenIndex)
        self.tokenizerDirectory = tokenizerDirectory?.standardizedFileURL
        self.scratchStorageKey = "com.glinerswift.tokenizer.scratch.\(UUID().uuidString)" as NSString
        if let config = gliner2Config {
            var specialIds: [String: Int32] = [:]
            let tokens = [
                config.schemaSeparatorToken,
                config.textSeparatorToken,
                config.promptToken,
                config.classificationToken,
                config.entityToken,
                config.relationToken,
                config.listToken
            ]
            for token in tokens {
                if let id = model.tokenId(for: token) {
                    specialIds[token] = id
                }
            }
            self.gliner2SpecialTokenIds = specialIds
        } else {
            self.gliner2SpecialTokenIds = [:]
        }
    }
    
    /// Encode text into token IDs, attention mask, and token strings matching the Python tokenizer.
    /// - Parameters:
    ///   - text: Input text to tokenize.
    ///   - padToMaxLength: If `true`, the result is padded/truncated to `maxLength` with `[PAD]` tokens.
    public func encode(_ text: String, padToMaxLength: Bool = false) throws -> TokenizedInput {
        return withScratch { scratch in
            scratch.resetBaseOutputBuffers()
            scratch.resetPieceBuffers()
            tokenize(text, into: &scratch.pieceIds, tokens: &scratch.pieceTokens)

            let maxContentTokens = max(0, maxLength - 2)
            let contentCount = min(maxContentTokens, scratch.pieceIds.count)
            
            scratch.tokens.append("[CLS]")
            scratch.inputIds.append(model.clsId)
            scratch.attentionMask.append(Int32(1))
            
            if contentCount > 0 {
                scratch.tokens.reserveCapacity(contentCount + 2)
                scratch.inputIds.reserveCapacity(contentCount + 2)
                scratch.attentionMask.reserveCapacity(contentCount + 2)
            }
            
            for index in 0..<contentCount {
                scratch.tokens.append(scratch.pieceTokens[index])
                scratch.inputIds.append(scratch.pieceIds[index])
                scratch.attentionMask.append(Int32(1))
            }
            
            scratch.tokens.append("[SEP]")
            scratch.inputIds.append(model.sepId)
            scratch.attentionMask.append(Int32(1))
            
            if scratch.inputIds.count > maxLength {
                scratch.inputIds.removeLast(scratch.inputIds.count - maxLength)
                scratch.attentionMask.removeLast(scratch.attentionMask.count - maxLength)
                scratch.tokens.removeLast(scratch.tokens.count - maxLength)
            } else if padToMaxLength, scratch.inputIds.count < maxLength {
                let padCount = maxLength - scratch.inputIds.count
                scratch.inputIds.reserveCapacity(maxLength)
                scratch.attentionMask.reserveCapacity(maxLength)
                scratch.tokens.reserveCapacity(maxLength)
                for _ in 0..<padCount {
                    scratch.inputIds.append(model.padId)
                    scratch.attentionMask.append(Int32(0))
                    scratch.tokens.append("[PAD]")
                }
            }
            
            return TokenizedInput(
                inputIds: scratch.inputIds,
                attentionMask: scratch.attentionMask,
                tokens: scratch.tokens
            )
        }
    }

    /// Encode text along with label prompts (<<ENT>> label <<SEP>> text ...).
    public func encodePrompted(text: String, labels: [String], padToMaxLength: Bool = false) throws -> PromptEncoding {
        guard !labels.isEmpty else {
            return PromptEncoding(
                inputIds: [model.clsId, model.sepId],
                attentionMask: [1, 1],
                tokens: ["[CLS]", "[SEP]"],
                wordMask: [0, 0],
                textWordRanges: [],
                classTokenPositions: [],
                textWordCount: 0
            )
        }

        let wordTokens = wordSplitter.split(text)
        return try withScratch { scratch in
            scratch.resetPromptBuffers()
            scratch.tokens.append("[CLS]")
            scratch.inputIds.append(model.clsId)
            scratch.attentionMask.append(Int32(1))
            scratch.wordMask.append(0)
            var textWordIndex = 0
            
            scratch.entries.removeAll(keepingCapacity: true)
            for label in labels {
                scratch.entries.append(.special(entToken))
                scratch.entries.append(.label(label))
            }
            scratch.entries.append(.special(sepToken))
            for word in wordTokens {
                scratch.entries.append(.text(word))
            }
            
            func appendSpecial(_ token: String, id: Int32) {
                scratch.tokens.append(token)
                scratch.inputIds.append(id)
                scratch.attentionMask.append(Int32(1))
                scratch.wordMask.append(0)
                if id == entTokenId {
                    scratch.classPositions.append(scratch.inputIds.count - 1)
                }
            }
            
            func appendTextPieces(text: String, isTextWord: Bool, range: Range<String.Index>?) {
                scratch.resetPieceBuffers()
                tokenize(text, into: &scratch.pieceIds, tokens: &scratch.pieceTokens)
                guard !scratch.pieceIds.isEmpty else { return }
                if isTextWord, let range {
                    scratch.textWordRanges.append(range)
                }
                for (idx, id) in scratch.pieceIds.enumerated() {
                    scratch.tokens.append(scratch.pieceTokens[idx])
                    scratch.inputIds.append(id)
                    scratch.attentionMask.append(Int32(1))
                    if id == entTokenId {
                        scratch.classPositions.append(scratch.inputIds.count - 1)
                    }
                    if isTextWord {
                        scratch.wordMask.append(idx == 0 ? textWordIndex + 1 : 0)
                    } else {
                        scratch.wordMask.append(0)
                    }
                }
                if isTextWord {
                    textWordIndex += 1
                }
            }
            
            for entry in scratch.entries {
                switch entry {
                case .special(let token):
                    guard let id = model.tokenId(for: token) else {
                        throw GLiNERError.tokenizerError("Unknown special token: \(token)")
                    }
                    appendSpecial(token, id: id)
                case .label(let label):
                    appendTextPieces(text: label, isTextWord: false, range: nil)
                case .text(let wordToken):
                    appendTextPieces(text: wordToken.text, isTextWord: true, range: wordToken.range)
                }
            }

            scratch.tokens.append("[SEP]")
            scratch.inputIds.append(model.sepId)
            scratch.attentionMask.append(Int32(1))
            scratch.wordMask.append(0)
            
            guard scratch.inputIds.count <= maxLength else {
                throw GLiNERError.tokenizerError("Prompted input exceeds maximum length of \(maxLength) tokens")
            }

            if padToMaxLength, scratch.inputIds.count < maxLength {
                let padCount = maxLength - scratch.inputIds.count
                scratch.tokens.reserveCapacity(maxLength)
                scratch.inputIds.reserveCapacity(maxLength)
                scratch.attentionMask.reserveCapacity(maxLength)
                scratch.wordMask.reserveCapacity(maxLength)
                for _ in 0..<padCount {
                    scratch.tokens.append("[PAD]")
                    scratch.inputIds.append(model.padId)
                    scratch.attentionMask.append(Int32(0))
                    scratch.wordMask.append(0)
                }
            }

            return PromptEncoding(
                inputIds: scratch.inputIds,
                attentionMask: scratch.attentionMask,
                tokens: scratch.tokens,
                wordMask: scratch.wordMask,
                textWordRanges: scratch.textWordRanges,
                classTokenPositions: scratch.classPositions,
                textWordCount: scratch.textWordRanges.count
            )
        }
    }
    
    /// Build the GLiNER2 schema/text input sequence used by the Core ML runtime.
    /// Currently supports the entity-only path and mirrors `SchemaTransformer.format_input_with_mapping`.
    public func encodeGLiNER2SchemaInput(
        text: String,
        labels: [String],
        maxSpanWidth: Int? = nil
    ) throws -> GLiNER2SchemaEncoding {
        guard let gliner2Config else {
            throw GLiNERError.tokenizerError("GLiNER2 prompt configuration not provided")
        }
        guard !labels.isEmpty else {
            throw GLiNERError.tokenizerError("At least one label is required for GLiNER2 schema encoding")
        }
        let spanWidth = maxSpanWidth ?? metadata.maxWidth
        guard spanWidth > 0 else {
            throw GLiNERError.tokenizerError("GLiNER2 span width must be greater than zero")
        }

        // Build schema tokens for the entity task (single schema entry)
        var schemaTokens: [String] = ["(", gliner2Config.promptToken, "entities", "("]
        for label in labels {
            schemaTokens.append(gliner2Config.entityToken)
            schemaTokens.append(label)
        }
        schemaTokens.append(")")
        schemaTokens.append(")")
        let schemaTokensList = [schemaTokens]

        // Tokenize the text using the whitespace splitter to keep word boundaries + ranges
        let wordTokens = wordSplitter.split(text)
        var normalizedWords: [String] = []
        normalizedWords.reserveCapacity(wordTokens.count)
        var ranges: [Range<String.Index>] = []
        ranges.reserveCapacity(wordTokens.count)
        var startOffsets: [Int] = []
        var endOffsets: [Int] = []
        startOffsets.reserveCapacity(wordTokens.count)
        endOffsets.reserveCapacity(wordTokens.count)
        for token in wordTokens {
            normalizedWords.append(token.text.lowercased())
            ranges.append(token.range)
            startOffsets.append(text.distance(from: text.startIndex, to: token.range.lowerBound))
            endOffsets.append(text.distance(from: text.startIndex, to: token.range.upperBound))
        }

        return try withScratch { scratch in
            scratch.resetBaseOutputBuffers()
            scratch.resetPieceBuffers()

            // Build the combined schema/text token list
            var combinedTokens: [String] = []
            combinedTokens.reserveCapacity(schemaTokensList.reduce(0) { $0 + $1.count } + normalizedWords.count + schemaTokensList.count)
            for (index, tokens) in schemaTokensList.enumerated() {
                combinedTokens.append(contentsOf: tokens)
                if index < schemaTokensList.count - 1 {
                    combinedTokens.append(gliner2Config.schemaSeparatorToken)
                }
            }
            combinedTokens.append(gliner2Config.textSeparatorToken)
            combinedTokens.append(contentsOf: normalizedWords)

            var mappedIndices: [GLiNER2SchemaEncoding.Mapping] = []
            mappedIndices.reserveCapacity(combinedTokens.count)
            var promptLocations: [GLiNER2SchemaEncoding.PromptLocation] = []
            promptLocations.reserveCapacity(labels.count + 1)

            let textSchemaIndex = schemaTokensList.count
            var currentSchemaIndex = 0
            var foundSeparator = false
            var runningSubwordIndex = 0

            for (origIdx, token) in combinedTokens.enumerated() {
                let segment: GLiNER2SchemaEncoding.Segment
                var schemaIndex = currentSchemaIndex
                if token == gliner2Config.textSeparatorToken {
                    segment = .separator
                    schemaIndex = textSchemaIndex
                    foundSeparator = true
                } else if !foundSeparator {
                    segment = .schema
                    if token == gliner2Config.schemaSeparatorToken {
                        currentSchemaIndex += 1
                    }
                } else {
                    segment = .text
                    schemaIndex = textSchemaIndex
                }

                scratch.resetPieceBuffers()
                if let specialId = gliner2SpecialTokenIds[token] {
                    scratch.pieceIds.append(specialId)
                    scratch.pieceTokens.append(token)
                } else {
                    tokenize(token, into: &scratch.pieceIds, tokens: &scratch.pieceTokens)
                }
                let pieceCount = scratch.pieceIds.count

                for idx in 0..<pieceCount {
                    scratch.inputIds.append(scratch.pieceIds[idx])
                    scratch.attentionMask.append(Int32(1))
                    scratch.tokens.append(scratch.pieceTokens[idx])
                    mappedIndices.append(GLiNER2SchemaEncoding.Mapping(
                        segment: segment,
                        originalIndex: origIdx,
                        schemaIndex: schemaIndex
                    ))
                }

                if segment == .schema,
                   pieceCount > 0,
                   let kind = GLiNER2SchemaEncoding.PromptLocation.Kind(token: token, configuration: gliner2Config) {
                    let location = GLiNER2SchemaEncoding.PromptLocation(
                        kind: kind,
                        schemaIndex: schemaIndex,
                        originalIndex: origIdx,
                        subwordRange: runningSubwordIndex..<(runningSubwordIndex + pieceCount)
                    )
                    promptLocations.append(location)
                }

                runningSubwordIndex += pieceCount
            }

            guard scratch.inputIds.count <= maxLength else {
                throw GLiNERError.tokenizerError("GLiNER2 schema input exceeds maximum length of \(maxLength) tokens")
            }

            let spanPlan = GLiNER2TokenizerHelpers.buildSpanPlan(wordCount: normalizedWords.count, maxWidth: spanWidth)

            if normalizedWords.count < 40 {
                let sampleTokens = scratch.tokens.prefix(24).joined(separator: ", ")
                print("[DEBUG] Schema/text tokens: [\(sampleTokens)]")
                let sampleIds = scratch.inputIds.prefix(24).map(String.init).joined(separator: ", ")
                print("[DEBUG] Schema/text ids: [\(sampleIds)]")
            }

            return GLiNER2SchemaEncoding(
                schemaTokensList: schemaTokensList,
                textTokens: normalizedWords,
                inputIds: scratch.inputIds,
                attentionMask: scratch.attentionMask,
                mappedIndices: mappedIndices,
                subwordTokens: scratch.tokens,
                textWordRanges: ranges,
                startTokenCharacterMap: startOffsets,
                endTokenCharacterMap: endOffsets,
                maxSpanWidth: spanWidth,
                spanIndices: spanPlan.indices,
                spanMask: spanPlan.mask,
                promptLocations: promptLocations,
                entityLabels: labels,
                sequenceCapacity: maxLength
            )
        }
    }

    private func withScratch<T>(_ body: (inout ScratchBuffers) throws -> T) rethrows -> T {
        let threadDict = Thread.current.threadDictionary
        let box: ScratchBufferBox
        if let existing = threadDict[scratchStorageKey] as? ScratchBufferBox {
            box = existing
        } else {
            let newBox = ScratchBufferBox()
            threadDict[scratchStorageKey] = newBox
            box = newBox
        }
        return try body(&box.buffers)
    }

    private func tokenize(_ text: String, into ids: inout [Int32], tokens: inout [String]) {
        modelAccessLock.lock()
        defer { modelAccessLock.unlock() }
        model.tokenize(text, into: &ids, tokens: &tokens)
    }
    
    private static func loadModel(metadata: SpanHeadMetadata, tokenizerDirectory: URL?) throws -> SentencePieceUnigramModel {
        modelLock.lock()
        defer { modelLock.unlock() }
        let resourceURL: URL
        if let tokenizerDirectory {
            resourceURL = tokenizerDirectory
        } else if let baseURL = Bundle.module.resourceURL {
            resourceURL = baseURL
        } else {
            throw GLiNERError.tokenizerError("Tokenizer resources not found in bundle")
        }
        let key = resourceURL.standardizedFileURL
        if let cached = cachedModels[key] {
            return cached
        }
        let model = try SentencePieceUnigramModel(resourceDirectory: resourceURL)
        model.registerSpecialToken(metadata.specialTokens.flertToken, id: Int32(metadata.specialTokens.flertTokenIndex))
        model.registerSpecialToken(metadata.entToken, id: Int32(metadata.specialTokens.entTokenIndex))
        model.registerSpecialToken(metadata.sepToken, id: Int32(metadata.specialTokens.sepTokenIndex))
        cachedModels[key] = model
        return model
    }
    
    private final class ScratchBufferBox {
        var buffers = ScratchBuffers()
    }
}

private enum GLiNER2TokenizerHelpers {
    static func buildSpanPlan(wordCount: Int, maxWidth: Int) -> (indices: [[Int32]], mask: [[Float]]) {
        guard wordCount > 0, maxWidth > 0 else {
            return ([], [])
        }
        var indices: [[Int32]] = []
        indices.reserveCapacity(wordCount * maxWidth)
        var mask = Array(repeating: Array(repeating: Float(0), count: maxWidth), count: wordCount)
        for start in 0..<wordCount {
            for width in 0..<maxWidth {
                let end = start + width
                if end < wordCount {
                    mask[start][width] = 1
                    indices.append([Int32(start), Int32(end)])
                } else {
                    indices.append([0, 0])
                }
            }
        }
        return (indices, mask)
    }
}
