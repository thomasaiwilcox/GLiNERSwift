import Foundation

/// Represents a single token piece in the SentencePiece vocabulary.
private struct TokenPiece {
    let id: Int32
    let token: String
    let score: Float
}

/// Candidate token spanning a substring of the normalized input.
private struct CandidatePiece {
    let length: Int
    let piece: TokenPiece
}

/// Backpointer used during Viterbi decoding.
private struct BackPointer {
    let previousIndex: Int
    let candidate: CandidatePiece
}

/// Decodes text using a SentencePiece unigram model exported from the Hugging Face tokenizer.
final class SentencePieceUnigramModel {
    private struct TokenizerFile: Decodable {
        let model: Model
        let addedTokens: [AddedToken]?

        struct Model: Decodable {
            let type: String
            let unkId: Int32
            let vocab: [VocabEntry]

            enum CodingKeys: String, CodingKey {
                case type
                case unkId = "unk_id"
                case vocab
            }
        }

        struct AddedToken: Decodable {
            let id: Int32
            let content: String

            enum CodingKeys: String, CodingKey {
                case id
                case content
            }
        }

        enum CodingKeys: String, CodingKey {
            case model
            case addedTokens = "added_tokens"
        }
    }
    
    private struct VocabEntry: Decodable {
        let token: String
        let score: Float
        
        init(from decoder: Decoder) throws {
            var container = try decoder.unkeyedContainer()
            token = try container.decode(String.self)
            score = try container.decode(Float.self)
        }
    }
    
    /// Regex matching all whitespace variants handled by the tokenizer's precompiled map.
    private static let whitespaceRegex: NSRegularExpression = {
        let ideographicSpace = "\u{3000}"
        let byteOrderMark = "\u{FEFF}"
        let pattern = "[\\s" + ideographicSpace + byteOrderMark + "]+"
        return try! NSRegularExpression(pattern: pattern, options: [])
    }()
    
    private var tokenDictionary: [String: TokenPiece]
    private var maxTokenLength: Int
    private let unkPiece: TokenPiece
    let padId: Int32
    let clsId: Int32
    let sepId: Int32
    let maskId: Int32?
    let unkId: Int32
    
    private var characterBuffer: [Character] = []
    private var lattice: [[CandidatePiece]] = []
    private var scores: [Float] = []
    private var backPointers: [BackPointer?] = []
    
    init(resourceDirectory: URL) throws {
        let fm = FileManager.default
        let candidates = [
            resourceDirectory.appendingPathComponent("tokenizer/tokenizer.json"),
            resourceDirectory.appendingPathComponent("tokenizer.json")
        ]
        guard let tokenizerURL = candidates.first(where: { fm.fileExists(atPath: $0.path) }) else {
            throw GLiNERError.tokenizerError("Unable to locate tokenizer.json under \(resourceDirectory.path)")
        }
        let data = try Data(contentsOf: tokenizerURL)
        let tokenizerFile = try JSONDecoder().decode(TokenizerFile.self, from: data)
        guard tokenizerFile.model.type.lowercased() == "unigram" else {
            throw GLiNERError.tokenizerError("Unsupported tokenizer model type: \(tokenizerFile.model.type)")
        }
        
        var dictionary: [String: TokenPiece] = [:]
        dictionary.reserveCapacity(tokenizerFile.model.vocab.count)
        var maxLength = 0
        for (idx, entry) in tokenizerFile.model.vocab.enumerated() {
            let piece = TokenPiece(id: Int32(idx), token: entry.token, score: entry.score)
            dictionary[entry.token] = piece
            maxLength = max(maxLength, entry.token.count)
        }

        if let added = tokenizerFile.addedTokens {
            for entry in added {
                guard dictionary[entry.content] == nil else { continue }
                let piece = TokenPiece(id: entry.id, token: entry.content, score: 0)
                dictionary[entry.content] = piece
                maxLength = max(maxLength, entry.content.count)
            }
        }
        
        guard let unkPiece = dictionary["[UNK]"],
              let padPiece = dictionary["[PAD]"],
              let clsPiece = dictionary["[CLS]"],
              let sepPiece = dictionary["[SEP]"] else {
            throw GLiNERError.tokenizerError("Tokenizer vocabulary missing special tokens")
        }
        
        self.tokenDictionary = dictionary
        self.maxTokenLength = maxLength
        self.unkPiece = unkPiece
        self.padId = padPiece.id
        self.clsId = clsPiece.id
        self.sepId = sepPiece.id
        self.unkId = tokenizerFile.model.unkId
        self.maskId = dictionary["[MASK]"]?.id
    }
    
    /// Tokenize raw text into token IDs and string pieces using caller-provided buffers.
    func tokenize(_ text: String, into ids: inout [Int32], tokens: inout [String]) {
        ids.removeAll(keepingCapacity: true)
        tokens.removeAll(keepingCapacity: true)
        let normalized = normalize(text)
        guard !normalized.isEmpty else { return }
        
        populateCharacterBuffer(with: normalized)
        let length = characterBuffer.count
        guard length > 0 else { return }
        prepareLattice(for: length)
        buildLattice(for: length)
        prepareDynamicProgrammingStorage(for: length)
        runViterbi(length: length)
        backtrackResults(length: length, ids: &ids, tokens: &tokens)
    }

    /// Convenience API maintaining the previous tuple return signature for existing call sites.
    func tokenize(_ text: String) -> (ids: [Int32], tokens: [String]) {
        var ids: [Int32] = []
        var tokens: [String] = []
        tokenize(text, into: &ids, tokens: &tokens)
        return (ids, tokens)
    }

    /// Register an additional special token (e.g., <<ENT>>) with a concrete ID.
    func registerSpecialToken(_ token: String, id: Int32) {
        tokenDictionary[token] = TokenPiece(id: id, token: token, score: 0)
        if token.count > maxTokenLength {
            maxTokenLength = token.count
        }
    }

    func tokenId(for token: String) -> Int32? {
        tokenDictionary[token]?.id
    }
    
    private func normalize(_ text: String) -> String {
        var normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        normalized = normalized.applyingTransform(.init("NFKC"), reverse: false) ?? normalized
        let nsRange = NSRange(normalized.startIndex..<normalized.endIndex, in: normalized)
        normalized = SentencePieceUnigramModel.whitespaceRegex.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: nsRange,
            withTemplate: " "
        )
        
        if normalized.isEmpty {
            return ""
        }
        
        if !normalized.hasPrefix(" ") {
            normalized = " " + normalized
        }
        
        return normalized.replacingOccurrences(of: " ", with: "▁")
    }
}

// MARK: - Private helpers

extension SentencePieceUnigramModel {
    private func populateCharacterBuffer(with text: String) {
        characterBuffer.removeAll(keepingCapacity: true)
        characterBuffer.reserveCapacity(text.count)
        for scalar in text {
            characterBuffer.append(scalar)
        }
    }
    
    private func prepareLattice(for length: Int) {
        if lattice.count < length {
            lattice = Array(repeating: [], count: length)
        }
        for index in 0..<length {
            lattice[index].removeAll(keepingCapacity: true)
        }
    }
    
    private func buildLattice(for length: Int) {
        for index in 0..<length {
            let available = min(maxTokenLength, length - index)
            guard available > 0 else { continue }
            for size in 1...available {
                let piece = String(characterBuffer[index..<(index + size)])
                if let tokenPiece = tokenDictionary[piece] {
                    lattice[index].append(CandidatePiece(length: size, piece: tokenPiece))
                }
            }
            if lattice[index].isEmpty {
                lattice[index].append(CandidatePiece(length: 1, piece: unkPiece))
            }
        }
    }
    
    private func prepareDynamicProgrammingStorage(for length: Int) {
        let required = length + 1
        if scores.count < required {
            scores = Array(repeating: -Float.greatestFiniteMagnitude, count: required)
        }
        if backPointers.count < required {
            backPointers = Array(repeating: nil, count: required)
        }
        for idx in 0..<required {
            scores[idx] = -Float.greatestFiniteMagnitude
            backPointers[idx] = nil
        }
        scores[0] = 0
    }
    
    private func runViterbi(length: Int) {
        for position in 0..<length {
            let currentScore = scores[position]
            if currentScore == -Float.greatestFiniteMagnitude { continue }
            for candidate in lattice[position] {
                let nextIndex = position + candidate.length
                let nextScore = currentScore + candidate.piece.score
                if nextScore > scores[nextIndex] {
                    scores[nextIndex] = nextScore
                    backPointers[nextIndex] = BackPointer(previousIndex: position, candidate: candidate)
                }
            }
        }
    }
    
    private func backtrackResults(length: Int, ids: inout [Int32], tokens: inout [String]) {
        var pos = length
        while pos > 0 {
            if let back = backPointers[pos] {
                ids.append(back.candidate.piece.id)
                tokens.append(back.candidate.piece.token)
                pos = back.previousIndex
            } else {
                ids.append(unkPiece.id)
                tokens.append(unkPiece.token)
                pos -= 1
            }
        }
        ids.reverse()
        tokens.reverse()
    }
}
