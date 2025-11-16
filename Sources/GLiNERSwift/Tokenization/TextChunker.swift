import Foundation

public struct TextChunk {
    public let text: String
    public let startOffset: Int
    public let endOffset: Int
    public let wordCount: Int
}

public final class TextChunker {
    public static let defaultMaxCharacters = 4096
    public static let defaultOverlapCharacters = 512
    public static let defaultMaxWordLimit = 240

    private let maxCharacters: Int
    private let overlapCharacters: Int
    private let maxWords: Int
    private let wordSplitter: WordSplitter

    public init(
        maxCharacters: Int = TextChunker.defaultMaxCharacters,
        overlapCharacters: Int = TextChunker.defaultOverlapCharacters,
        maxWords: Int = TextChunker.defaultMaxWordLimit
    ) {
        self.maxCharacters = max(256, maxCharacters)
        self.overlapCharacters = max(0, overlapCharacters)
        self.maxWords = max(1, maxWords)
        self.wordSplitter = WordSplitter()
    }

    public func shouldChunk(text: String) -> Bool {
        wordSplitter.countWords(in: text) > maxWords
    }

    public func chunk(text: String) -> [TextChunk] {
        guard !text.isEmpty else { return [] }
        var chunks: [TextChunk] = []
        var startIndex = text.startIndex
        let textEnd = text.endIndex

        while startIndex < textEnd {
            let limit = text.index(startIndex, offsetBy: maxCharacters, limitedBy: textEnd) ?? textEnd
            var candidateEnd = limit
            if limit != textEnd {
                let window = text[startIndex..<limit]
                if let newline = window.lastIndex(of: "\n") {
                    candidateEnd = newline
                } else if let whitespace = window.lastIndex(where: { $0.isWhitespace }) {
                    candidateEnd = text.index(after: whitespace)
                }
            }
            if candidateEnd <= startIndex {
                candidateEnd = limit
            }

            var chunkRange = startIndex..<candidateEnd
            chunkRange = trimWhitespace(in: text, range: chunkRange)
            if chunkRange.isEmpty {
                startIndex = candidateEnd
                continue
            }

            var chunkSubstring = text[chunkRange]
            var chunkWordCount = wordSplitter.countWords(in: chunkSubstring)
            while chunkWordCount > maxWords,
                text.distance(from: chunkRange.lowerBound, to: chunkRange.upperBound) > 1 {
                if let shrinkIndex = chunkSubstring.lastIndex(where: { $0.isWhitespace || $0 == "\n" }) {
                    if shrinkIndex <= chunkRange.lowerBound { break }
                    chunkRange = chunkRange.lowerBound..<shrinkIndex
                } else {
                    let newEnd = text.index(before: chunkRange.upperBound)
                    guard newEnd > chunkRange.lowerBound else { break }
                    chunkRange = chunkRange.lowerBound..<newEnd
                }
                chunkRange = trimWhitespace(in: text, range: chunkRange)
                if chunkRange.isEmpty { break }
                chunkSubstring = text[chunkRange]
                chunkWordCount = wordSplitter.countWords(in: chunkSubstring)
            }

            guard !chunkRange.isEmpty else {
                startIndex = candidateEnd
                continue
            }

            let chunkText = String(chunkSubstring)
            let startOffset = text.distance(from: text.startIndex, to: chunkRange.lowerBound)
            let endOffset = text.distance(from: text.startIndex, to: chunkRange.upperBound)
            chunks.append(TextChunk(text: chunkText, startOffset: startOffset, endOffset: endOffset, wordCount: chunkWordCount))

            if chunkRange.upperBound >= textEnd {
                break
            }

            let chunkLength = text.distance(from: chunkRange.lowerBound, to: chunkRange.upperBound)
            let overlapDistance = min(overlapCharacters, max(0, chunkLength - 1))
            if overlapDistance <= 0 {
                startIndex = chunkRange.upperBound
            } else {
                let newStart = text.index(chunkRange.upperBound, offsetBy: -overlapDistance, limitedBy: text.startIndex) ?? text.startIndex
                startIndex = newStart < chunkRange.lowerBound ? chunkRange.lowerBound : newStart
            }
        }

        return chunks
    }

    private func trimWhitespace(in text: String, range: Range<String.Index>) -> Range<String.Index> {
        var lower = range.lowerBound
        var upper = range.upperBound
        while lower < upper, text[lower].isWhitespace || text[lower] == "\n" {
            lower = text.index(after: lower)
        }
        while upper > lower {
            let before = text.index(before: upper)
            if text[before].isWhitespace || text[before] == "\n" {
                upper = before
            } else {
                break
            }
        }
        if lower >= upper {
            return lower..<lower
        }
        return lower..<upper
    }
}
