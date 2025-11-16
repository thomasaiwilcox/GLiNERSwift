import Foundation

struct WordToken {
    let text: String
    let range: Range<String.Index>
}

/// Regex-based splitter that mirrors GLiNER's default whitespace splitter.
final class WordSplitter {
    private let regex: NSRegularExpression

    init() {
        // Matches word characters optionally joined by '-' or '_' as in Python implementation.
        let pattern = "\\w+(?:[-_]\\w+)*|\\S"
        regex = try! NSRegularExpression(pattern: pattern, options: [])
    }

    func split(_ text: String) -> [WordToken] {
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: nsRange)
        return matches.compactMap { match in
            guard let range = Range(match.range, in: text) else { return nil }
            let token = String(text[range])
            return WordToken(text: token, range: range)
        }
    }

    func countWords(in text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.numberOfMatches(in: text, options: [], range: nsRange)
    }

    func countWords(in text: Substring) -> Int {
        countWords(in: String(text))
    }
}
