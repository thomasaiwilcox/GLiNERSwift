import Foundation

public struct PromptEncoding {
    public let inputIds: [Int32]
    public let attentionMask: [Int32]
    public let tokens: [String]
    public let wordMask: [Int]
    public let textWordRanges: [Range<String.Index>]
    public let classTokenPositions: [Int]
    public let textWordCount: Int

    var sequenceLength: Int {
        inputIds.count
    }
}
