import Foundation

/// Describes the special tokens used by the GLiNER2 prompt/schema formatter.
public struct GLiNER2PromptConfiguration {
    public let schemaSeparatorToken: String
    public let textSeparatorToken: String
    public let promptToken: String
    public let classificationToken: String
    public let entityToken: String
    public let relationToken: String
    public let listToken: String

    public init(
        schemaSeparatorToken: String = "[SEP_STRUCT]",
        textSeparatorToken: String = "[SEP_TEXT]",
        promptToken: String = "[P]",
        classificationToken: String = "[C]",
        entityToken: String = "[E]",
        relationToken: String = "[R]",
        listToken: String = "[L]"
    ) {
        self.schemaSeparatorToken = schemaSeparatorToken
        self.textSeparatorToken = textSeparatorToken
        self.promptToken = promptToken
        self.classificationToken = classificationToken
        self.entityToken = entityToken
        self.relationToken = relationToken
        self.listToken = listToken
    }
}
