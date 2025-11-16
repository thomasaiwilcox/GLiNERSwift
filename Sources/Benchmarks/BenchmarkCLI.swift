import Foundation
@preconcurrency import CoreML
import GLiNERSwift

@main
struct BenchmarkCLI {
    static func main() async {
        do {
            let options = try BenchmarkOptions.parse()
            let runner = BenchmarkRunner(options: options)
            try await runner.run()
        } catch {
            fputs("Benchmark failed: \(error)\n", stderr)
            exit(1)
        }
    }
}

enum BenchmarkMode: String, Codable {
    case latency
    case throughput
}

enum ComputeUnitsOption: String, Codable, CaseIterable {
    case cpu
    case cpuAndGPU = "cpu-gpu"
    case all

    init?(argument: String) {
        let normalized = argument.lowercased()
        if let match = ComputeUnitsOption.allCases.first(where: { $0.rawValue == normalized }) {
            self = match
        } else if normalized == "cpuandgpu" {
            self = .cpuAndGPU
        } else {
            return nil
        }
    }

    var coreMLValue: MLComputeUnits {
        switch self {
        case .cpu:
            return .cpuOnly
        case .cpuAndGPU:
            return .cpuAndGPU
        case .all:
            return .all
        }
    }
}

struct BenchmarkOptions {
    var manifestURL: URL
    var fixturesURL: URL?
    var outputURL: URL?
    var iterations: Int
    var warmup: Int
    var mode: BenchmarkMode
    var batchSize: Int
    var concurrency: Int
    var encoderComputeUnits: ComputeUnitsOption
    var encoderWarmup: Int
    var printEntities: Bool
    var singleText: String?
    var singleLabels: [String]
    var singleTextThreshold: Float?
    var textFileURL: URL?
    var textFileChunkChars: Int
    var textFileChunkOverlap: Int

    static func parse() throws -> BenchmarkOptions {
        var manifestOverride: URL?
        var options = BenchmarkOptions(
            manifestURL: URL(fileURLWithPath: "CoreMLArtifacts/export_manifest.json"),
            fixturesURL: nil,
            outputURL: nil,
            iterations: 5,
            warmup: 1,
            mode: .latency,
            batchSize: 8,
            concurrency: 1,
            encoderComputeUnits: .cpuAndGPU,
            encoderWarmup: 0,
            printEntities: false,
            singleText: nil,
            singleLabels: [],
            singleTextThreshold: nil,
            textFileURL: nil,
            textFileChunkChars: 1600,
            textFileChunkOverlap: 200
        )
        var iterator = CommandLine.arguments.dropFirst().makeIterator()

        while let argument = iterator.next() {
            switch argument {
            case "--fixtures":
                guard let path = iterator.next() else {
                    throw BenchmarkError.invalidArgument("Missing value for --fixtures")
                }
                options.fixturesURL = URL(fileURLWithPath: path)
            case "--manifest":
                guard let path = iterator.next() else {
                    throw BenchmarkError.invalidArgument("Missing value for --manifest")
                }
                manifestOverride = URL(fileURLWithPath: path)
            case "--output":
                guard let path = iterator.next() else {
                    throw BenchmarkError.invalidArgument("Missing value for --output")
                }
                options.outputURL = URL(fileURLWithPath: path)
            case "--iterations":
                guard let value = iterator.next(), let iterations = Int(value), iterations > 0 else {
                    throw BenchmarkError.invalidArgument("--iterations must be a positive integer")
                }
                options.iterations = iterations
            case "--warmup":
                guard let value = iterator.next(), let warmup = Int(value), warmup >= 0 else {
                    throw BenchmarkError.invalidArgument("--warmup must be zero or a positive integer")
                }
                options.warmup = warmup
            case "--mode":
                guard let value = iterator.next(), let parsed = BenchmarkMode(rawValue: value.lowercased()) else {
                    throw BenchmarkError.invalidArgument("--mode must be either latency or throughput")
                }
                options.mode = parsed
            case "--batch-size":
                guard let value = iterator.next(), let size = Int(value), size > 0 else {
                    throw BenchmarkError.invalidArgument("--batch-size must be a positive integer")
                }
                options.batchSize = size
            case "--concurrency":
                guard let value = iterator.next(), let limit = Int(value), limit > 0 else {
                    throw BenchmarkError.invalidArgument("--concurrency must be a positive integer")
                }
                options.concurrency = limit
            case "--encoder-compute-units":
                guard let value = iterator.next(), let units = ComputeUnitsOption(argument: value) else {
                    throw BenchmarkError.invalidArgument("--encoder-compute-units must be one of cpu, cpu-gpu, or all")
                }
                options.encoderComputeUnits = units
            case "--encoder-warmup":
                guard let value = iterator.next(), let passes = Int(value), passes >= 0 else {
                    throw BenchmarkError.invalidArgument("--encoder-warmup must be zero or a positive integer")
                }
                options.encoderWarmup = passes
            case "--print-entities":
                options.printEntities = true
            case "--text":
                guard let value = iterator.next(), !value.isEmpty else {
                    throw BenchmarkError.invalidArgument("--text requires a non-empty string")
                }
                options.singleText = value
            case "--labels":
                guard let value = iterator.next(), !value.isEmpty else {
                    throw BenchmarkError.invalidArgument("--labels requires a comma-separated label list")
                }
                let parsed = value
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                guard !parsed.isEmpty else {
                    throw BenchmarkError.invalidArgument("--labels must include at least one label")
                }
                options.singleLabels.append(contentsOf: parsed)
            case "--text-threshold":
                guard let value = iterator.next(), let threshold = Float(value) else {
                    throw BenchmarkError.invalidArgument("--text-threshold must be a floating point value")
                }
                options.singleTextThreshold = threshold
            case "--text-file":
                guard let path = iterator.next() else {
                    throw BenchmarkError.invalidArgument("--text-file requires a path to a .txt file")
                }
                options.textFileURL = URL(fileURLWithPath: path)
            case "--text-chunk-chars":
                guard let value = iterator.next(), let size = Int(value), size >= 256 else {
                    throw BenchmarkError.invalidArgument("--text-chunk-chars must be an integer >= 256")
                }
                options.textFileChunkChars = size
            case "--text-chunk-overlap":
                guard let value = iterator.next(), let overlap = Int(value), overlap >= 0 else {
                    throw BenchmarkError.invalidArgument("--text-chunk-overlap must be zero or a positive integer")
                }
                options.textFileChunkOverlap = overlap
            case "--help", "-h":
                BenchmarkCLI.printUsage()
                exit(EXIT_SUCCESS)
            default:
                throw BenchmarkError.invalidArgument("Unknown argument: \(argument)")
            }
        }

        options.manifestURL = try resolveManifestURL(explicit: manifestOverride)
        return options
    }

    private static func resolveManifestURL(explicit: URL?) throws -> URL {
        let fm = FileManager.default
        if let explicit {
            let standardized = explicit.standardizedFileURL
            guard fm.fileExists(atPath: standardized.path) else {
                throw BenchmarkError.manifestNotFound("Manifest not found at \(standardized.path)")
            }
            return standardized
        }

        let cwdDefault = URL(fileURLWithPath: "CoreMLArtifacts/export_manifest.json")
        if fm.fileExists(atPath: cwdDefault.path) {
            return cwdDefault
        }

        if let bundled = Bundle.module.url(
            forResource: "export_manifest",
            withExtension: "json"
        ), fm.fileExists(atPath: bundled.path) {
            return bundled
        }

        throw BenchmarkError.manifestNotFound("Unable to locate CoreMLArtifacts/export_manifest.json; pass --manifest <path>.")
    }
}

extension BenchmarkCLI {
    static func printUsage() {
                let usage = """
        Usage: gliner-benchmarks [options]

        Options:
                    --manifest <path>  Path to CoreML export manifest (default: ./CoreMLArtifacts/export_manifest.json)
          --fixtures <path>   Path to JSONL fixtures (defaults to bundled samples)
          --text <string>     Run against a single inline text sample (disables --fixtures)
          --labels <a,b,c>    Comma-separated labels used with --text (can be repeated)
          --text-threshold <v>Override entity threshold for --text input
          --text-file <path>  Load a .txt file and split it into benchmark chunks
          --text-chunk-chars <n>
                               Max characters per chunk when using --text-file (default: 1600)
          --text-chunk-overlap <n>
                               Overlap in characters between chunks (default: 200)
          --iterations <n>    Iterations per fixture to measure (default: 5)
          --warmup <n>        Warmup passes per fixture (default: 1)
                    --mode <name>       Either latency (default) or throughput
                    --batch-size <n>    Samples per batch when mode=throughput (default: 8)
                                        --concurrency <n>   Max in-flight requests per batch (default: 1)
                    --encoder-compute-units <value>
                                                         One of cpu, cpu-gpu, or all (default: cpu-gpu)
                    --encoder-warmup <n>  Number of dummy encoder passes before measurements (default: 0)
          --output <path>     Write JSON summary to the given path
                    --print-entities    Print entity types and their extracted values
          -h, --help          Show this message
        """
        print(usage)
    }
}

enum BenchmarkError: Error, CustomStringConvertible {
    case invalidArgument(String)
    case fixturesNotFound
    case emptyFixtures(URL)
    case manifestNotFound(String)

    var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        case .fixturesNotFound:
            return "Unable to locate benchmark fixtures"
        case .emptyFixtures(let url):
            return "No fixtures found at \(url.path)"
        case .manifestNotFound(let message):
            return message
        }
    }
}

struct BenchmarkSample: Decodable {
    let id: String
    let text: String
    let labels: [String]
    let threshold: Float?
}

struct BenchmarkStatistics: Codable {
    let mode: BenchmarkMode
    let samples: Int
    let iterationsPerSample: Int
    let totalRequests: Int
    let averageLatencyMs: Double
    let medianLatencyMs: Double
    let p95LatencyMs: Double
    let minLatencyMs: Double
    let maxLatencyMs: Double
    let charactersPerSecond: Double
    let wordsAnalyzed: Int
    let entitiesExtracted: Int
    let batchSize: Int?
    let batches: Int?
    let averageBatchLatencyMs: Double?
    let medianBatchLatencyMs: Double?
    let p95BatchLatencyMs: Double?
    let requestsPerSecond: Double?
}

struct BenchmarkRunner {
    let options: BenchmarkOptions

    func run() async throws {
        let fixtures = try loadFixtures()
        print("Loaded \(fixtures.count) benchmark samples")

        let model = try await GLiNERModel(
            manifestURL: options.manifestURL,
            encoderComputeUnits: options.encoderComputeUnits.coreMLValue
        )
        let entityReporter = options.printEntities ? EntityReporter() : nil
        try await performEncoderWarmup(fixtures: fixtures, model: model)
        let stats: BenchmarkStatistics

        switch options.mode {
        case .latency:
            stats = try await runLatencyMode(fixtures: fixtures, model: model, entityReporter: entityReporter)
        case .throughput:
            stats = try await runThroughputMode(fixtures: fixtures, model: model, entityReporter: entityReporter)
        }
        prettyPrint(stats: stats)
        entityReporter?.printSummary()

        if let outputURL = options.outputURL {
            try write(stats: stats, to: outputURL)
            print("Summary written to \(outputURL.path)")
        }
    }

    private func loadFixtures() throws -> [BenchmarkSample] {
        if let fileURL = options.textFileURL {
            return try loadTextFileFixtures(from: fileURL)
        }

        if let inlineText = options.singleText {
            guard !options.singleLabels.isEmpty else {
                throw BenchmarkError.invalidArgument("--labels is required when using --text")
            }
            let sample = BenchmarkSample(
                id: "cli_text",
                text: inlineText,
                labels: Array(Set(options.singleLabels)),
                threshold: options.singleTextThreshold
            )
            return [sample]
        }

        if let url = options.fixturesURL {
            return try decodeFixtures(from: url)
        }

        guard let bundledURL = Bundle.module.url(
            forResource: "benchmark_samples",
            withExtension: "jsonl",
            subdirectory: "Fixtures"
        ) else {
            throw BenchmarkError.fixturesNotFound
        }

        return try decodeFixtures(from: bundledURL)
    }

    private func loadTextFileFixtures(from url: URL) throws -> [BenchmarkSample] {
        guard !options.singleLabels.isEmpty else {
            throw BenchmarkError.invalidArgument("--labels is required when using --text-file")
        }
        let contents = try String(contentsOf: url, encoding: .utf8)
        let normalized = contents.replacingOccurrences(of: "\r\n", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            throw BenchmarkError.invalidArgument("--text-file \(url.path) is empty")
        }

        let chunker = TextChunker(
            maxCharacters: options.textFileChunkChars,
            overlapCharacters: options.textFileChunkOverlap,
            maxWords: TextChunker.defaultMaxWordLimit
        )
        let chunks = chunker.chunk(text: normalized)
        guard !chunks.isEmpty else {
            throw BenchmarkError.invalidArgument("Unable to split \(url.path) into benchmark chunks")
        }
        let dedupedLabels = Array(Set(options.singleLabels))
        return chunks.enumerated().map { index, chunk in
            BenchmarkSample(
                id: String(format: "text_chunk_%03d", index + 1),
                text: chunk.text,
                labels: dedupedLabels,
                threshold: options.singleTextThreshold
            )
        }
    }

    private func decodeFixtures(from url: URL) throws -> [BenchmarkSample] {
        let contents = try String(contentsOf: url, encoding: .utf8)
        let decoder = JSONDecoder()
        var samples: [BenchmarkSample] = []

        for rawLine in contents.split(whereSeparator: { $0.isNewline }) {
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            let data = Data(trimmed.utf8)
            let sample = try decoder.decode(BenchmarkSample.self, from: data)
            samples.append(sample)
        }

        guard !samples.isEmpty else {
            throw BenchmarkError.emptyFixtures(url)
        }

        return samples
    }


    private func performWarmup(sample: BenchmarkSample, model: GLiNERModel) async throws {
        guard options.warmup > 0 else { return }
        for _ in 0..<options.warmup {
            _ = try await model.extractEntities(
                from: sample.text,
                labels: sample.labels,
                threshold: sample.threshold
            )
        }
    }

    private func performWarmup(batch: [BenchmarkSample], model: GLiNERModel) async throws {
        guard options.warmup > 0 else { return }
        for sample in batch {
            try await performWarmup(sample: sample, model: model)
        }
    }

    private func performEncoderWarmup(fixtures: [BenchmarkSample], model: GLiNERModel) async throws {
        guard options.encoderWarmup > 0 else { return }
        guard !fixtures.isEmpty else { return }
        print("Running encoder warmup (\(options.encoderWarmup) passes)...")
        for pass in 0..<options.encoderWarmup {
            let sample = fixtures[pass % fixtures.count]
            guard !sample.labels.isEmpty else { continue }
            _ = try await model.extractEntities(
                from: sample.text,
                labels: sample.labels,
                threshold: sample.threshold
            )
        }
    }

    private func runLatencyMode(
        fixtures: [BenchmarkSample],
        model: GLiNERModel,
        entityReporter: EntityReporter?
    ) async throws -> BenchmarkStatistics {
        var latencies: [Double] = []
        latencies.reserveCapacity(fixtures.count * max(options.iterations, 1))
        var totalCharacters: Int = 0
        var totalWords: Int = 0
        var totalEntities: Int = 0
        let clock = ContinuousClock()

        for sample in fixtures {
            try await performWarmup(sample: sample, model: model)
            for iteration in 0..<options.iterations {
                let start = clock.now
                let entities = try await model.extractEntities(
                    from: sample.text,
                    labels: sample.labels,
                    threshold: sample.threshold
                )
                entityReporter?.record(entities)
                let elapsed = start.duration(to: clock.now)
                let latencyMs = elapsed.milliseconds
                latencies.append(latencyMs)
                totalCharacters += sample.text.count
                totalWords += BenchmarkRunner.wordCount(in: sample.text)
                totalEntities += entities.count

                let idColumn: String
                if sample.id.count >= 20 {
                    idColumn = String(sample.id.prefix(20))
                } else {
                    let padding = String(repeating: " ", count: 20 - sample.id.count)
                    idColumn = sample.id + padding
                }
                print("\(idColumn) | iter \(iteration + 1) | \(String(format: "%.2f", latencyMs)) ms")
            }
        }

        return buildStats(
            mode: .latency,
            latencies: latencies,
            characters: totalCharacters,
            words: totalWords,
            entities: totalEntities,
            fixtures: fixtures.count,
            batchSize: nil,
            batchLatencies: nil
        )
    }

    private func runThroughputMode(
        fixtures: [BenchmarkSample],
        model: GLiNERModel,
        entityReporter: EntityReporter?
    ) async throws -> BenchmarkStatistics {
        let batchSize = max(options.batchSize, 1)
        let concurrency = max(options.concurrency, 1)
        let batches = fixtures.chunked(into: batchSize)
        var latencies: [Double] = []
        latencies.reserveCapacity(fixtures.count * max(options.iterations, 1))
        var batchLatencies: [Double] = []
        batchLatencies.reserveCapacity(batches.count * max(options.iterations, 1))
        var totalCharacters: Int = 0
        var totalWords: Int = 0
        var totalEntities: Int = 0
        let clock = ContinuousClock()
        let collectEntities = entityReporter != nil

        for (batchIndex, batch) in batches.enumerated() {
            try await performWarmup(batch: batch, model: model)
            for iteration in 0..<options.iterations {
                let batchStart = clock.now
                let (
                    iterationLatencies,
                    charactersProcessed,
                    wordsProcessed,
                    entitiesFound,
                    detailedEntities
                ) = try await executeBatch(
                    batch,
                    model: model,
                    concurrency: concurrency,
                    clock: clock,
                    collectEntities: collectEntities
                )
                latencies.append(contentsOf: iterationLatencies)
                totalCharacters += charactersProcessed
                totalWords += wordsProcessed
                totalEntities += entitiesFound
                if let detailedEntities {
                    for entitySet in detailedEntities {
                        entityReporter?.record(entitySet)
                    }
                }
                let batchElapsed = batchStart.duration(to: clock.now).milliseconds
                batchLatencies.append(batchElapsed)
                print(
                    "batch \(batchIndex + 1)/\(batches.count) | iter \(iteration + 1) | \(String(format: "%.2f", batchElapsed)) ms (\(batch.count) samples)"
                )
            }
        }

        return buildStats(
            mode: .throughput,
            latencies: latencies,
            characters: totalCharacters,
            words: totalWords,
            entities: totalEntities,
            fixtures: fixtures.count,
            batchSize: batchSize,
            batchLatencies: batchLatencies
        )
    }

    private func executeBatch(
        _ batch: [BenchmarkSample],
        model: GLiNERModel,
        concurrency: Int,
        clock: ContinuousClock,
        collectEntities: Bool
    ) async throws -> (
        latencies: [Double],
        characters: Int,
        words: Int,
        entities: Int,
        detailedEntities: [[Entity]]?
    ) {
        func runSample(_ sample: BenchmarkSample) async throws -> (Double, Int, Int, [Entity]?) {
            let requestStart = clock.now
            let entities = try await model.extractEntities(
                from: sample.text,
                labels: sample.labels,
                threshold: sample.threshold
            )
            let latency = requestStart.duration(to: clock.now).milliseconds
            let capturedEntities = collectEntities ? entities : nil
            return (latency, sample.text.count, entities.count, capturedEntities)
        }

        guard concurrency > 1 else {
            var latencies: [Double] = []
            latencies.reserveCapacity(batch.count)
            var characters = 0
            var words = 0
            var entities = 0
            var detailedEntities: [[Entity]] = []
            for sample in batch {
                let (latency, processedCharacters, extracted, extractedEntities) = try await runSample(sample)
                latencies.append(latency)
                characters += processedCharacters
                words += BenchmarkRunner.wordCount(in: sample.text)
                entities += extracted
                if collectEntities, let extractedEntities {
                    detailedEntities.append(extractedEntities)
                }
            }
            return (
                latencies,
                characters,
                words,
                entities,
                collectEntities ? detailedEntities : nil
            )
        }

        var latencies: [Double] = []
        latencies.reserveCapacity(batch.count)
        var charactersProcessed = 0
        var wordsProcessed = 0
        var entitiesExtracted = 0
        var detailedResults: [[Entity]] = []
        var iterator = batch.makeIterator()

        try await withThrowingTaskGroup(of: (Double, Int, Int, [Entity]?).self) { group in
            var permits = min(concurrency, batch.count)
            while permits > 0, let sample = iterator.next() {
                permits -= 1
                group.addTask {
                    try await runSample(sample)
                }
                wordsProcessed += BenchmarkRunner.wordCount(in: sample.text)
            }

            while let (latency, processedCharacters, extracted, extractedEntities) = try await group.next() {
                latencies.append(latency)
                charactersProcessed += processedCharacters
                entitiesExtracted += extracted
                if collectEntities, let extractedEntities {
                    detailedResults.append(extractedEntities)
                }
                if let sample = iterator.next() {
                    group.addTask {
                        try await runSample(sample)
                    }
                    wordsProcessed += BenchmarkRunner.wordCount(in: sample.text)
                }
            }
        }

        return (
            latencies,
            charactersProcessed,
            wordsProcessed,
            entitiesExtracted,
            collectEntities ? detailedResults : nil
        )
    }

    private func buildStats(
        mode: BenchmarkMode,
        latencies: [Double],
        characters: Int,
        words: Int,
        entities: Int,
        fixtures: Int,
        batchSize: Int?,
        batchLatencies: [Double]?
    ) -> BenchmarkStatistics {
        let sorted = latencies.sorted()
        let total = sorted.reduce(0, +)
        let totalRequests = latencies.count
        let average = total / Double(max(totalRequests, 1))
        let median = percentile(sorted, percentile: 0.5)
        let p95 = percentile(sorted, percentile: 0.95)
        let minLatency = sorted.first ?? 0
        let maxLatency = sorted.last ?? 0
        let charactersPerSecond: Double
        if total > 0 {
            charactersPerSecond = Double(characters) / (total / 1000.0)
        } else {
            charactersPerSecond = 0
        }

        let batchStats: (avg: Double, median: Double, p95: Double)? = {
            guard let batchLatencies, !batchLatencies.isEmpty else { return nil }
            let sortedBatches = batchLatencies.sorted()
            let avg = sortedBatches.reduce(0, +) / Double(sortedBatches.count)
            let med = percentile(sortedBatches, percentile: 0.5)
            let p95 = percentile(sortedBatches, percentile: 0.95)
            return (avg, med, p95)
        }()

        let totalBatchDurationMs = batchLatencies?.reduce(0, +)
        let requestsPerSecond: Double?
        if let batchTotal = totalBatchDurationMs, batchTotal > 0 {
            requestsPerSecond = Double(totalRequests) / (batchTotal / 1000.0)
        } else if total > 0 {
            requestsPerSecond = Double(totalRequests) / (total / 1000.0)
        } else {
            requestsPerSecond = nil
        }

        let batchesCount: Int?
        if let batchLatencies, !batchLatencies.isEmpty {
            batchesCount = batchLatencies.count
        } else {
            batchesCount = nil
        }

        return BenchmarkStatistics(
            mode: mode,
            samples: fixtures,
            iterationsPerSample: options.iterations,
            totalRequests: totalRequests,
            averageLatencyMs: average,
            medianLatencyMs: median,
            p95LatencyMs: p95,
            minLatencyMs: minLatency,
            maxLatencyMs: maxLatency,
            charactersPerSecond: charactersPerSecond,
            wordsAnalyzed: words,
            entitiesExtracted: entities,
            batchSize: batchSize,
            batches: batchesCount,
            averageBatchLatencyMs: batchStats?.avg,
            medianBatchLatencyMs: batchStats?.median,
            p95BatchLatencyMs: batchStats?.p95,
            requestsPerSecond: requestsPerSecond
        )
    }

    private static func wordCount(in text: String) -> Int {
        text.split { !$0.isLetter && !$0.isNumber }.count
    }

    private func percentile(_ data: [Double], percentile: Double) -> Double {
        guard !data.isEmpty else { return 0 }
        let rank = (Double(data.count) - 1) * percentile
        let lowerIndex = Int(floor(rank))
        let upperIndex = Int(ceil(rank))
        if lowerIndex == upperIndex {
            return data[lowerIndex]
        }
        let fraction = rank - Double(lowerIndex)
        return data[lowerIndex] * (1 - fraction) + data[upperIndex] * fraction
    }

    private func prettyPrint(stats: BenchmarkStatistics) {
        print("\n=== Benchmark Summary ===")
        print("Mode: \(stats.mode.rawValue)")
        print(String(format: "Samples: %d", stats.samples))
        print(String(format: "Iterations per sample: %d", stats.iterationsPerSample))
        print(String(format: "Total requests: %d", stats.totalRequests))
        print(String(format: "Average latency: %.2f ms", stats.averageLatencyMs))
        print(String(format: "Median latency: %.2f ms", stats.medianLatencyMs))
        print(String(format: "p95 latency: %.2f ms", stats.p95LatencyMs))
        print(String(format: "Min latency: %.2f ms", stats.minLatencyMs))
        print(String(format: "Max latency: %.2f ms", stats.maxLatencyMs))
        print(String(format: "Characters/sec: %.2f", stats.charactersPerSecond))
        print(String(format: "Words analyzed: %d", stats.wordsAnalyzed))
        print(String(format: "Entities extracted: %d", stats.entitiesExtracted))
        if let requestsPerSecond = stats.requestsPerSecond {
            print(String(format: "Requests/sec: %.2f", requestsPerSecond))
        }
        if let batchSize = stats.batchSize {
            print(String(format: "Batch size: %d", batchSize))
        }
        if let batches = stats.batches {
            print(String(format: "Batches: %d", batches))
        }
        if let avgBatch = stats.averageBatchLatencyMs {
            print(String(format: "Average batch latency: %.2f ms", avgBatch))
        }
        if let medianBatch = stats.medianBatchLatencyMs {
            print(String(format: "Median batch latency: %.2f ms", medianBatch))
        }
        if let p95Batch = stats.p95BatchLatencyMs {
            print(String(format: "p95 batch latency: %.2f ms", p95Batch))
        }
    }

    private func write(stats: BenchmarkStatistics, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(stats)
        try data.write(to: url, options: .atomic)
    }
}

private extension Duration {
    var milliseconds: Double {
        let components = self.components
        let seconds = Double(components.seconds)
        let attoseconds = Double(components.attoseconds)
        return seconds * 1_000.0 + attoseconds / 1_000_000_000_000_000.0
    }
}

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        guard size > 0 else { return [self] }
        var result: [[Element]] = []
        var index = startIndex
        while index < endIndex {
            let end = self.index(index, offsetBy: size, limitedBy: endIndex) ?? endIndex
            result.append(Array(self[index..<end]))
            index = end
        }
        return result
    }
}
