import SwiftUI
import GLiNERSwift

@main
struct GLiNERDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = GLiNERViewModel()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Input Section
                VStack(alignment: .leading, spacing: 10) {
                    Text("Input Text")
                        .font(.headline)
                    
                    TextEditor(text: $viewModel.inputText)
                        .frame(height: 150)
                        .border(Color.gray, width: 1)
                        .disabled(viewModel.isProcessing)
                }
                .padding(.horizontal)
                
                // Labels Section
                VStack(alignment: .leading, spacing: 10) {
                    Text("Entity Labels (comma-separated)")
                        .font(.headline)
                    
                    TextField("person, organization, location", text: $viewModel.labelsText)
                        .textFieldStyle(.roundedBorder)
                        .disabled(viewModel.isProcessing)
                }
                .padding(.horizontal)
                
                // Threshold Slider
                VStack(alignment: .leading, spacing: 10) {
                    Text("Confidence Threshold: \(String(format: "%.2f", viewModel.threshold))")
                        .font(.headline)
                    
                    Slider(value: $viewModel.threshold, in: 0.0...1.0, step: 0.05)
                        .disabled(viewModel.isProcessing)
                }
                .padding(.horizontal)
                
                // Extract Button
                Button(action: {
                    Task {
                        await viewModel.extractEntities()
                    }
                }) {
                    HStack {
                        if viewModel.isProcessing {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .tint(.white)
                        }
                        Text(viewModel.isProcessing ? "Processing..." : "Extract Entities")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isProcessing ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(viewModel.isProcessing || viewModel.inputText.isEmpty)
                .padding(.horizontal)
                
                // Results Section
                if !viewModel.entities.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Extracted Entities (\(viewModel.entities.count))")
                            .font(.headline)
                        
                        List(viewModel.entities, id: \.text) { entity in
                            VStack(alignment: .leading, spacing: 5) {
                                HStack {
                                    Text(entity.text)
                                        .font(.body)
                                        .bold()
                                    Spacer()
                                    Text(String(format: "%.2f", entity.score))
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                                
                                Text(entity.label)
                                    .font(.caption)
                                    .padding(4)
                                    .background(labelColor(for: entity.label))
                                    .foregroundColor(.white)
                                    .cornerRadius(4)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .padding(.horizontal)
                }
                
                // Error Message
                if let error = viewModel.errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding()
                }
                
                Spacer()
            }
            .navigationTitle("GLiNER Demo")
            .task {
                await viewModel.initialize()
            }
        }
    }
    
    private func labelColor(for label: String) -> Color {
        let colors: [String: Color] = [
            "person": .blue,
            "organization": .green,
            "location": .orange,
            "date": .purple,
            "event": .pink
        ]
        return colors[label.lowercased()] ?? .gray
    }
}

@MainActor
class GLiNERViewModel: ObservableObject {
    @Published var inputText: String = "John Smith works at Apple Inc. in Cupertino, California."
    @Published var labelsText: String = "person, organization, location"
    @Published var threshold: Double = 0.5
    @Published var entities: [Entity] = []
    @Published var isProcessing: Bool = false
    @Published var errorMessage: String?
    
    private var model: GLiNERModel?
    
    func initialize() async {
        do {
            guard let manifestURL = locateManifest() else {
                errorMessage = "Unable to find CoreMLArtifacts/export_manifest.json. Set GLINER_MANIFEST_PATH or place the app inside the repo."
                return
            }
            model = try await GLiNERModel(manifestURL: manifestURL)
            errorMessage = nil
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }
    
    func extractEntities() async {
        guard let model = model else {
            errorMessage = "Model not initialized"
            return
        }
        
        isProcessing = true
        errorMessage = nil
        entities = []
        
        do {
            let labels = labelsText
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            
            guard !labels.isEmpty else {
                errorMessage = "Please enter at least one label"
                isProcessing = false
                return
            }
            
            let extractedEntities = try await model.extractEntities(
                from: inputText,
                labels: labels,
                threshold: Float(threshold)
            )
            
            entities = extractedEntities.sorted { $0.score > $1.score }
            
        } catch {
            errorMessage = "Extraction failed: \(error.localizedDescription)"
        }
        
        isProcessing = false
    }
}

private extension GLiNERViewModel {
    func locateManifest() -> URL? {
        let fm = FileManager.default
        let processEnv = ProcessInfo.processInfo.environment
        if let envPath = processEnv["GLINER_MANIFEST_PATH"], !envPath.isEmpty {
            let url = URL(fileURLWithPath: envPath).standardizedFileURL
            if fm.fileExists(atPath: url.path) {
                return url
            }
        }

        if let bundled = Bundle.main.url(forResource: "export_manifest", withExtension: "json"),
           fm.fileExists(atPath: bundled.path) {
            return bundled
        }

        var bundleURL = Bundle.main.bundleURL
        for _ in 0..<10 {
            let candidate = bundleURL
                .appendingPathComponent("CoreMLArtifacts")
                .appendingPathComponent("export_manifest.json")
            if fm.fileExists(atPath: candidate.path) {
                return candidate
            }
            bundleURL.deleteLastPathComponent()
        }

        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("CoreMLArtifacts")
            .appendingPathComponent("export_manifest.json")
        if fm.fileExists(atPath: cwd.path) {
            return cwd
        }

        return nil
    }
}

#Preview {
    ContentView()
}
