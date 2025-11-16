# GLiNER Demo App

A simple SwiftUI demo application showcasing GLiNERSwift entity extraction capabilities.

## Features

- Text input for entity extraction
- Customizable entity labels
- Adjustable confidence threshold
- Real-time entity extraction
- Visual results display with color-coded labels

## Running the Demo

### Option 1: Xcode

1. Open the GLiNERSwift package in Xcode
2. Select the GLiNERDemo scheme
3. Choose a simulator or device
4. Run (⌘R)

### Option 2: Command Line

```bash
cd Examples/GLiNERDemo
swift build
swift run
```

## Usage

1. Enter or modify text in the input field
2. Specify entity labels (comma-separated)
3. Adjust the confidence threshold slider
4. Tap "Extract Entities"
5. View results with scores and labels

## Example Inputs

### Business Text
```
Text: "Tim Cook presented the new iPhone at the Apple Park in California."
Labels: person, organization, location, product
```

### Academic Text
```
Text: "Dr. Jane Smith published her research in Nature journal."
Labels: person, publication, title
```

### Technical Text
```
Text: "The API returns JSON data with a 200 status code."
Labels: technology, data format, status code
```

## Requirements

- iOS 16.0+ / macOS 13.0+
- Xcode 15.0+
- GLiNERSwift package with bundled model

## Note

This is a demonstration app. For production use, consider:
- Error handling improvements
- Caching for better performance
- UI/UX enhancements
- Accessibility features
