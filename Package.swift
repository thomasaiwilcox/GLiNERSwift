// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GLiNERSwift",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "GLiNERSwift",
            targets: ["GLiNERSwift"]
        ),
        .executable(
            name: "gliner-benchmarks",
            targets: ["Benchmarks"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "GLiNERSwift",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics")
            ],
            resources: [
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "GLiNERSwiftTests",
            dependencies: ["GLiNERSwift"],
            resources: [
                .copy("Fixtures")
            ]
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["GLiNERSwift"],
            resources: [
                .process("Resources")
            ]
        )
    ]
)
