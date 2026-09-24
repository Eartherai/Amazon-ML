// Generic Foundation/ICU transliteration; audit probe only, no external lookup.
import Foundation
while let line = readLine() {
    guard let data = line.data(using: .utf8),
          let value = try JSONSerialization.jsonObject(with: data) as? [String] else { fatalError("Expected JSON string array") }
    let transformed = value.map { $0.applyingTransform(StringTransform("Any-Latin; Latin-ASCII"), reverse: false) ?? $0 }
    let result = try JSONSerialization.data(withJSONObject: transformed, options: [.fragmentsAllowed])
    print(String(data: result, encoding: .utf8)!)
}
