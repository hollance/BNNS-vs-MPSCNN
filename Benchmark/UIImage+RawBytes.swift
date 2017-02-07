import UIKit
import QuartzCore

extension UIImage {
  /* Converts the image into an array of bytes for the RGBA pixels */
  func asByteArray() -> [UInt8] {
    let width = Int(size.width)
    let height = Int(size.height)

    var bytes = [UInt8](repeating: 0, count: width * height * 4)

    bytes.withUnsafeMutableBytes { ptr in
      if let context = CGContext(
                    data: ptr.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {

        if let image = self.cgImage {
          let rect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
          context.draw(image, in: rect)
        }
      }
    }
    return bytes
  }
}
