import MetalPerformanceShaders

extension MPSImage {
  /* 
    We receive the predicted output as an MPSImage. We need to convert this
    to an array of Floats that we can use from Swift.

    Because Metal is a graphics API, MPSImage stores the data in MTLTexture 
    objects. Each pixel from the texture stores 4 channels: R contains the 
    first channel, G is the second channel, B is the third, A is the fourth. 

    In addition, these individual R,G,B,A pixel components can be stored as 
    float16, in which case we also have to convert the data type.
  */
  public func toFloatArray() -> [Float] {
    switch pixelFormat {
      case .rgba16Float: return fromFloat16()
      case .rgba32Float: return fromFloat32()
      default: fatalError("Pixel format \(pixelFormat) not supported")
    }
  }

  private func fromFloat16() -> [Float] {
    var outputFloat16 = convert(initial: Float16(0))
    return float16to32(&outputFloat16, count: outputFloat16.count)
  }

  private func fromFloat32() -> [Float] {
    return convert(initial: Float(0))
  }

  private func convert<T>(initial: T) -> [T] {
    precondition(featureChannels > 3, "This code crashes on MPSImages with <= 3 channels")

    let count = width * height * featureChannels
    var output = [T](repeating: initial, count: count)

    let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: width, height: height, depth: 1))

    let numSlices = (featureChannels + 3)/4
    for i in 0..<numSlices {
      texture.getBytes(&(output[width * height * 4 * i]),
                       bytesPerRow: width * 4 * MemoryLayout<T>.size,
                       bytesPerImage: 0,
                       from: region,
                       mipmapLevel: 0,
                       slice: i)
    }
    return output
  }
}
