import Foundation
import MetalPerformanceShaders
import MetalKit

/*
  The convolutional neural network implemented with Metal Performance Shaders.
*/
class MetalTest {
  let weights: Weights
  let multiplier: Int

  var device: MTLDevice!
  let commandQueue: MTLCommandQueue
  let texture: MTLTexture

  // These describe the formats of the data volumes that flow through the CNN.
  let conv1imgDesc: MPSImageDescriptor
  let pool1imgDesc: MPSImageDescriptor
  let conv2imgDesc: MPSImageDescriptor
  let pool2imgDesc: MPSImageDescriptor
  let outputImgDesc: MPSImageDescriptor

  // The layers in our neural network.
  let conv1: MPSCNNConvolution
  let pool1: MPSCNNPoolingMax
  let conv2: MPSCNNConvolution
  let pool2: MPSCNNPoolingAverage
  let fc3: MPSCNNFullyConnected

  // All data that goes through MPSCNN layers needs to be in the form of an 
  // MPSImage or an MPSTemporaryImage. We allocate the MPSImages ahead of time.
  let inputImage: MPSImage
  let outputImage: MPSImage

  // The final output appears in this array.
  var results = [Float](repeating: 0, count: 100)

  init(weights: Weights, imageName: String, channelFormat: MPSImageFeatureChannelFormat, multiplier: Int) {
    self.weights = weights
    self.multiplier = multiplier

    device = MTLCreateSystemDefaultDevice()
    guard device != nil else {
      fatalError("Error: This device does not support Metal")
    }

    guard MPSSupportsMTLDevice(device) else {
      fatalError("Error: This device does not support Metal Performance Shaders")
    }

    commandQueue = device.makeCommandQueue()

    // Load the 256x256-pixel source image into a Metal texture.
    let textureLoader = MTKTextureLoader(device: device)
    let url = Bundle.main.url(forResource: imageName, withExtension: "png")
    texture = try! textureLoader.newTexture(withContentsOf: url!, options: [
      MTKTextureLoaderOptionSRGB : NSNumber(value: false)
    ])

    // NOTE: For some reason, MPSCNN gives an error on the next layer when the
    // pool layer images are not .float16.
    conv1imgDesc  = MPSImageDescriptor(channelFormat: channelFormat, width: 256, height: 256, featureChannels:  16*multiplier)
    pool1imgDesc  = MPSImageDescriptor(channelFormat: .float16,      width: 128, height: 128, featureChannels:  16*multiplier)
    conv2imgDesc  = MPSImageDescriptor(channelFormat: channelFormat, width:  64, height:  64, featureChannels:  32*multiplier)
    pool2imgDesc  = MPSImageDescriptor(channelFormat: .float16,      width:  16, height:  16, featureChannels:  32*multiplier)
    outputImgDesc = MPSImageDescriptor(channelFormat: channelFormat, width:   1, height:   1, featureChannels: 100*multiplier)

    // For MPSCNN your input data must be put into an MSPImage object, which 
    // really means it must be put into one or more MTLTextures. Most of the 
    // time you will be using CNNs with images so this is no problem. Note that
    // the pixel format of this texture is bgra8Unorm_srgb; Metal will convert
    // the texture pixels to floats automatically.
    inputImage = MPSImage(texture: texture, featureChannels: 3)
    outputImage = MPSImage(device: device, imageDescriptor: outputImgDesc)

    // Define the objects that make up the neural network:

    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let sigmoid = MPSCNNNeuronSigmoid(device: device)

    let conv1desc = MPSCNNConvolutionDescriptor(kernelWidth: 5, kernelHeight: 5, inputFeatureChannels: 3, outputFeatureChannels: 16*multiplier, neuronFilter: relu)

    conv1 = MPSCNNConvolution(device: device, convolutionDescriptor: conv1desc, kernelWeights: weights.conv1weights, biasTerms: weights.conv1bias, flags: .none)

    pool1 = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 2, strideInPixelsY: 2)

    // Note: To get the same results as the BNNS pooling layer, we have to
    // change where in the MPSImage the pooling layer begins reading data.
    pool1.offset = MPSOffset(x: 1, y: 1, z: 0)

    let conv2desc = MPSCNNConvolutionDescriptor(kernelWidth: 3, kernelHeight: 3, inputFeatureChannels: 16*multiplier, outputFeatureChannels: 32*multiplier, neuronFilter: relu)
    conv2desc.strideInPixelsX = 2
    conv2desc.strideInPixelsY = 2

    conv2 = MPSCNNConvolution(device: device, convolutionDescriptor: conv2desc, kernelWeights: weights.conv2weights, biasTerms: weights.conv2bias, flags: .none)

    pool2 = MPSCNNPoolingAverage(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 4, strideInPixelsY: 4)
    pool2.offset = MPSOffset(x: 1, y: 1, z: 0)

    let fc3desc = MPSCNNConvolutionDescriptor(kernelWidth: 16, kernelHeight: 16, inputFeatureChannels: 32*multiplier, outputFeatureChannels: 100*multiplier, neuronFilter: sigmoid)

    fc3 = MPSCNNFullyConnected(device: device, convolutionDescriptor: fc3desc, kernelWeights: weights.fc3weights, biasTerms: weights.fc3bias, flags: .none)
  }

  func predict() {
    autoreleasepool {
      let commandBuffer = commandQueue.makeCommandBuffer()

      // This lets us squeeze some extra speed out of Metal.
      MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: [
        conv1imgDesc, pool1imgDesc, conv2imgDesc, pool2imgDesc ])

      let conv1img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1imgDesc)
      conv1.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: conv1img)

      let pool1img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool1imgDesc)
      pool1.encode(commandBuffer: commandBuffer, sourceImage: conv1img, destinationImage: pool1img)

      let conv2img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2imgDesc)
      conv2.encode(commandBuffer: commandBuffer, sourceImage: pool1img, destinationImage: conv2img)

      let pool2img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool2imgDesc)
      pool2.encode(commandBuffer: commandBuffer, sourceImage: conv2img, destinationImage: pool2img)

      fc3.encode(commandBuffer: commandBuffer, sourceImage: pool2img, destinationImage: outputImage)

      // Tell the GPU to start, and wait until it's done.
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }

    // The output of MPSCNN is also an MPSImage, but this time the data usually
    // no longer represents pixels. We want to convert the data to a regular
    // array of Floats that we can use from Swift. It is fair to include this 
    // conversion in the speed test, since it's a disavantage of using the GPU
    // and BNNS does not need such a conversion step at the end.
    results = outputImage.toFloatArray()
    assert(results.count == 100*multiplier)
  }
}
