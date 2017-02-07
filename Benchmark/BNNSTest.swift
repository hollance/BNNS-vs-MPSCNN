import Foundation
import UIKit
import Accelerate

/*
  The convolutional neural network implemented with BNNS.
*/
class BNNSTest {
  let weights: Weights
  let dataType: BNNSDataType
  let multiplier: Int

  var imageAsFloat32: [Float]!
  var imageAsFloat16: [Float16]!
  var imageAsBytes: [UInt8]!
  var imagePointer: UnsafeRawPointer!

  // These describe the formats of the data volumes that flow through the CNN.
  var inputImgDesc: BNNSImageStackDescriptor
  var conv1imgDesc: BNNSImageStackDescriptor
  var pool1imgDesc: BNNSImageStackDescriptor
  var conv2imgDesc: BNNSImageStackDescriptor
  var pool2imgDesc: BNNSImageStackDescriptor
  var fc3inputDesc: BNNSVectorDescriptor
  var fc3outputDesc: BNNSVectorDescriptor

  // The layers in our neural network.
  var conv1: BNNSFilter?
  var pool1: BNNSFilter?
  var conv2: BNNSFilter?
  var pool2: BNNSFilter?
  var fc3: BNNSFilter?

  // With BNNS we need to manage our own temporary buffers to store the
  // intermediate results computed by the various neural network layers.
  var temp1: [Float]
  var temp2: [Float]

  // The final output appears in this array.
  var results: [Float]

  // These are the weights as 16-bit floats.
  var conv1weights_16: [Float16]!
  var conv1bias_16: [Float16]!
  var conv2weights_16: [Float16]!
  var conv2bias_16: [Float16]!

  init(weights: Weights, imageName: String, dataType: BNNSDataType, multiplier: Int) {
    self.weights = weights
    self.dataType = dataType
    self.multiplier = multiplier

    // With Metal we could just load the source image into an MTLTexture but
    // we cannot use such textures directly with BNNS. Here, we load the RGBA
    // pixel data and convert it to an array of Floats that first has all the
    // R values, then all the G values, and then all the B values (instead of
    // interleaved RGBA values as in MTLTexture). This can be a somewhat costly
    // preprocessing step and I'm sure there is a more efficient way to do it
    // than we're doing here (using vImage perhaps?).

    if let image = UIImage(named: imageName) {
      imageAsBytes = image.asByteArray()
      imageAsFloat32 = [Float](repeating: 0, count: 256*256*3)
      for i in 0..<256*256 {
        imageAsFloat32[            i] = Float(imageAsBytes[i*4    ]) / 255
        imageAsFloat32[256*256   + i] = Float(imageAsBytes[i*4 + 1]) / 255
        imageAsFloat32[256*256*2 + i] = Float(imageAsBytes[i*4 + 2]) / 255
      }
    }

    temp1 = [Float](repeating: 0, count: 256*256*16*multiplier)
    temp2 = [Float](repeating: 0, count: 128*128*16*multiplier)
    results = [Float](repeating: 0, count: 100*multiplier)

    // If the data type is float16 then we have to convert the image data
    // and also the weights and bias values to 16-bit floats.
    var conv1weights = UnsafeRawPointer(weights.conv1weights)
    var conv1bias = UnsafeRawPointer(weights.conv1bias)
    var conv2weights = UnsafeRawPointer(weights.conv2weights)
    var conv2bias = UnsafeRawPointer(weights.conv2bias)
    switch dataType {
    case BNNSDataTypeFloat32:
      imagePointer = UnsafeRawPointer(imageAsFloat32)
    case BNNSDataTypeFloat16:
      imageAsFloat16 = float32to16(UnsafeMutablePointer(mutating: imageAsFloat32), count: imageAsFloat32.count)
      imagePointer = UnsafeRawPointer(imageAsFloat16)

      // Note that the fully connected layer (fc3) doesn't appear to work
      // with 16-bit floats, so we don't convert the weights for that layer.
      var temp = weights
      conv1weights_16 = float32to16(&temp.conv1weights, count: temp.conv1weights.count)
      conv1weights = UnsafeRawPointer(conv1weights_16)
      conv1bias_16 = float32to16(&temp.conv1bias, count: temp.conv1bias.count)
      conv1bias = UnsafeRawPointer(conv1bias_16)
      conv2weights_16 = float32to16(&temp.conv2weights, count: temp.conv2weights.count)
      conv2weights = UnsafeRawPointer(conv2weights_16)
      conv2bias_16 = float32to16(&temp.conv2bias, count: temp.conv2bias.count)
      conv2bias = UnsafeRawPointer(conv2bias_16)
    default:
      fatalError("Data type \(dataType) not supported")
    }

    inputImgDesc = BNNSImageStackDescriptor(width: 256, height: 256, channels: 3, row_stride: 256, image_stride: 256*256, data_type: dataType, data_scale: 0, data_bias: 0)

    conv1imgDesc = BNNSImageStackDescriptor(width: 256, height: 256, channels: 16*multiplier, row_stride: 256, image_stride: 256*256, data_type: dataType, data_scale: 0, data_bias: 0)

    pool1imgDesc = BNNSImageStackDescriptor(width: 128, height: 128, channels: 16*multiplier, row_stride: 128, image_stride: 128*128, data_type: dataType, data_scale: 0, data_bias: 0)

    conv2imgDesc = BNNSImageStackDescriptor(width: 64, height: 64, channels: 32*multiplier, row_stride: 64, image_stride: 64*64, data_type: dataType, data_scale: 0, data_bias: 0)

    pool2imgDesc = BNNSImageStackDescriptor(width: 16, height: 16, channels: 32*multiplier, row_stride: 16, image_stride: 16*16, data_type: dataType, data_scale: 0, data_bias: 0)

    fc3inputDesc = BNNSVectorDescriptor(size: 16*16*32*multiplier, data_type: BNNSDataTypeFloat32, data_scale: 0, data_bias: 0)

    fc3outputDesc = BNNSVectorDescriptor(size: 100*multiplier, data_type: BNNSDataTypeFloat32, data_scale: 0, data_bias: 0)

    // Workaround: BNNSFilterCreateConvolutionLayer() crashes if you don't give
    // it a pointer to a valid BNNSFilterParameters instance -- but it seems
    // impossible to construct a BNNSFilterParameters object in Swift. So as a
    // kludge we just allocate some all-zero memory and pretend it's a valid
    // BNNSFilterParameters object. Seems to work.
    struct FakeParams { var a = 0.0; var b = 0.0; var c = 0.0; var d = 0.0 }
    let fake = FakeParams()
    var filterParams = unsafeBitCast(fake, to: BNNSFilterParameters.self)

    // Define the objects that make up the neural network:

    let relu = BNNSActivation(function: BNNSActivationFunctionRectifiedLinear, alpha: 0, beta: 0)
    let identity = BNNSActivation(function: BNNSActivationFunctionIdentity, alpha: 0, beta: 0)
    let sigmoid = BNNSActivation(function: BNNSActivationFunctionSigmoid, alpha: 0, beta: 0)

    let conv1weightsData = BNNSLayerData(data: conv1weights, data_type: dataType, data_scale: 0, data_bias: 0, data_table: nil)
    let conv1biasData = BNNSLayerData(data: conv1bias, data_type: dataType, data_scale: 0, data_bias: 0, data_table: nil)

    // Note: unlike Metal we have to be explicit about the amount of padding
    // that is needed to make the output image the same size as the input image.

    var conv1desc = BNNSConvolutionLayerParameters(x_stride: 1, y_stride: 1, x_padding: 2, y_padding: 2, k_width: 5, k_height: 5, in_channels: 3, out_channels: 16*multiplier, weights: conv1weightsData, bias: conv1biasData, activation: relu)

    conv1 = BNNSFilterCreateConvolutionLayer(&inputImgDesc, &conv1imgDesc, &conv1desc, &filterParams)
    assert(conv1 != nil, "BNNSFilterCreateConvolutionLayer failed for layer conv1")

    let pool1bias = [Float](repeating: 0, count: 16*multiplier)
    let pool1biasData = BNNSLayerData(data: pool1bias, data_type: dataType, data_scale: 0, data_bias: 0, data_table: nil)

    var pool1desc = BNNSPoolingLayerParameters(x_stride: 2, y_stride: 2, x_padding: 0, y_padding: 0, k_width: 2, k_height: 2, in_channels: 16*multiplier, out_channels: 16*multiplier, pooling_function: BNNSPoolingFunctionMax, bias: pool1biasData, activation: identity)

    pool1 = BNNSFilterCreatePoolingLayer(&conv1imgDesc, &pool1imgDesc, &pool1desc, &filterParams)
    assert(pool1 != nil, "BNNSFilterCreateConvolutionLayer failed for layer pool1")

    let conv2weightsData = BNNSLayerData(data: conv2weights, data_type: dataType, data_scale: 0, data_bias: 0, data_table: nil)
    let conv2biasData = BNNSLayerData(data: conv2bias, data_type: dataType, data_scale: 0, data_bias: 0, data_table: nil)

    var conv2desc = BNNSConvolutionLayerParameters(x_stride: 2, y_stride: 2, x_padding: 1, y_padding: 1, k_width: 3, k_height: 3, in_channels: 16*multiplier, out_channels: 32*multiplier, weights: conv2weightsData, bias: conv2biasData, activation: relu)

    conv2 = BNNSFilterCreateConvolutionLayer(&pool1imgDesc, &conv2imgDesc, &conv2desc, &filterParams)
    assert(conv2 != nil, "BNNSFilterCreateConvolutionLayer failed for layer conv2")

    let pool2bias = [Float](repeating: 0, count: 32*multiplier)
    let pool2biasData = BNNSLayerData(data: pool2bias, data_type: dataType, data_scale: 0, data_bias: 0, data_table: nil)

    var pool2desc = BNNSPoolingLayerParameters(x_stride: 4, y_stride: 4, x_padding: 0, y_padding: 0, k_width: 2, k_height: 2, in_channels: 32*multiplier, out_channels: 32*multiplier, pooling_function: BNNSPoolingFunctionAverage, bias: pool2biasData, activation: identity)

    pool2 = BNNSFilterCreatePoolingLayer(&conv2imgDesc, &pool2imgDesc, &pool2desc, &filterParams)
    assert(pool2 != nil, "BNNSFilterCreateConvolutionLayer failed for layer pool2")

    let fc3weightsData = BNNSLayerData(data: weights.fc3weights, data_type: BNNSDataTypeFloat32, data_scale: 0, data_bias: 0, data_table: nil)
    let fc3biasData = BNNSLayerData(data: weights.fc3bias, data_type: BNNSDataTypeFloat32, data_scale: 0, data_bias: 0, data_table: nil)

    var fc3desc = BNNSFullyConnectedLayerParameters(in_size: 16*16*32*multiplier, out_size: 100*multiplier, weights: fc3weightsData, bias: fc3biasData, activation: sigmoid)

    fc3 = BNNSFilterCreateFullyConnectedLayer(&fc3inputDesc, &fc3outputDesc, &fc3desc, &filterParams)
  }

  deinit {
    BNNSFilterDestroy(conv1)
    BNNSFilterDestroy(pool1)
    BNNSFilterDestroy(conv2)
    BNNSFilterDestroy(pool2)
    BNNSFilterDestroy(fc3)
  }

  func predict() {
    autoreleasepool {
      if BNNSFilterApply(conv1, imagePointer, &temp1) != 0 {
        print("BNNSFilterApply failed on layer conv1")
      }

      if BNNSFilterApply(pool1, temp1, &temp2) != 0 {
        print("BNNSFilterApply failed on layer pool1")
      }

      if BNNSFilterApply(conv2, temp2, &temp1) != 0 {
        print("BNNSFilterApply failed on layer conv2")
      }

      if BNNSFilterApply(pool2, temp1, &temp2) != 0 {
        print("BNNSFilterApply failed on layer pool2")
      }

      // For some reason the fully-connected layer cannot handle float16
      // data, so convert the contents of temp2 to 32-bit Floats here.
      var temp3 = temp2
      if dataType == BNNSDataTypeFloat16 {
        temp3 = float16to32(&temp2, count: 16*16*32*multiplier)
      }

      if BNNSFilterApply(fc3, temp3, &results) != 0 {
        print("BNNSFilterApply failed on layer fc3")
      }

      // Note: when comparing the output of a BNNS layer to the output from a
      // Metal layer, keep in mind that the Metal output is interleaved RGBA 
      // (where some of these pixel components may not count, depending on how
      // many channels there are), while BNNS output is in planes, so first you
      // get all the values from channel 0, then all the values from channel 1,
      // and so on. (On the final output this does not matter, since the output
      // width and height are 1.)
    }
  }
}
