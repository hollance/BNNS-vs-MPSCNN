import Foundation

/*
  The design of the convolutional neural network used for testing:

    conv1: 256x256x3 input, 16 5x5 kernels, stride 1
    pool1: 256x256x16 input, stride 2
    conv2: 128x128x16 input, 32 3x3 kernels, stride 2
    pool2: 64x64x32 input, stride 4
    fc3:   16x16x32 input, 100 neurons

  The network takes a 256x256 RGB image (no alpha channel) as input and 
  produces an array with 100 Floats.

  The "multiplier" is used to quickly try networks of different size without
  having to add more layers. It simply makes the number of channels larger,
  so a multiplier of 2 gives conv1 32 channels instead of 16, fc3 200 neurons
  instead of 100, and so on.
*/

/* 
  The weights store the things that the neural network has learned during
  training. Since we don't really care what this particular network computes,
  we'll just use reasonable random values.
*/
struct Weights {
  let multiplier: Int

  var conv1weights: [Float]
  var conv1bias   : [Float]
  var conv2weights: [Float]
  var conv2bias   : [Float]
  var fc3weights  : [Float]
  var fc3bias     : [Float]

  init(multiplier: Int, transpose: Bool = false) {
    self.multiplier = multiplier

    // We use fixed random seeds so that the results are reproducible.
    conv1weights = randomWeights(count: 5*5*3*16*multiplier, seed: 123)
    conv1bias    = randomWeights(count: 16*multiplier, seed: 456)
    conv2weights = randomWeights(count: 3*3*16*multiplier*32*multiplier, seed: 789)
    conv2bias    = randomWeights(count: 32*multiplier, seed: 321)
    fc3weights   = randomWeights(count: 16*16*32*multiplier*100*multiplier, seed: 654)
    fc3bias      = randomWeights(count: 100*multiplier, seed: 987)

    // This is needed for the BNNS version of the network.
    if transpose {
      conv1weights = transposeWeights(conv1weights, kernelWidth: 5, kernelHeight: 5, inputChannels: 3, outputChannels: 16*multiplier)
      conv2weights = transposeWeights(conv2weights, kernelWidth: 3, kernelHeight: 3, inputChannels: 16*multiplier, outputChannels: 32*multiplier)
      fc3weights = transposeWeights(fc3weights, kernelWidth: 16, kernelHeight: 16, inputChannels: 32*multiplier, outputChannels: 100*multiplier)
    }
  }
}

func randomWeights(count: Int, seed: Int) -> [Float] {
  srand48(seed)
  var a = [Float](repeating: 0, count: count)
  for i in 0..<count {
    a[i] = Float(drand48() - 0.5) * 0.3
  }
  return a
}

/*
  Metal expects the weights to be organized like this in memory:

      weights[ output_channel ][ kernel_y ][ kernel_x ][ input_channel ]

  but BNNS expects them in a different order:

      weights[ output_channel ][ input_channel ][ kernel_y ][ kernel_x ]

  We use this function to rearrange the weights so that both the Metal and
  BNNS tests use the exact same weights for their computations. This is not 
  really important for the speed test, but it lets us make sure that both
  APIs produce (roughly) the same outputs.
*/
private func transposeWeights(_ weights: [Float], kernelWidth: Int, kernelHeight: Int, inputChannels: Int, outputChannels: Int) -> [Float] {
  var transposed = [Float](repeating: 0, count: weights.count)
  var i = 0
  for oc in 0..<outputChannels {
    for ky in 0..<kernelHeight {
      for kx in 0..<kernelWidth {
        for ic in 0..<inputChannels {
          transposed[kx + kernelWidth*(ky + kernelHeight*(ic + inputChannels*oc))] = weights[i]
          i += 1
        }
      }
    }
  }
  return transposed
}
