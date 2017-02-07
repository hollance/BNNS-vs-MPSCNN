import UIKit
import QuartzCore
import Accelerate
import MetalPerformanceShaders

class ViewController: UIViewController {

  @IBOutlet weak var textView: UITextView!

  var firstTime = true

  override func viewDidLoad() {
    super.viewDidLoad()
    textView.text = ""
  }

  func log(message: String) {
    let text = textView.text + message + "\n"
    textView.text = text
  }

  func timeIt(_ code: () -> Void) {
    let startTime = CACurrentMediaTime()
    code()
    let endTime = CACurrentMediaTime()
    log(message: "Elapsed time: \(endTime - startTime) sec")
  }

  private var metal: MetalTest!
  private var bnns: BNNSTest!

  @IBAction func testButtonPressed(_ sender: UIButton) {
    // Note: we're running this stuff on the main thread so the UI becomes
    // unresponsive for a few seconds. Don't panic, this is as intended. :-)

    /*
      Here you can change what gets tested:
      
      - data type: Metal's .float16/32 versus BNNSDataTypeFloat16/32
      - size of the network: use "multiplier" to give each layer more channels,
        which increases the amount of data used in the computations
      - change the number of iterations
    */
    let imageName = "Floortje"
    let metalFormat = MPSImageFeatureChannelFormat.float16   // 16 or 32
    let bnnsFormat = BNNSDataTypeFloat16                     // 16 or 32
    let multiplier = 1                                       // try 1, 2, 4
    let iterations = 100

    // Create the neural networks and measure how long this takes. Startup
    // time is not really that important but I was curious.

    let metalWeights = Weights(multiplier: multiplier)
    let bnnsWeights = Weights(multiplier: multiplier, transpose: true)

    log(message: "Setting up neural network with Metal...")
    timeIt {
      metal = MetalTest(weights: metalWeights, imageName: imageName, channelFormat: metalFormat, multiplier: multiplier)
    }

    log(message: "\nSetting up neural network with BNNS...")
    timeIt {
      bnns = BNNSTest(weights: bnnsWeights, imageName: imageName, dataType: bnnsFormat, multiplier: multiplier)
    }

    // To get a reasonable average time estimate, we run the inference step
    // many times in a row.

    log(message: "\nPerforming inference with Metal...")
    timeIt {
      for _ in 0..<iterations {
        metal.predict()
      }
    }

    log(message: "\nPerforming inference with BNNS...")
    timeIt {
      for _ in 0..<iterations {
        bnns.predict()
      }
    }

    // To make sure we're really comparing the speeds of two identical neural
    // networks, both Metal and BNNS should output the same results, allowing
    // for small differences due to rounding errors. Precision only goes up to
    // 3 decimals because Metal always uses float16s internally.
    if firstTime {
      firstTime = false
      log(message: "\nComparing output of both APIs:")
      for i in 0..<metal.results.count {
        let diff = metal.results[i] - bnns.results[i]
        let warn = (abs(diff) > 0.001) ? "!!! ERROR !!!" : ""
        log(message: "\(i): \(metal.results[i])   difference = \(diff)   \(warn)")
      }
    }

    log(message: "\n*** Done! ***")
  }
}
