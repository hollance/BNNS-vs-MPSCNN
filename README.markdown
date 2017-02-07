# BNNS vs Metal CNN benchmark

This app compares the speed of Apple's two deep learning frameworks: BNNS and Metal Performance Shaders (MPSCNN).

It creates a basic convolutional neural network with 2 convolutional layers, 2 pooling layers, and a fully-connected layer. Then it measures how long it takes to sends the same image 100 times through the network.

To run the app you need Xcode 8 and an iOS 10-compatible device with at least an A8 processor.

See also the [blog post](http://machinethink.net/blog/apple-deep-learning-bnns-versus-metal-cnn/).
