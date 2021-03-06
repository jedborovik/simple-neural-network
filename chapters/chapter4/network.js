var Layer = require('./layer.js');

module.exports = Network;

/**
 * @param {Number} inputLayerSize
 * @param {Number} hiddenLayerSize
 * @param {Number} outputLayerSize
 */
function Network(inputLayerSize, hiddenLayerSize, outputLayerSize) {
  this.hiddenLayer = new Layer(hiddenLayerSize, inputLayerSize+1);
  this.outputLayer = new Layer(outputLayerSize, hiddenLayerSize+1);
}

Network.prototype = {

  /**
   * @param {Array} inputs The inputs to the network.
   * @return {Array} The output of the network.
   */
  forward: function(inputs) {
    var hiddenLayerInputs = [1].concat(inputs);
    var hiddenOutput = this.hiddenLayer.forward(hiddenLayerInputs);

    var outputLayerInputs = [1].concat(hiddenOutput);
    var output = this.outputLayer.forward(outputLayerInputs);

    return output;
  },

  /**
   * @param {Array} errors The errors of the network's output.
   */
  backwardError: function(errors) {
    var hiddenLayerErrors = this.outputLayer.backwardError(errors);

    // Ignore the error that the hidden layer passes
    // back. This corresponds to the input, which we
    // cannot change.
    this.hiddenLayer.backwardError(hiddenLayerErrors);
  },

  updateWeights: function() {
    this.outputLayer.updateWeights();
    this.hiddenLayer.updateWeights();
  }
}
