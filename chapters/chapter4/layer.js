var math = require('forwardjs-ml-math');
var Neuron = require('./neuron.js');

module.exports = Layer;

/**
 * @param {Number} size
 * @param {Number} inputs
 */
function Layer(size, inputs) {
  this.neurons = [];
  for (var i = 0; i < size; i++) {
    var neuron = new Neuron(inputs);
    this.neurons.push(neuron);
  }
}

Layer.prototype = {

  /**
   * @param {Array} inputs
   * @return {Array}
   */
  forward: function(inputs) {
    var outputs = this.neurons.map(function(neuron) {
      return neuron.forward(inputs);
    });
    return outputs;
  },

  /**
   * @param {Array}
   * @return {Array}
   */
  backwardError: function(errors) {
    var allBackwardErrors = [];
    for (var i = 0; i < this.neurons.length; i++) {
      var neuron = this.neurons[i];
      var error = errors[i];
      var backwardError = neuron.backwardError(error);
      allBackwardErrors.push(backwardError);
    }

    var totalBackwardError = allBackwardErrors[0];
    for (var i = 1; i < allBackwardErrors.length; i++) {
      totalBackwardError = math.arrayAdd(
        totalBackwardError,
        allBackwardErrors[i]
      );
    }
    return totalBackwardError;
  },

  updateWeights: function() {
    this.neurons.forEach(function(n) {
      n.updateWeights();
    });
  }
}
