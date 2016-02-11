var math = require('forwardjs-ml-math');

var STEP_SIZE = 0.1;

module.exports = Neuron;

function Neuron() {
  this.weights = [];
  for (var i = 0; i < 3; i++) {
    this.weights.push(Math.random() - 0.5);
  }
}

Neuron.prototype = {

  /**
   * @param {Array} inputs
   * @return {Number}
   */
  forward: function(inputs) {
    this.inputs = inputs;
    this.z = math.arrayMultiply(inputs, this.weights);
    return math.sigmoid(this.z);
  },

  /**
   * @param {Array} error
   * @return {Array}
   */
  backwardError: function(error) {
    this.error = error;
    var backErrors = this.weights.map(function(weight) {
      return weight * error;
    });
    return backErrors.slice(1);
  },

  updateWeights: function() {
    var deltas = this.inputs.map(function(input) {
      return input * this.error * math.sigmoidGradient(this.z) * STEP_SIZE;
    }, this);
    this.weights = math.arraySubtract(this.weights, deltas);
  },
}
