'use strict';

var math = require('forwardjs-ml-math');

var STEP_SIZE = 0.5;

module.exports = class Neuron {
  constructor() {
    this.weights = [];
    for (var i = 0; i < 3; i++) {
      this.weights.push(Math.random() - 0.5);
    }
  }

  /**
   * @param {Array} inputs
   * @return {Number}
   */
  forward(inputs) {
    // Save inputs for updating the weights later.
    this.inputs = inputs;

    this.z = math.arrayMultiply(inputs, this.weights);
    return math.sigmoid(this.z);
  }

  /**
   * @param {Number} error
   */
  updateWeights(error) {
    var deltas = this.inputs.map(input =>
      input * error * math.sigmoidGradient(this.z) * STEP_SIZE
    );
    this.weights = math.arraySubtract(this.weights, deltas);
  }
}
