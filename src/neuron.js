'use strict';

const math = require('./math');

const STEP_SIZE = 0.1;

module.exports = class Neuron {

  /**
   * @param {Number} n The number of inputs
   */
  constructor(n) {
    this.weights = [];
    for (var i = 0; i < n; i++) {
      this.weights.push(Math.random() - 0.5);
    }
  }

  /**
   * @param {Array} inputs
   * @return {Number}
   */
  forward(inputs) {
    this.inputs = inputs;
    this.z = math.arrayMultiply(inputs, this.weights);
    return math.sigmoid(this.z);
  }

  /**
   * @param {Number}
   * @return {Array}
   */
  backward(error) {
    this.error = error;
    var backErrors = this.weights.map(w => w * error);

    // Don't return bias error.
    return backErrors.slice(1);
  }

  updateWeights() {
    const deltas = this.inputs.map(input =>
      input * math.sigmoidGradient(this.z) * this.error * STEP_SIZE
    );
    this.weights = math.arraySubtract(this.weights, deltas);
  }

}
