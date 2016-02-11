'use strict';

var math = require('forwardjs-ml-math');

var STEP_SIZE = 0.1;

module.exports = class Neuron {

  /**
   * @param {Number} n
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
   * @param {Number} error
   * @return {Array}
   */
  backwardError(error) {
    this.error = error;
    var backErrors = this.weights.map(w => w * error);
    return backErrors.slice(1);
  }
  
  updateWeights() {
    var deltas = this.inputs.map(input =>
      input * this.error * math.sigmoidGradient(this.z) * STEP_SIZE
    );
    this.weights = math.arraySubtract(this.weights, deltas);
  }
}
