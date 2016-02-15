  'use strict';

const math = require('forwardjs-ml-math');
const _ = require('lodash');

const STEP_SIZE = 0.1;

const delta = (input, error, z) =>
  input * error * math.sigmoidGradient(z) * STEP_SIZE

module.exports = class Neuron {

  /**
   * @param {Number} n
    */
  constructor(n) {
    this.weights = _.times(n, () => Math.random() - 0.5)
  }

  /**
   * @param {Array} inputs
   * @return {Number}
   */
  forward(inputs) {
    this.inputs = inputs;
    this.z = math.arrayMultiply(inputs, this.weights)
    return math.sigmoid(this.z);
  }

  /**
   * @param {Number} error
   * @return {Array}
   */
  backwardError(error) {
    this.error = error;
    return _.map(this.weights, (w) => w * error).slice(1)
  }

  updateWeights() {
    this.weights = _.map(this.weights, (weight,i) =>
      weight - delta(this.inputs[i], this.error, this.z)
    )
  }

}
