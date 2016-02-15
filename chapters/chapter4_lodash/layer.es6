'use strict';

const math = require('forwardjs-ml-math');
const Neuron = require('./neuron.es6');
const _ = require('lodash')

module.exports = class Layer {
  /**
   * @param {Number} size
   * @param {Number} inputs
   */
  constructor(size, inputs) {
    this.neurons = _.times(size, n => new Neuron(inputs))
  }

  /**
   * @param {Array} inputs
   * @return {Array}
   */
  forward(inputs) {
    return _.map(this.neurons, (n) => n.forward(inputs));
  }

  /**
   * @param {Array}
   * @return {Array}
   */
  backwardError(errors) {
    return _(this.neurons)
            .map((neuron,i) => neuron.backwardError(errors[i]))
            .reduce((a,b) => math.arrayAdd(a,b))
  }

  updateWeights() {
    this.neurons.forEach(n => n.updateWeights());
  }
}
