'use strict';

const Neuron = require('./neuron');
const math = require('./math');

module.exports = class Layer {

  /**
   * @param {Number} size
   * @param {Number} inputs
   */
  constructor(size, inputs) {
    this.neurons = [];
    for (let i = 0; i < size; i++) {
      this.neurons.push(new Neuron(inputs));
    }
  }

  /**
   * @param {Array} inputs
   * @return {Number}
   */
  forward(inputs) {
    return this.neurons.map(neuron => neuron.forward(inputs));
  }

  /**
   * @param {Array} errors
   * @return {Array}
   */
  backward(errors) {
    var backErrors = this.neurons.map((neuron, i) =>
      neuron.backward(errors[i])
    );
    return backErrors.reduce(math.arrayAdd);
  }

  updateWeights() {
    this.neurons.forEach(neuron => neuron.updateWeights());
  }
}
