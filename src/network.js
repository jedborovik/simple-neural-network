'use strict';

const Layer = require('./layer');

module.exports = class Network {

  constructor(/* layer sizes */) {
    const sizes = [...arguments];
    this.layers = [];
    for (let i = 0; i < sizes.length-1; i++) {
      const layer = new Layer(sizes[i+1], sizes[i]+1);
      this.layers.push(layer);
    }
  }

  /**
   * @param {Array} inputs
   * @return {Number}
   */
  forward(inputs) {
    return this.layers.reduce((input, layer) => {
      input = [1].concat(input); // Add bias
      return layer.forward(input);
    }, inputs);
  }

  /**
   * @param {Array} errors
   */
  backward(errors) {
    this.layers.reverse().reduce((error, layer) => {
      return layer.backward(error);
    }, errors)

    // `reverse` is in place, so reverse back.
    this.layers.reverse();
  }

  updateWeights() {
    this.layers.forEach(layer => layer.updateWeights());
  }
}
