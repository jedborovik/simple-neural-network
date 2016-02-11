'use strict';

var math = require('forwardjs-ml-math');
var Neuron = require('./neuron.es6');

module.exports = class Layer {

  /**
   * @param {Number} size
   * @param {Number} inputs
   */
  constructor(size, inputs) {
    this.neurons = [];
    for (var i = 0; i < size; i++) {
      var neuron = new Neuron(inputs);
      this.neurons.push(neuron);
    }
  }

  /**
   * @param {Array} inputs
   * @return {Array}
   */
  forward(inputs) {
    var outputs = this.neurons.map(n => n.forward(inputs));
    return outputs;
  }

  /**
   * @param {Array}
   * @return {Array}
   */
  backwardError(errors) {
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
  }

  updateWeights() {
    this.neurons.forEach(n => n.updateWeights());
  }
}
