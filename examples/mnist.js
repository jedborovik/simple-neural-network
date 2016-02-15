var mnist = require('mnistjs');
var Network = require('../src/network');
var math = require('../src/math');

var training = mnist.training.slice(0, 1000);
var testing = mnist.testing;

var network = new Network(400, 25, 10);

var MAX_ITER = 5000;
for (var iter = 0; iter < MAX_ITER; iter++) {
  var i = Math.floor(Math.random() * training.length);

  var input = training[i].input;
  var output = training[i].output;

  var hs = network.forward(input);
  var outputError = math.arraySubtract(hs, output);

  network.backward(outputError);
  network.updateWeights();

  if (iter % (MAX_ITER / 100) === 0) process.stdout.write('.')
}

var correct = 0;
for (var i = 0; i < testing.length; i++) {
  var input = testing[i].input;
  var hs = network.forward(input);
  if (maxElemIndex(hs) === testing[i].label) correct++;
}

console.log('\nTest accuracy:', correct / testing.length);

/**
 * @return {Number} The index of the greatest element.
 */
function maxElemIndex(arr) {
  var index = 0;
  for (var i = 1; i < arr.length; i++) {
    if (arr[index] < arr[i]) index = i;
  }
  return index;
}
