var math = require('forwardjs-ml-math');
var mnist = require('mnistjs');
var Network = require('./network.js');
// var Network = require('./network.es6');

var training = mnist.training.slice(0, 1000); // 1,000 training samples
var testing = mnist.testing; // 10,000 testing samples

var network = new Network(400, 25, 10);

console.log('starting');
console.time('total computation time:')
var MAX_ITER = 10000;
for (var iter = 0; iter < MAX_ITER; iter++) {
  var i = Math.floor(Math.random() * training.length);

  var input = training[i].input;
  var output = training[i].output;

  var hs = network.forward(input);
  var outputError = math.arraySubtract(hs, output);

  network.backwardError(outputError);
  network.updateWeights();

  if (iter % 2500 === 0) {
    console.log('Training accuracy at iter %s: %s', iter, accuracy(training));
  }

}

console.log('Training accuracy at iter %s: %s', MAX_ITER, accuracy(training));
console.log('\nTest accuracy:', accuracy(testing));
console.timeEnd('total computation time:')

/**
 * @param {Array} data
 * @return {Number}
 */
function accuracy(data) {
  var correct = 0;
  for (var i = 0; i < data.length; i++) {
    var input = data[i].input;
    var output = data[i].output;

    var hs = network.forward(input);

    if (maxElemIndex(hs) === maxElemIndex(output)) correct++;
  }
  return correct / data.length;
}

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
