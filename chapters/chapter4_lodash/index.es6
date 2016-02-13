const math = require('forwardjs-ml-math');
const mnist = require('mnistjs');
const Network = require('./network.es6');
const _ = require('lodash');

const training = mnist.training.slice(0, 1000); // 1,000 training samples
const testing = mnist.testing; // 10,000 testing samples
const network = new Network(400, 25, 10);
const MAX_ITER = 10000;
const trainingLength = training.length;

const randomNum = () => Math.floor(Math.random() * trainingLength)

const numCorrect = (data) =>
  _.sumBy(data, (d) =>
    maxElemIndex(network.forward(d.input)) === maxElemIndex(d.output)
  )

const accuracy = (data) =>
  numCorrect(data) / data.length;

const logAccuracy = (iter) => {
  if (iter % 2500 === 0) console.log(`Training accuracy at iter ${iter}: ${accuracy(training)}`);
}

const delta = (networkOutput,trainingOutput) =>
  math.arraySubtract(networkOutput, trainingOutput)

console.time('total computation time:')
var i;
_.times(MAX_ITER, iter => {
  i = randomNum()
  network.backwardError(
    delta(network.forward(training[i].input), training[i].output)
  )
  network.updateWeights()
  logAccuracy(iter)
})

console.log(`Training accuracy at iter ${MAX_ITER}: ${accuracy(training)}`);
console.log(`\nTest accuracy: ${accuracy(testing)}`);
console.timeEnd('total computation time:');

/**
 * @param {Array} data
 * @return {Number} The index of the greatest element.
 */
function maxElemIndex(arr) {
  return _.indexOf(arr, _.max(arr));
}
