const math = require('forwardjs-ml-math');
const mnist = require('mnistjs');
const Network = require('./network.es6');
const _ = require('lodash');

const training = mnist.training.slice(0, 1000); // 1,000 training samples
const testing = mnist.testing; // 10,000 testing samples
const network = new Network(400, 25, 10);
const MAX_ITER = 10000;
const trainingLength = training.length;

/**
 * @param {Array} networkOutput
 * @param {Array} output
 * @return {Boolean} Does index of the greatest elements match?
 */
const correct = (networkOutput, output) =>
  _.indexOf(networkOutput, _.max(networkOutput)) === _.indexOf(output, _.max(output))

/**
 * @param {Array} data
 * @return {Number} Number of elements classified correctly
 */
const numCorrect = (data) =>
  _.sumBy(data, (d) => correct(network.forward(d.input), d.output))

/**
 * @param {Array} data
 * @return {Number} Percentage of elements classified correctly
 */
const accuracy = (data) => numCorrect(data) / data.length;

/**
 * @param {Array} networkOutput
 * @param {Array} trainingOutput
 * @return {Array} Difference in value of elements via index.
 */
const getErrors = (networkOutput, trainingOutput) =>
  math.arraySubtract(networkOutput, trainingOutput)

/**
 * @return {Number} Random index of training set
 */
const randomNum = () =>
  Math.floor(Math.random() * trainingLength)

const logAccuracy = (iter) => {
  if (iter % 2500 === 0) console.log(`Training accuracy at iter ${iter}: ${accuracy(training)}`);
}

console.time('total computation time:')

var i;
_.times(MAX_ITER, iter => {
  i = randomNum()

  network.backwardError(
    getErrors(network.forward(training[i].input), training[i].output)
  )

  network.updateWeights()
  logAccuracy(iter)
})

console.log(`Training accuracy at iter ${MAX_ITER}: ${accuracy(training)}`);
console.log(`\nTest accuracy: ${accuracy(testing)}`);
console.timeEnd('total computation time:');


