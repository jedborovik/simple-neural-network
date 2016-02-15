'use strict';

const Network = require('../src/network');
const math = require('../src/math');

const data = [{
  input: [0, 0],
  output: [0],
}, {
  input: [1, 0],
  output: [1],
}, {
  input: [0, 1],
  output: [1],
}, {
  input: [1, 1],
  output: [0],
}];

const MAX_ITERS = 40000;

const network = new Network(2, 3, 1);

for (let iter = 0; iter < MAX_ITERS; iter++) {
  const i = Math.floor(Math.random() * data.length);

  const input = data[i].input;
  const output = data[i].output;

  const h = network.forward(input);
  const error = math.arraySubtract(h, output);

  network.backward(error);
  network.updateWeights();
}

for (let i = 0; i < data.length; i++) {
  const input = data[i].input;
  const h = network.forward(input);
  console.log('%s -> %s', input, h[0]);
}
